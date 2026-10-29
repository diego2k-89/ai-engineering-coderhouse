"""
El worker: saca jobs de la cola de Redis y corre el grafo en segundo plano.

Corre dentro del mismo proceso que la API, como una tarea de asyncio que arranca
con ella (la consigna acepta "un thread asíncrono simple en esta etapa"). Como
todo el grafo es async, mientras un job espera al modelo o a Pinecone el event
loop sigue atendiendo requests y otros jobs.

Ciclo de vida de un job:

    PENDING -> RUNNING -> DONE
                       -> FAILED            (cualquier excepción del grafo)
                       -> WAITING_APPROVAL  (pausa del nodo de aprobación)
                              |
                              v  POST /approve
                          PENDING -> RUNNING -> DONE / FAILED
"""

import asyncio
import json
import logging
import time

from langgraph.types import Command
from redis.asyncio import Redis

from app.api import jobs
from app.config import MAX_CONCURRENCIA, TIMEOUT_JOB_SEGUNDOS
from app.graph.grafo import estado_inicial
from app.observability import config_de_traza
from app.schemas import EstadoJob

logger = logging.getLogger("api.worker")

# Techo de pasos del grafo por tramo de ejecución, igual que en la pre-entrega 6.
RECURSION_LIMIT = 20

# Cuánto espera BLPOP por un job antes de volver a intentar. Tiene que ser
# menor que el socket_timeout del cliente de Redis, que en redis-py 8 es de 5 s
# por defecto. Con BLPOP también en 5 s, los dos plazos vencen a la vez, y en
# las pruebas el worker moría unos 5 s después de quedarse sin jobs.
ESPERA_COLA_SEGUNDOS = 2



def _texto_final(valores: dict) -> str:
    """La respuesta de la síntesis: el último mensaje del estado."""
    mensajes = valores.get("messages", [])
    return mensajes[-1].content if mensajes else ""


async def procesar(redis: Redis, grafo, pedido: dict) -> None:
    """Corre un tramo del grafo para un job y deja el resultado en Redis.

    Toda excepción se captura y el job pasa a FAILED con el detalle. Si no, el
    cliente que hace polling vería RUNNING para siempre.
    """
    job_id = pedido["job_id"]
    tipo = pedido["tipo"]
    config = {
        "configurable": {"thread_id": job_id},
        "recursion_limit": RECURSION_LIMIT,
        **config_de_traza(job_id, tipo),
    }
    inicio = time.perf_counter()

    try:
        job = await jobs.leer(redis, job_id)
        if job is None:
            logger.warning("[%s] el job no existe, se descarta el pedido", job_id)
            return

        await jobs.actualizar(redis, job_id, estado=EstadoJob.RUNNING)
        logger.info("[%s] RUNNING (%s)", job_id, tipo)

        if tipo == "reanudar":
            # El valor de resume es lo que devuelve interrupt() en hitl.py.
            entrada = Command(resume={"aprobado": pedido["aprobado"]})
        else:
            entrada = estado_inicial(job.pregunta)

        # Techo de tiempo por tramo: si el grafo se traba esperando una API
        # externa, el job pasa a FAILED en vez de quedar en RUNNING.
        async with asyncio.timeout(TIMEOUT_JOB_SEGUNDOS):
            await grafo.ainvoke(entrada, config)

        # Después de ainvoke, el estado guardado dice si el grafo terminó o
        # quedó pausado: si hay nodos pendientes (next), es una pausa.
        estado = await grafo.aget_state(config)
        segundos = round(time.perf_counter() - inicio, 2)

        if estado.next:
            pausa = estado.tasks[0].interrupts[0].value if estado.tasks and estado.tasks[0].interrupts else None
            await jobs.actualizar(
                redis, job_id, estado=EstadoJob.WAITING_APPROVAL, aprobacion=pausa, segundos=segundos
            )
            logger.info("[%s] WAITING_APPROVAL en %.2fs", job_id, segundos)
        else:
            await jobs.actualizar(
                redis, job_id,
                estado=EstadoJob.DONE,
                respuesta=_texto_final(estado.values),
                agentes=[c["agente"] for c in estado.values.get("contribuciones", [])],
                segundos=segundos,
            )
            logger.info("[%s] DONE en %.2fs", job_id, segundos)

    except asyncio.CancelledError:
        # La cancelación no hereda de Exception, así que el except de abajo no
        # la ve. Pasa al apagar la API con un job a medio correr: sin esto, el
        # job quedaría en RUNNING para siempre.
        segundos = round(time.perf_counter() - inicio, 2)
        logger.warning("[%s] cancelado después de %.2fs", job_id, segundos)
        await jobs.actualizar(
            redis, job_id, estado=EstadoJob.FAILED,
            error="Cancelado: la API se detuvo antes de terminar", segundos=segundos,
        )
        raise

    except Exception as exc:  # noqa: BLE001 - cualquier falla tiene que terminar en FAILED
        segundos = round(time.perf_counter() - inicio, 2)
        logger.exception("[%s] FAILED en %.2fs", job_id, segundos)
        await jobs.actualizar(
            redis, job_id, estado=EstadoJob.FAILED,
            error=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
            segundos=segundos,
        )


async def correr(redis: Redis, grafo) -> None:
    """Bucle principal: espera pedidos en la cola y lanza cada uno como tarea.

    El semáforo limita cuántos jobs corren a la vez. Las tareas en curso se
    guardan en un set para que el recolector de basura no las elimine antes de
    terminar.
    """
    semaforo = asyncio.Semaphore(MAX_CONCURRENCIA)
    en_curso: set[asyncio.Task] = set()
    logger.info("Worker escuchando la cola '%s' (hasta %d jobs en paralelo)", jobs.COLA, MAX_CONCURRENCIA)

    async def con_limite(pedido: dict) -> None:
        async with semaforo:
            await procesar(redis, grafo, pedido)

    try:
        while True:
            try:
                # BLPOP espera hasta que haya algo en la cola, sin consumir CPU.
                # Si no llega nada en ESPERA_COLA_SEGUNDOS devuelve None y el
                # bucle vuelve a esperar.
                resultado = await redis.blpop([jobs.COLA], timeout=ESPERA_COLA_SEGUNDOS)
                if resultado is None:
                    continue
                _, crudo = resultado
                tarea = asyncio.create_task(con_limite(json.loads(crudo)))
                en_curso.add(tarea)
                tarea.add_done_callback(en_curso.discard)
            except Exception:  # noqa: BLE001 - un error no puede matar al worker
                # Sin esto, una excepción terminaría la tarea del worker sin
                # dejar rastro y los jobs quedarían en PENDING para siempre.
                # La cancelación al apagar la API no entra acá: CancelledError
                # no hereda de Exception.
                logger.exception("Error en el bucle del worker; se reintenta en 1 s")
                await asyncio.sleep(1)
    finally:
        # Al apagar la API se cancelan los jobs en curso. Quedan en RUNNING en
        # Redis; su último checkpoint sigue guardado.
        for tarea in en_curso:
            tarea.cancel()
