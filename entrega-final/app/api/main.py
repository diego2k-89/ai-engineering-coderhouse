"""
La API: recibe tareas, las encola y devuelve un job_id sin esperar al grafo.

    POST /tasks                   -> 202 TareaCreada   (milisegundos)
    GET  /tasks/{job_id}          -> Job; 404 si no existe
    POST /tasks/{job_id}/approve  -> 202 TareaCreada; 409 si el job no está pausado
    GET  /health                  -> Salud: Redis responde y el worker está vivo
    GET  /docs                    -> documentación interactiva (OpenAPI)

Ningún endpoint corre el grafo: hacen como mucho dos operaciones async contra
Redis. Entradas y salidas son modelos de Pydantic (schemas.py), así que también
quedan documentadas en /docs.

Al arrancar, el `lifespan` abre los recursos que viven lo mismo que la API:
Redis, el recuperador (conexiones a Pinecone), el checkpointer y el worker.

Uso:
    uvicorn app.api.main:app
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from redis.asyncio import Redis

from app.api import jobs, worker
from app.config import REDIS_URL, configurar_logs
from app.observability import verificar_observabilidad
from app.schemas import DecisionHumana, EstadoJob, Job, PedidoTarea, Salud, TareaCreada

configurar_logs()
logger = logging.getLogger("api")


def crear_app(redis: Redis | None = None, grafo=None) -> FastAPI:
    """Arma la aplicación.

    Sin argumentos se conecta a Redis, abre el recuperador y construye el grafo
    real. Los tests le pasan un Redis y un grafo falsos, para probar la API sin
    API keys ni servidores.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if redis is not None:
            app.state.redis, app.state.grafo = redis, grafo
            async with _worker_en_segundo_plano(app):
                yield
            return

        # Imports acá: solo los necesita la app real, no la de los tests.
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        from app.graph.grafo import construir_grafo
        from app.rag.recuperador import abrir_recuperador

        verificar_observabilidad()
        app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
        await app.state.redis.ping()
        logger.info("Redis conectado en %s", REDIS_URL)

        # El checkpointer abre su propia conexión y, al entrar al `async with`,
        # crea los índices que necesita en Redis si no existen.
        async with (
            abrir_recuperador() as recuperador,
            AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer,
        ):
            app.state.grafo = construir_grafo(checkpointer, recuperador)
            async with _worker_en_segundo_plano(app):
                yield
        await app.state.redis.aclose()

    app = FastAPI(
        title="Intelligence System: httpx",
        description="Equipo de agentes (supervisor, investigador con RAG híbrido y analista) "
        "que responde preguntas técnicas sobre httpx. Las tareas se encolan y se consultan "
        "por polling; las que superan el presupuesto esperan aprobación humana.",
        lifespan=lifespan,
    )

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TareaCreada)
    async def crear_tarea(pedido: PedidoTarea, request: Request) -> TareaCreada:
        redis = request.app.state.redis
        job_id = str(uuid.uuid4())
        # Primero el registro, después la cola (ver jobs.crear).
        await jobs.crear(redis, job_id, pedido.pregunta)
        await jobs.encolar(redis, job_id, "iniciar")
        logger.info("[%s] PENDING: %s", job_id, pedido.pregunta[:80])
        return TareaCreada(job_id=job_id, estado=EstadoJob.PENDING)

    @app.get("/tasks/{job_id}", response_model=Job)
    async def ver_tarea(job_id: str, request: Request) -> Job:
        job = await jobs.leer(request.app.state.redis, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        return job

    @app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED, response_model=TareaCreada)
    async def aprobar(job_id: str, decision: DecisionHumana, request: Request) -> TareaCreada:
        """Registra la decisión humana sobre un job pausado y lo devuelve a la cola."""
        redis = request.app.state.redis
        job = await jobs.leer(redis, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        if job.estado != EstadoJob.WAITING_APPROVAL:
            raise HTTPException(status.HTTP_409_CONFLICT, f"El job está en {job.estado}, no esperando aprobación")

        await jobs.actualizar(redis, job_id, estado=EstadoJob.PENDING, aprobacion=None)
        await jobs.encolar(redis, job_id, "reanudar", aprobado=decision.aprobado)
        logger.info("[%s] decisión humana: %s", job_id, "aprobado" if decision.aprobado else "rechazado")
        return TareaCreada(job_id=job_id, estado=EstadoJob.PENDING)

    @app.get("/health", response_model=Salud)
    async def health(request: Request) -> Salud:
        return Salud(
            redis=await request.app.state.redis.ping(),
            worker=not request.app.state.tarea_worker.done(),
        )

    return app


def _avisar_si_termino(tarea: asyncio.Task) -> None:
    """Deja en el log por qué terminó el worker, si no fue al apagar la API.

    Una tarea de asyncio que nadie espera puede morir con una excepción sin que
    se vea en ningún lado. /health lo detecta, pero el motivo queda acá.
    """
    if tarea.cancelled():
        return
    if (error := tarea.exception()) is not None:
        logger.error("El worker se detuvo por un error", exc_info=error)


@asynccontextmanager
async def _worker_en_segundo_plano(app: FastAPI):
    """Arranca el worker como tarea de asyncio y lo cancela al apagar la API."""
    app.state.tarea_worker = asyncio.create_task(worker.correr(app.state.redis, app.state.grafo))
    app.state.tarea_worker.add_done_callback(_avisar_si_termino)
    try:
        yield
    finally:
        app.state.tarea_worker.cancel()
        try:
            await app.state.tarea_worker
        except asyncio.CancelledError:
            pass


app = crear_app()
