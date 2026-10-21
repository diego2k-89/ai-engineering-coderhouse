"""
Registro de los trabajos (jobs) en Redis y la cola que los conecta con el worker.

Cada job es una clave `job:<id>` con un JSON. La API lo crea, el worker lo
actualiza y el cliente lo consulta por polling. Se guarda con SET y no en
memoria del proceso: si la API se reinicia, los jobs siguen ahí.

La cola es una lista de Redis. La API agrega al final (RPUSH) y el worker saca
del principio (BLPOP), así que los jobs se atienden en el orden en que llegaron.
"""

import json
import time
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis

PREFIJO = "job:"
COLA = "cola:tareas"


class Estado(StrEnum):
    """Estados posibles de un job.

    PENDING           creado y en la cola, todavía no lo tomó el worker
    RUNNING           el worker está corriendo el grafo
    WAITING_APPROVAL  el grafo se pausó en el nodo de aprobación humana
    DONE              terminó con respuesta
    FAILED            el grafo lanzó una excepción; el detalle queda en "error"
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


async def crear(redis: Redis, job_id: str, pregunta: str) -> dict[str, Any]:
    """Guarda el job en PENDING. Se llama antes de encolarlo.

    El orden importa: si se encolara primero, un worker rápido podría tomar el
    job antes de que exista la clave, y una consulta inmediata devolvería 404.
    """
    ahora = time.time()
    job = {
        "job_id": job_id,
        "estado": Estado.PENDING,
        "pregunta": pregunta,
        "respuesta": None,
        "aprobacion": None,
        "error": None,
        "creado": ahora,
        "actualizado": ahora,
    }
    await redis.set(PREFIJO + job_id, json.dumps(job, ensure_ascii=False))
    return job


async def leer(redis: Redis, job_id: str) -> dict[str, Any] | None:
    datos = await redis.get(PREFIJO + job_id)
    return json.loads(datos) if datos else None


async def actualizar(redis: Redis, job_id: str, **cambios: Any) -> dict[str, Any]:
    """Pisa los campos indicados y la fecha de actualización."""
    job = await leer(redis, job_id) or {"job_id": job_id}
    job.update(cambios, actualizado=time.time())
    await redis.set(PREFIJO + job_id, json.dumps(job, ensure_ascii=False))
    return job


async def encolar(redis: Redis, job_id: str, tipo: str, **datos: Any) -> None:
    """Agrega un pedido para el worker.

    `tipo` es "iniciar" (correr el grafo desde el principio) o "reanudar"
    (seguir un grafo pausado, con la decisión del humano en `datos`).
    """
    await redis.rpush(COLA, json.dumps({"job_id": job_id, "tipo": tipo, **datos}))
