"""
Registro de los jobs en Redis y la cola que los conecta con el worker.

Cada job es una clave `job:<id>` con el JSON del modelo `Job` (schemas.py). Se
valida al guardarlo y al leerlo: un registro corrupto en Redis falla con un
error claro en vez de devolverle basura al cliente.

La cola es una lista de Redis: la API agrega al final (RPUSH) y el worker saca
del principio (BLPOP), así que los jobs se atienden en el orden en que llegaron.
"""

import json
import time
from typing import Any

from redis.asyncio import Redis

from app.schemas import EstadoJob, Job

PREFIJO = "job:"
COLA = "cola:tareas"


async def crear(redis: Redis, job_id: str, pregunta: str) -> Job:
    """Guarda el job en PENDING. Se llama antes de encolarlo.

    El orden importa: si se encolara primero, un worker rápido podría tomar el
    job antes de que exista la clave, y una consulta inmediata daría 404.
    """
    ahora = time.time()
    job = Job(job_id=job_id, estado=EstadoJob.PENDING, pregunta=pregunta, creado=ahora, actualizado=ahora)
    await redis.set(PREFIJO + job_id, job.model_dump_json())
    return job


async def leer(redis: Redis, job_id: str) -> Job | None:
    datos = await redis.get(PREFIJO + job_id)
    return Job.model_validate_json(datos) if datos else None


async def actualizar(redis: Redis, job_id: str, **cambios: Any) -> Job:
    """Aplica los cambios, los valida con el modelo y guarda."""
    actual = await leer(redis, job_id)
    if actual is None:
        raise KeyError(f"No existe el job {job_id}")
    job = Job.model_validate({**actual.model_dump(), **cambios, "actualizado": time.time()})
    await redis.set(PREFIJO + job_id, job.model_dump_json())
    return job


async def encolar(redis: Redis, job_id: str, tipo: str, **datos: Any) -> None:
    """Agrega un pedido para el worker.

    `tipo` es "iniciar" (correr el grafo desde el principio) o "reanudar"
    (seguir un grafo pausado, con la decisión del humano en `datos`).
    """
    await redis.rpush(COLA, json.dumps({"job_id": job_id, "tipo": tipo, **datos}))
