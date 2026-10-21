"""
La API: recibe tareas, las encola y devuelve un job_id sin esperar al grafo.

    POST /tasks                 -> 202 {"job_id", "estado": "PENDING"}   (milisegundos)
    GET  /tasks/{job_id}        -> el estado del job; 404 si no existe
    POST /tasks/{job_id}/approve -> decisión humana para un job pausado (hitl.py)
    GET  /health                -> Redis responde y el worker está vivo

Ningún endpoint corre el grafo. Todos hacen como mucho dos operaciones contra
Redis, que son async, así que el event loop nunca queda bloqueado esperando al
modelo.

Uso:
    uvicorn app.main:app
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app import jobs, worker
from app.config import REDIS_URL
from app.hitl import router as router_hitl
from app.observability import verificar_observabilidad

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
for ruidoso in ("httpx", "httpx2", "httpcore", "anthropic", "urllib3"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

logger = logging.getLogger("api")


class PedidoTarea(BaseModel):
    pregunta: str = Field(min_length=3, max_length=2000)


def crear_app(redis: Redis | None = None, grafo=None) -> FastAPI:
    """Arma la aplicación.

    Sin argumentos se conecta a Redis y construye el grafo real con el
    checkpointer de Redis. Los tests le pasan un Redis falso y un grafo falso,
    para probar la API sin API keys ni servidores.
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

        from app.graph import construir_grafo

        verificar_observabilidad()
        app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
        await app.state.redis.ping()
        logger.info("Redis conectado en %s", REDIS_URL)

        # El checkpointer abre su propia conexión. Al entrar al `async with`
        # crea los índices que necesita en Redis, si no existen.
        async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
            app.state.grafo = construir_grafo(checkpointer)
            async with _worker_en_segundo_plano(app):
                yield
        await app.state.redis.aclose()

    app = FastAPI(title="API de agentes - PE7", lifespan=lifespan)
    app.include_router(router_hitl)

    @app.post("/tasks", status_code=status.HTTP_202_ACCEPTED)
    async def crear_tarea(pedido: PedidoTarea, request: Request) -> dict:
        redis = request.app.state.redis
        job_id = str(uuid.uuid4())
        # Primero el registro, después la cola (ver jobs.crear).
        await jobs.crear(redis, job_id, pedido.pregunta)
        await jobs.encolar(redis, job_id, "iniciar")
        logger.info("[%s] PENDING: %s", job_id, pedido.pregunta[:80])
        return {"job_id": job_id, "estado": jobs.Estado.PENDING}

    @app.get("/tasks/{job_id}")
    async def ver_tarea(job_id: str, request: Request) -> dict:
        job = await jobs.leer(request.app.state.redis, job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
        return job

    @app.get("/health")
    async def health(request: Request) -> dict:
        tarea = request.app.state.tarea_worker
        return {
            "redis": await request.app.state.redis.ping(),
            "worker": not tarea.done(),
        }

    return app


@asynccontextmanager
async def _worker_en_segundo_plano(app: FastAPI):
    """Arranca el worker como tarea de asyncio y lo cancela al apagar la API."""
    app.state.tarea_worker = asyncio.create_task(worker.correr(app.state.redis, app.state.grafo))
    try:
        yield
    finally:
        app.state.tarea_worker.cancel()
        try:
            await app.state.tarea_worker
        except asyncio.CancelledError:
            pass


app = crear_app()
