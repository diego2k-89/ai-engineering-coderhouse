"""
Piezas falsas para probar la API sin Redis, sin API keys y sin red.

- Redis: `fakeredis.FakeAsyncRedis`, que implementa los mismos comandos en
  memoria (SET, GET, RPUSH, BLPOP).
- El grafo: `GrafoFalso`, que tiene los dos métodos que usa el worker
  (`ainvoke` y `aget_state`) y se puede configurar para terminar bien, fallar o
  pausarse como lo haría el nodo de aprobación.
"""

import asyncio
from types import SimpleNamespace

import fakeredis
import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command

from app.main import crear_app


class GrafoFalso:
    def __init__(self, modo: str = "ok", demora: float = 0.0):
        self.modo = modo
        self.demora = demora
        self.entradas: list = []
        self.pausado = False

    async def ainvoke(self, entrada, config):
        self.entradas.append(entrada)
        await asyncio.sleep(self.demora)
        if self.modo == "falla":
            raise RuntimeError("Pinecone no responde")
        if self.modo == "pausa":
            # Primer tramo: se pausa. Al reanudar con Command, termina.
            self.pausado = not isinstance(entrada, Command)

    async def aget_state(self, config):
        if self.pausado:
            interrupcion = SimpleNamespace(value={"motivo": "presupuesto superado"})
            return SimpleNamespace(next=("aprobacion",), tasks=(SimpleNamespace(interrupts=(interrupcion,)),))
        return SimpleNamespace(
            next=(),
            tasks=(),
            values={
                "messages": [SimpleNamespace(content="respuesta de prueba")],
                "contribuciones": [{"agente": "investigador"}, {"agente": "analista"}],
            },
        )


@pytest.fixture
def armar_cliente():
    """Devuelve una función que arma un TestClient con un grafo falso dado.

    El `with` del TestClient corre el lifespan de la app, así que el worker
    arranca de verdad y procesa los jobs en segundo plano.
    """
    clientes = []

    def _armar(grafo: GrafoFalso) -> TestClient:
        redis = fakeredis.FakeAsyncRedis(decode_responses=True)
        cliente = TestClient(crear_app(redis=redis, grafo=grafo))
        cliente.__enter__()
        clientes.append(cliente)
        return cliente

    yield _armar
    for cliente in clientes:
        cliente.__exit__(None, None, None)
