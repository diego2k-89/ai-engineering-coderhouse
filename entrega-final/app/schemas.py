"""
Contratos de datos del sistema, todos con Pydantic.

Hay un modelo en cada frontera donde entra o sale un dato:

- **API**: lo que recibe cada endpoint y lo que devuelve (`response_model`).
- **Herramientas de los agentes**: lo que el modelo de lenguaje les pasa
  (`args_schema`) y lo que devuelven.
- **Estado del grafo**: la decisión del supervisor y cada contribución de un
  especialista, que se validan antes de escribirse en el estado.
- **Redis**: el registro de cada job, que se valida al leerlo y al guardarlo.

Si el modelo de lenguaje alucina un argumento con el tipo equivocado, o un
cliente manda un pedido mal formado, el error aparece en el borde y con un
mensaje claro, no tres capas más adentro.
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class EstadoJob(StrEnum):
    """Estados posibles de un job.

    PENDING           creado y en la cola, todavía no lo tomó el worker
    RUNNING           el worker está corriendo el grafo
    WAITING_APPROVAL  el grafo se pausó en el nodo de aprobación humana
    DONE              terminó con respuesta
    FAILED            el grafo lanzó una excepción o superó el tiempo máximo
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


class PedidoTarea(BaseModel):
    pregunta: str = Field(min_length=3, max_length=2000, description="Pregunta sobre httpx")


class DecisionHumana(BaseModel):
    aprobado: bool


class Pausa(BaseModel):
    """Lo que el nodo de aprobación le muestra a quien tiene que decidir."""

    motivo: str
    siguiente_agente: str
    instruccion: str


class Job(BaseModel):
    """Registro de un job en Redis. Es también lo que devuelve GET /tasks/{id}."""

    job_id: str
    estado: EstadoJob
    pregunta: str
    respuesta: str | None = None
    agentes: list[str] = Field(default_factory=list)
    aprobacion: Pausa | None = None
    error: str | None = None
    segundos: float | None = None
    creado: float
    actualizado: float


class TareaCreada(BaseModel):
    job_id: str
    estado: EstadoJob


class Salud(BaseModel):
    redis: bool
    worker: bool


# ---------------------------------------------------------------------------
# Herramientas de los agentes
# ---------------------------------------------------------------------------


class BusquedaEntrada(BaseModel):
    consulta: str = Field(
        min_length=3,
        max_length=300,
        description="Qué buscar en la documentación de httpx. En inglés recupera mejor.",
    )


class Fragmento(BaseModel):
    fuente: str
    seccion: str
    texto: str


class ResultadoBusqueda(BaseModel):
    consulta: str
    resultados: list[Fragmento] = Field(default_factory=list)
    error: str | None = None


class CalculoEntrada(BaseModel):
    expresion: str = Field(
        min_length=1,
        max_length=200,
        description='Expresión aritmética con números y + - * / // % **, por ejemplo "250 - 100"',
    )


class ResultadoCalculo(BaseModel):
    expresion: str
    resultado: float | None = None
    error: str | None = None


class ComparacionEntrada(BaseModel):
    valor: float = Field(description="El valor a comparar, por ejemplo lo que se necesita")
    limite: float = Field(description="El máximo permitido, por ejemplo el valor por defecto")
    descripcion: str = Field(min_length=1, max_length=200, description="Qué se compara")


class ResultadoComparacion(BaseModel):
    descripcion: str
    valor: float
    limite: float
    supera_el_limite: bool
    diferencia: float
    porcentaje_del_limite: float | None


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------

Especialista = Literal["investigador", "analista"]


class Decision(BaseModel):
    """La decisión del supervisor en un paso.

    `razon` va primero: el modelo genera los campos en orden, así que escribe
    el porqué antes de decidir.
    """

    razon: str = Field(description="Por qué, en una o dos oraciones, citando la rúbrica")
    siguiente: Literal["investigador", "analista", "FINALIZAR"]
    instruccion: str = Field(
        default="", description="Tarea para el especialista. Vacía si es FINALIZAR"
    )


class Contribucion(BaseModel):
    """Lo que aportó un especialista en un paso.

    Se valida antes de escribirse en el estado. En el estado se guarda como dict
    (`model_dump()`), que es lo que el checkpointer serializa en Redis.
    """

    paso: int = Field(ge=1)
    agente: Especialista
    instruccion: str
    herramientas: list[str] = Field(default_factory=list)
    resultado: str


def a_dict(modelo: BaseModel) -> dict[str, Any]:
    return modelo.model_dump(mode="json")
