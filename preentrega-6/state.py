"""
Estado compartido del equipo de agentes.

Todos los nodos leen y escriben este estado. Además de los mensajes, registra
qué agente aportó qué, para que el supervisor decida con esa información y para
que la traza muestre el recorrido completo.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import MessagesState


class Contribucion(TypedDict):
    """Lo que aportó un especialista en un paso."""

    paso: int
    agente: str
    instruccion: str
    herramientas: list[str]
    resultado: str


class EstadoEquipo(MessagesState):
    """Estado del grafo.

    - messages: la pregunta del usuario, el resultado de cada especialista y la
      respuesta final. Viene de MessagesState, con su reducer add_messages.
    - siguiente_agente: a quién deriva el supervisor en este paso.
    - instruccion: la tarea concreta para ese agente. Es lo único que el
      especialista recibe del supervisor.
    - contribuciones: historial de aportes. operator.add hace que cada
      especialista agregue el suyo sin pisar los anteriores.
    - decisiones: cada decisión del supervisor con su razón, para la traza.
    - pasos: cuántas veces decidió el supervisor. Sirve de freno.
    - tarea_completa: lo marca la síntesis al terminar.
    """

    siguiente_agente: str
    instruccion: str
    contribuciones: Annotated[list[Contribucion], operator.add]
    decisiones: Annotated[list[dict], operator.add]
    pasos: int
    tarea_completa: bool
