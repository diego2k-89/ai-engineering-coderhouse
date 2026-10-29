"""
Estado compartido del equipo de agentes.

Todos los nodos leen y escriben este estado. Además de los mensajes, registra
qué agente aportó qué, para que el supervisor decida con esa información y para
que la traza muestre el recorrido completo.
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class EstadoEquipo(MessagesState):
    """Estado del grafo.

    - messages: la pregunta, el resultado de cada especialista y la respuesta
      final. Viene de MessagesState, con su reducer add_messages.
    - siguiente_agente / instruccion: la última decisión del supervisor. Se
      pisan en cada paso.
    - contribuciones: cada aporte de un especialista, ya validado con el modelo
      Contribucion de schemas.py. operator.add agrega sin pisar.
    - decisiones: cada decisión del supervisor (modelo Decision) con su razón.
    - pasos: cuántas veces decidió el supervisor. Sirve de freno.
    - tarea_completa: lo marca la síntesis al terminar.
    - presupuesto_aprobado: lo marca el nodo de aprobación humana (hitl.py).

    Las listas guardan dicts y no los modelos de Pydantic porque es lo que el
    checkpointer serializa en Redis. La validación ocurre antes de escribirlos.
    """

    siguiente_agente: str
    instruccion: str
    contribuciones: Annotated[list[dict], operator.add]
    decisiones: Annotated[list[dict], operator.add]
    pasos: int
    tarea_completa: bool
    presupuesto_aprobado: bool
