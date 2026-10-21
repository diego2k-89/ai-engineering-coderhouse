"""
El orquestador de la pre-entrega 6, con dos cambios:

1. Un nodo de aprobación humana (`aprobacion`, en hitl.py) entre el supervisor y
   los especialistas, para cuando la tarea supera el presupuesto.
2. El checkpointer se recibe de afuera. La API le pasa un `AsyncRedisSaver`, así
   el estado de cada job vive en Redis y una tarea pausada sobrevive a un
   reinicio.

    START -> supervisor --enrutar--> investigador -> supervisor
                        \\----------> analista     -> supervisor
                         \\---------> aprobacion   --> investigador / analista / sintesis
                          \\--------> sintesis     -> END
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agents.analyst_agent import NOMBRE as ANALISTA, crear_analista
from app.agents.comun import ejecutar_especialista, texto
from app.agents.research_agent import NOMBRE as INVESTIGADOR, crear_investigador
from app.agents.supervisor import crear_supervisor, formatear, pregunta_actual
from app.config import crear_modelo
from app.hitl import nodo_aprobacion, requiere_aprobacion
from app.state import EstadoEquipo

PROMPT_SINTESIS = """Redactá la respuesta final para el usuario a partir de las \
contribuciones del equipo.

Reglas:
- Usá solo lo que está en las contribuciones. No agregues datos propios.
- Los números salen del investigador (datos de la documentación) o del analista
  (cuentas). Indicá de qué archivo de la documentación sale cada dato.
- Si una parte de la pregunta quedó sin resolver, decilo.
- Español, breve y directo."""


def enrutar(estado: EstadoEquipo) -> Literal["investigador", "analista", "aprobacion", "sintesis"]:
    """Arista condicional del supervisor.

    Igual que en la pre-entrega 6, salvo que una derivación que supera el
    presupuesto pasa antes por el nodo de aprobación.
    """
    siguiente = estado["siguiente_agente"]
    if siguiente == "FINALIZAR":
        return "sintesis"
    if requiere_aprobacion(estado):
        return "aprobacion"
    return siguiente


def despues_de_aprobacion(estado: EstadoEquipo) -> Literal["investigador", "analista", "sintesis"]:
    """Si se aprobó, sigue al especialista que pidió el supervisor; si no, cierra."""
    siguiente = estado["siguiente_agente"]
    return "sintesis" if siguiente == "FINALIZAR" else siguiente


def construir_grafo(checkpointer):
    investigador = crear_investigador()
    analista = crear_analista()
    modelo_sintesis = crear_modelo()

    async def nodo_investigador(estado: EstadoEquipo) -> dict:
        return await ejecutar_especialista(
            investigador, INVESTIGADOR, estado["instruccion"], "", estado["pasos"]
        )

    async def nodo_analista(estado: EstadoEquipo) -> dict:
        datos = [c for c in estado["contribuciones"] if c["agente"] == INVESTIGADOR]
        contexto = "\n\n".join(c["resultado"] for c in datos)
        return await ejecutar_especialista(
            analista, ANALISTA, estado["instruccion"], contexto, estado["pasos"]
        )

    async def nodo_sintesis(estado: EstadoEquipo) -> dict:
        ultima = estado["decisiones"][-1]
        respuesta = await modelo_sintesis.ainvoke(
            [
                SystemMessage(PROMPT_SINTESIS),
                HumanMessage(
                    f"Pregunta:\n{pregunta_actual(estado)}\n\n"
                    f"Contribuciones:\n{formatear(estado['contribuciones'])}\n\n"
                    f"Motivo del cierre: {ultima['razon']}"
                ),
            ]
        )
        return {
            "messages": [AIMessage(texto(respuesta), name="sintesis")],
            "tarea_completa": True,
        }

    grafo = StateGraph(EstadoEquipo)
    grafo.add_node("supervisor", crear_supervisor())
    grafo.add_node(INVESTIGADOR, nodo_investigador)
    grafo.add_node(ANALISTA, nodo_analista)
    grafo.add_node("aprobacion", nodo_aprobacion)
    grafo.add_node("sintesis", nodo_sintesis)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar)
    grafo.add_conditional_edges("aprobacion", despues_de_aprobacion)
    grafo.add_edge(INVESTIGADOR, "supervisor")
    grafo.add_edge(ANALISTA, "supervisor")
    grafo.add_edge("sintesis", END)

    return grafo.compile(checkpointer=checkpointer)


def estado_inicial(pregunta: str) -> dict:
    return {
        "messages": [HumanMessage(pregunta)],
        "pasos": 0,
        "contribuciones": [],
        "decisiones": [],
        "presupuesto_aprobado": False,
    }
