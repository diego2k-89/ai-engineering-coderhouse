"""
El grafo del equipo: un supervisor que deriva a dos especialistas.

    START -> supervisor --enrutar--> investigador -> supervisor
                        \\----------> analista     -> supervisor
                         \\---------> sintesis     -> END

Los especialistas siempre vuelven al supervisor. Solo él decide si hace falta
otro paso o si la tarea está completa; la síntesis redacta la respuesta final
con lo que aportó cada uno.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.analyst_agent import NOMBRE as ANALISTA, crear_analista
from agents.comun import ejecutar_especialista, texto
from agents.research_agent import NOMBRE as INVESTIGADOR, crear_investigador
from agents.supervisor import crear_supervisor, enrutar, formatear, pregunta_actual
from config import crear_modelo
from state import EstadoEquipo

PROMPT_SINTESIS = """Redactá la respuesta final para el usuario a partir de las \
contribuciones del equipo.

Reglas:
- Usá solo lo que está en las contribuciones. No agregues datos propios.
- Los números salen del investigador (datos de la documentación) o del analista
  (cuentas). Indicá de qué archivo de la documentación sale cada dato.
- Si una parte de la pregunta quedó sin resolver, decilo.
- Español, breve y directo."""


def construir_grafo():
    investigador = crear_investigador()
    analista = crear_analista()
    modelo_sintesis = crear_modelo()

    async def nodo_investigador(estado: EstadoEquipo) -> dict:
        # Recibe solo la instrucción: buscar no necesita más contexto.
        return await ejecutar_especialista(
            investigador, INVESTIGADOR, estado["instruccion"], "", estado["pasos"]
        )

    async def nodo_analista(estado: EstadoEquipo) -> dict:
        # Recibe la instrucción y los datos que reportó el investigador, que
        # son lo único que necesita para calcular. Sus propios resultados
        # anteriores no se le pasan.
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
                    f"Motivo del cierre según el supervisor: {ultima['razon']}"
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
    grafo.add_node("sintesis", nodo_sintesis)

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges("supervisor", enrutar)
    grafo.add_edge(INVESTIGADOR, "supervisor")
    grafo.add_edge(ANALISTA, "supervisor")
    grafo.add_edge("sintesis", END)

    return grafo.compile()
