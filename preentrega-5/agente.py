"""
El grafo del agente: un ciclo ReAct entre el modelo y las herramientas.

    START -> modelo -> ¿pidió herramientas? -- sí --> herramientas -> modelo
                                            \\-- no --> END

El modelo decide solo si necesita una herramienta: no hay ningún if/else que
elija la ruta. Si su respuesta trae tool_calls, tools_condition manda el flujo
al nodo de herramientas, y el resultado vuelve al modelo. El ciclo se repite
hasta que el modelo responde sin pedir herramientas.

La memoria la da el checkpointer: guarda el estado después de cada paso, por
thread_id. Invocar el grafo con el mismo thread_id retoma la conversación.
"""

import operator
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from herramientas import HERRAMIENTAS

load_dotenv()

# Mensajes que se le pasan al modelo en cada turno, como máximo. El estado
# completo se sigue guardando en el checkpointer; lo que se recorta es solo lo
# que viaja al modelo, para que el contexto no crezca sin límite.
MAX_MENSAJES_CONTEXTO = 20

PROMPT_SISTEMA = """Sos un asistente que responde consultas sobre clientes y \
pedidos usando las herramientas disponibles.

Reglas:
- Usá las herramientas para obtener cualquier dato: no inventes clientes,
  pedidos ni montos.
- Si una herramienta devuelve un error, leelo: si sugiere una corrección clara,
  volvé a intentar; si hay ambigüedad, preguntale al usuario.
- Los montos están en pesos. Mostralos con separador de miles ($14.500).
- Respondé en español, de forma breve y directa."""


class EstadoAgente(MessagesState):
    """Estado del grafo.

    Hereda de MessagesState, que ya trae la lista `messages` con su reducer
    (add_messages): cada nodo devuelve solo los mensajes nuevos y LangGraph los
    agrega al historial en lugar de reemplazarlo.

    `llamadas_herramientas` usa operator.add como reducer: cada paso suma, no
    pisa el valor anterior.
    """

    llamadas_herramientas: Annotated[int, operator.add]


def crear_modelo():
    """Modelo de Anthropic con las herramientas vinculadas (bind_tools).

    bind_tools le describe las herramientas al modelo a partir de sus nombres,
    tipos y docstrings. Es lo que le permite pedir una herramienta en vez de
    responder con texto.
    """
    modelo = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        temperature=0,
        max_tokens=1024,
    )
    return modelo.bind_tools(HERRAMIENTAS)


def construir_grafo(checkpointer):
    """Arma y compila el grafo con el checkpointer recibido."""
    modelo = crear_modelo()

    async def nodo_modelo(estado: EstadoAgente) -> dict:
        # Se recorta lo que viaja al modelo, no lo que se guarda. start_on="human"
        # garantiza que el recorte no empiece por una respuesta de herramienta
        # suelta, sin el pedido que la originó: Anthropic la rechazaría.
        historial: list[BaseMessage] = trim_messages(
            estado["messages"],
            max_tokens=MAX_MENSAJES_CONTEXTO,
            token_counter=len,
            strategy="last",
            start_on="human",
        )
        respuesta = await modelo.ainvoke([SystemMessage(PROMPT_SISTEMA), *historial])
        return {
            "messages": [respuesta],
            "llamadas_herramientas": len(respuesta.tool_calls),
        }

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("modelo", nodo_modelo)
    # handle_tool_errors=True: si una herramienta igual lanzara una excepción,
    # el error le llega al modelo como mensaje en vez de cortar el grafo.
    grafo.add_node("herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True))

    grafo.add_edge(START, "modelo")
    # tools_condition mira el último mensaje: si trae tool_calls devuelve
    # "tools", si no devuelve END. El diccionario traduce "tools" al nombre del
    # nodo de herramientas de este grafo.
    grafo.add_conditional_edges(
        "modelo", tools_condition, {"tools": "herramientas", END: END}
    )
    grafo.add_edge("herramientas", "modelo")

    return grafo.compile(checkpointer=checkpointer)
