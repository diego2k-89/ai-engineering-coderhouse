"""
Lo que comparten los dos especialistas: cómo se los invoca y cómo se registra
su aporte en el estado.
"""

import warnings

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.schemas import Contribucion, a_dict

# create_react_agent figura como deprecado desde LangGraph 1.0 (se movió a
# langchain.agents.create_agent), pero sigue funcionando y es el que pide la
# consigna. El aviso se silencia para que no ensucie la traza.
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

# Techo de pasos dentro de cada especialista (modelo y herramientas cuentan
# por separado). Una búsqueda o un cálculo normal usan 3 o 5.
RECURSION_LIMIT_ESPECIALISTA = 8


def texto(mensaje: AIMessage) -> str:
    """Texto de un mensaje del modelo, sin los bloques de tool_use."""
    if isinstance(mensaje.content, str):
        return mensaje.content.strip()
    return " ".join(
        b.get("text", "") for b in mensaje.content if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


async def ejecutar_especialista(
    agente: CompiledStateGraph,
    nombre: str,
    instruccion: str,
    contexto: str,
    paso: int,
) -> dict:
    """Corre un especialista y devuelve la actualización del estado.

    El especialista arranca con un historial nuevo: recibe solo la instrucción
    del supervisor y, si hace falta, el contexto que le pasa el nodo. No ve la
    conversación completa ni los razonamientos de otros agentes. Eso evita la
    contaminación de contexto.
    """
    pedido = instruccion if not contexto else f"{instruccion}\n\nDatos disponibles:\n{contexto}"

    try:
        resultado = await agente.ainvoke(
            {"messages": [HumanMessage(pedido)]},
            {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA},
        )
        mensajes = resultado["messages"]
        herramientas = [m.name for m in mensajes if isinstance(m, ToolMessage)]
        respuesta = texto(mensajes[-1])
    except GraphRecursionError:
        # El especialista no llegó a una respuesta. Se informa como resultado
        # para que el supervisor decida, en vez de cortar todo el grafo.
        herramientas = []
        respuesta = (
            f"No terminó la tarea: alcanzó el límite de {RECURSION_LIMIT_ESPECIALISTA} pasos."
        )

    # El aporte se valida con Pydantic antes de escribirse en el estado: un
    # agente con nombre equivocado o un paso inválido fallan acá, no cuando el
    # supervisor lo lea.
    contribucion = Contribucion(
        paso=paso, agente=nombre, instruccion=instruccion,
        herramientas=herramientas, resultado=respuesta,
    )
    return {
        "messages": [AIMessage(respuesta, name=nombre)],
        "contribuciones": [a_dict(contribucion)],
    }
