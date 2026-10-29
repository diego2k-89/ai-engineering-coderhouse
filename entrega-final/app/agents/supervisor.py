"""
Supervisor: decide a qué especialista derivar en cada paso y cuándo la tarea
está completa.

Es el único nodo que decide el recorrido. Los especialistas no se hablan entre
sí: cada uno recibe una instrucción del supervisor, la resuelve y vuelve.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import crear_modelo
from app.graph.state import EstadoEquipo
from app.schemas import Decision

logger = logging.getLogger("equipo")

# Freno del supervisor: cantidad máxima de decisiones por pregunta. Una
# pregunta que necesita los dos especialistas usa 3 (investigador, analista,
# finalizar). Al llegar al límite se pasa a la síntesis con lo que haya.
MAX_PASOS = 6

PROMPT = f"""Sos el supervisor de un equipo que responde preguntas técnicas \
sobre httpx, la librería HTTP de Python.

El equipo:
- investigador: busca datos en la documentación oficial de httpx. No hace cuentas.
- analista: hace cuentas y compara valores contra límites. No tiene acceso a la
  documentación: trabaja con los datos que reportó el investigador, que se le
  pasan automáticamente.

En cada paso tomás una sola decisión: derivar a un especialista con una
instrucción, o FINALIZAR.

Cómo escribir la instrucción:
- Concreta y autocontenida: el especialista no ve la pregunta del usuario.
- Una tarea por vez, solo de la especialidad de ese agente.
- Si es una corrección, decir qué faltó o qué estuvo mal.

Rúbrica para FINALIZAR. Tienen que cumplirse las tres:
1. Cada dato de httpx que la pregunta necesita fue reportado por el investigador,
   con el archivo de donde sale.
2. Cada cuenta o comparación que la pregunta pide la hizo el analista.
3. Las contribuciones no se contradicen entre sí ni dejan una parte de la
   pregunta sin responder.

Si falla un criterio, derivá al agente que corresponde. Si el analista usó un
valor distinto al que reportó el investigador, manda el del investigador, que
sale de la documentación: derivá al analista para que rehaga la cuenta. Si el
investigador no encontró un dato después de dos intentos con instrucciones
distintas, se puede finalizar indicando que falta. No repitas la misma
instrucción al mismo agente.
Hay un máximo de {MAX_PASOS} decisiones por pregunta."""


def pregunta_actual(estado: EstadoEquipo) -> str:
    """El último mensaje del usuario: la pregunta que se está resolviendo."""
    for mensaje in reversed(estado["messages"]):
        if isinstance(mensaje, HumanMessage):
            return mensaje.content
    return ""


def formatear(contribuciones: list[dict]) -> str:
    if not contribuciones:
        return "(todavía ninguna)"
    bloques = []
    for c in contribuciones:
        herramientas = ", ".join(c["herramientas"]) or "ninguna"
        bloques.append(
            f"[paso {c['paso']}] {c['agente']}\n"
            f"  instrucción: {c['instruccion']}\n"
            f"  herramientas usadas: {herramientas}\n"
            f"  resultado: {c['resultado']}"
        )
    return "\n\n".join(bloques)


def crear_supervisor():
    modelo = crear_modelo().with_structured_output(Decision)

    async def supervisor(estado: EstadoEquipo) -> dict:
        pasos = estado.get("pasos", 0) + 1

        if pasos > MAX_PASOS:
            decision = Decision(
                razon=f"Se alcanzó el máximo de {MAX_PASOS} decisiones.",
                siguiente="FINALIZAR",
            )
        else:
            # El supervisor ve la pregunta y los resultados de cada agente, no
            # los mensajes internos de los especialistas.
            decision = await modelo.ainvoke(
                [
                    SystemMessage(PROMPT),
                    HumanMessage(
                        f"Pregunta del usuario:\n{pregunta_actual(estado)}\n\n"
                        f"Contribuciones hasta ahora:\n"
                        f"{formatear(estado.get('contribuciones', []))}\n\n"
                        f"Decisión {pasos} de {MAX_PASOS}."
                    ),
                ]
            )

        return {
            "siguiente_agente": decision.siguiente,
            "instruccion": decision.instruccion,
            "pasos": pasos,
            "decisiones": [{"paso": pasos, **decision.model_dump()}],
        }

    return supervisor
