"""
Prueba del agente: razonamiento multi-paso, reintento ante errores y memoria por
thread_id. Imprime la traza ReAct y la guarda en trazas/.

Uso:
    python main.py                                  # corre los escenarios de prueba
    python main.py --preguntar sesion-1 "¿y el primero?"
        # una pregunta suelta sobre un thread existente: muestra que la memoria
        # sobrevive a reiniciar el programa, porque está en SQLite. Se registra
        # en trazas/persistencia.log

Los escenarios borran la base de checkpoints al empezar, para que la traza sea
reproducible. --preguntar no la borra.
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agente import construir_grafo

BASE_DIR = Path(__file__).parent
CHECKPOINTS = BASE_DIR / "checkpoints.sqlite3"
TRAZAS = BASE_DIR / "trazas"

# Techo de pasos por invocación. Una pregunta normal usa 5 (modelo,
# herramienta, modelo, herramienta, modelo); el caso con reintento, 7. Si el
# agente entrara en un bucle, se corta acá en vez de seguir gastando API.
RECURSION_LIMIT = 10

ESCENARIOS: list[tuple[str, str, str]] = [
    # (thread_id, pregunta, qué demuestra)
    ("sesion-1", "¿Cuántos pedidos tuvo Distribuidora Norte y cuál fue el total?",
     "multi-paso: buscar el cliente por nombre y después sus pedidos"),
    ("sesion-1", "¿Y el último?",
     "memoria: mismo thread_id, el agente sabe de qué cliente se habla"),
    ("sesion-2", "¿Cuánto gastó en total Tecnosurr?",
     "ciclo de retorno: el nombre tiene un error de tipeo y la herramienta sugiere el correcto"),
    ("sesion-2", "¿Cuáles son los pedidos de Acme?",
     "aclaración: hay dos clientes Acme y el agente pregunta cuál"),
    ("sesion-2", "El de Rosario.",
     "memoria: resuelve la aclaración con el contexto anterior"),
    ("sesion-3", "¿Y el último?",
     "aislamiento: thread nuevo, el agente no tiene contexto y lo pide"),
]

logger = logging.getLogger("agente")


def configurar_logs(nombre_log: str) -> None:
    TRAZAS.mkdir(exist_ok=True)
    formato = logging.Formatter("%(message)s")

    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    archivo = logging.FileHandler(TRAZAS / nombre_log, mode="w", encoding="utf-8")
    archivo.setFormatter(formato)

    logger.setLevel(logging.INFO)
    logger.handlers = [consola, archivo]
    for ruidoso in ("httpx", "httpx2", "httpcore", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def _texto(mensaje: AIMessage) -> str:
    """Texto de un mensaje del modelo.

    Cuando Claude pide una herramienta, el contenido viene como una lista de
    bloques (texto y tool_use). El texto es el razonamiento que escribe antes de
    la llamada.
    """
    if isinstance(mensaje.content, str):
        return mensaje.content.strip()
    return " ".join(
        bloque.get("text", "")
        for bloque in mensaje.content
        if isinstance(bloque, dict) and bloque.get("type") == "text"
    ).strip()


def _resultado(mensaje: ToolMessage) -> Any:
    try:
        return json.loads(mensaje.content)
    except (TypeError, ValueError):
        return mensaje.content


async def preguntar(grafo, thread_id: str, pregunta: str) -> dict[str, Any]:
    """Corre una pregunta, imprime cada paso del ciclo ReAct y lo devuelve."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    pasos: list[dict[str, Any]] = []
    respuesta = ""
    inicio = time.perf_counter()

    logger.info('\n[%s] Usuario: "%s"', thread_id, pregunta)

    try:
        async for actualizacion in grafo.astream(
            {"messages": [HumanMessage(pregunta)]}, config, stream_mode="updates"
        ):
            for nodo, cambios in actualizacion.items():
                for mensaje in cambios.get("messages", []):
                    if isinstance(mensaje, AIMessage) and mensaje.tool_calls:
                        if razonamiento := _texto(mensaje):
                            logger.info("→ El agente razona: %s", razonamiento)
                            pasos.append({"tipo": "razonamiento", "texto": razonamiento})
                        for llamada in mensaje.tool_calls:
                            argumentos = ", ".join(f"{k}={v!r}" for k, v in llamada["args"].items())
                            logger.info("→ El agente decide usar: %s(%s)", llamada["name"], argumentos)
                            pasos.append({"tipo": "llamada", "herramienta": llamada["name"],
                                          "argumentos": llamada["args"]})

                    elif isinstance(mensaje, ToolMessage):
                        resultado = _resultado(mensaje)
                        logger.info("→ La herramienta devuelve: %s",
                                    json.dumps(resultado, ensure_ascii=False))
                        pasos.append({"tipo": "resultado", "herramienta": mensaje.name,
                                      "resultado": resultado})

                    elif isinstance(mensaje, AIMessage):
                        respuesta = _texto(mensaje)
                        pasos.append({"tipo": "respuesta", "texto": respuesta})

    except GraphRecursionError:
        respuesta = f"Cortado por recursion_limit={RECURSION_LIMIT}"
        pasos.append({"tipo": "corte", "texto": respuesta})

    segundos = time.perf_counter() - inicio
    llamadas = sum(1 for p in pasos if p["tipo"] == "llamada")
    logger.info('Respuesta: "%s"', respuesta)
    logger.info("(%d llamadas a herramientas, %.2fs)", llamadas, segundos)

    return {
        "thread_id": thread_id,
        "pregunta": pregunta,
        "pasos": pasos,
        "respuesta": respuesta,
        "llamadas_herramientas": llamadas,
        "segundos": round(segundos, 2),
    }


async def correr_escenarios() -> None:
    CHECKPOINTS.unlink(missing_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS)) as checkpointer:
        grafo = construir_grafo(checkpointer)
        resultados = []

        for thread_id, pregunta, demuestra in ESCENARIOS:
            logger.info("\n" + "=" * 72 + f"\n  {demuestra}\n" + "=" * 72)
            resultado = await preguntar(grafo, thread_id, pregunta)
            resultados.append({"demuestra": demuestra, **resultado})

    destino = TRAZAS / "traza_react.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("\nTraza guardada en %s y %s", destino.name, "traza_react.log")


async def pregunta_suelta(thread_id: str, pregunta: str) -> None:
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINTS)) as checkpointer:
        await preguntar(construir_grafo(checkpointer), thread_id, pregunta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba del agente ReAct")
    parser.add_argument("--preguntar", nargs=2, metavar=("THREAD_ID", "PREGUNTA"))
    args = parser.parse_args()

    # Cada modo escribe su propio log, para que una pregunta suelta no pise la
    # traza de los escenarios.
    if args.preguntar:
        configurar_logs("persistencia.log")
        asyncio.run(pregunta_suelta(*args.preguntar))
    else:
        configurar_logs("traza_react.log")
        asyncio.run(correr_escenarios())


if __name__ == "__main__":
    main()
