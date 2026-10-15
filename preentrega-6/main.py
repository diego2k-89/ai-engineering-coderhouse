"""
Corre las preguntas de demostración, imprime el recorrido de delegación y lo
guarda en trazas/.

Uso:
    python main.py                      # las preguntas de demostración
    python main.py "otra pregunta"      # una pregunta suelta
    python main.py --mermaid            # escribe grafo.mmd con el diagrama
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

BASE_DIR = Path(__file__).parent
TRAZAS = BASE_DIR / "trazas"

# Techo de pasos del grafo principal. Cada ida y vuelta a un especialista son
# 2 pasos; con MAX_PASOS=6 decisiones y la síntesis, el peor caso son 13. El
# límite de decisiones corta antes; este es el respaldo.
RECURSION_LIMIT = 20

PREGUNTAS: list[tuple[str, str]] = [
    (
        "Un servicio necesita mantener 250 conexiones simultáneas con un "
        "httpx.Client creado sin parámetros. ¿Alcanza el límite de conexiones "
        "por defecto? Si no, ¿cuántas faltan y qué porcentaje del pedido cubre?",
        "delegación completa: investigador busca el dato, analista calcula",
    ),
    (
        "¿Qué excepción lanza httpx cuando no consigue una conexión del pool a tiempo?",
        "solo investigación: el supervisor no deriva al analista",
    ),
]

logger = logging.getLogger("equipo")


def configurar_logs(nombre: str) -> None:
    TRAZAS.mkdir(exist_ok=True)
    formato = logging.Formatter("%(message)s")
    consola = logging.StreamHandler()
    consola.setFormatter(formato)
    archivo = logging.FileHandler(TRAZAS / f"{nombre}.log", mode="w", encoding="utf-8")
    archivo.setFormatter(formato)
    logger.setLevel(logging.INFO)
    logger.handlers = [consola, archivo]
    for ruidoso in ("httpx", "httpx2", "httpcore", "anthropic", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def _recortar(texto: str, largo: int = 300) -> str:
    texto = " ".join(texto.split())
    return texto if len(texto) <= largo else texto[:largo] + "..."


async def resolver(grafo, pregunta: str) -> dict[str, Any]:
    """Corre una pregunta y registra cada paso del recorrido."""
    logger.info('\nUsuario: "%s"', pregunta)
    inicio = time.perf_counter()
    estado_inicial = {"messages": [HumanMessage(pregunta)], "pasos": 0, "contribuciones": [], "decisiones": []}
    recorrido: list[dict[str, Any]] = []
    respuesta = ""

    try:
        async for actualizacion in grafo.astream(
            estado_inicial, {"recursion_limit": RECURSION_LIMIT}, stream_mode="updates"
        ):
            for nodo, cambios in actualizacion.items():
                if nodo == "supervisor":
                    d = cambios["decisiones"][-1]
                    logger.info("\n[supervisor %d] → %s", d["paso"], d["siguiente"])
                    logger.info("   razón: %s", d["razon"])
                    if d["instruccion"]:
                        logger.info("   instrucción: %s", d["instruccion"])
                    recorrido.append({"nodo": nodo, **d})

                elif nodo == "sintesis":
                    respuesta = cambios["messages"][-1].content
                    recorrido.append({"nodo": nodo, "respuesta": respuesta})

                else:
                    c = cambios["contribuciones"][-1]
                    logger.info("[%s] herramientas: %s", nodo, ", ".join(c["herramientas"]) or "ninguna")
                    logger.info("   resultado: %s", _recortar(c["resultado"]))
                    recorrido.append({"nodo": nodo, **c})

    except GraphRecursionError:
        respuesta = f"Cortado por recursion_limit={RECURSION_LIMIT}"
        recorrido.append({"nodo": "corte", "texto": respuesta})

    segundos = time.perf_counter() - inicio
    logger.info("\nRespuesta final:\n%s", respuesta)
    logger.info("(%.2fs)", segundos)

    return {
        "pregunta": pregunta,
        "recorrido": recorrido,
        "respuesta": respuesta,
        "agentes_usados": [p["nodo"] for p in recorrido if p["nodo"] not in ("supervisor", "sintesis")],
        "segundos": round(segundos, 2),
    }


async def correr(preguntas: list[tuple[str, str]], nombre: str) -> None:
    from graph import construir_grafo

    grafo = construir_grafo()
    resultados = []
    for pregunta, demuestra in preguntas:
        logger.info("\n" + "=" * 72 + f"\n  {demuestra}\n" + "=" * 72)
        resultados.append({"demuestra": demuestra, **await resolver(grafo, pregunta)})

    destino = TRAZAS / f"{nombre}.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("\nTraza guardada en trazas/%s.json y trazas/%s.log", nombre, nombre)


def escribir_mermaid() -> None:
    """Escribe el diagrama del grafo en formato Mermaid.

    draw_mermaid_png() necesita conexión a un servicio externo (mermaid.ink).
    El texto Mermaid sale del grafo sin red y GitHub lo dibuja en el README.
    """
    from graph import construir_grafo

    destino = BASE_DIR / "grafo.mmd"
    destino.write_text(construir_grafo().get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Diagrama escrito en {destino.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Equipo supervisor + especialistas")
    parser.add_argument("pregunta", nargs="?", help="una pregunta suelta")
    parser.add_argument("--mermaid", action="store_true", help="escribe grafo.mmd")
    args = parser.parse_args()

    if args.mermaid:
        escribir_mermaid()
        return

    # Una pregunta suelta escribe su propia traza, para no pisar la de la demo.
    nombre = "pregunta_suelta" if args.pregunta else "ejecucion"
    configurar_logs(nombre)
    preguntas = [(args.pregunta, "pregunta suelta")] if args.pregunta else PREGUNTAS
    asyncio.run(correr(preguntas, nombre))


if __name__ == "__main__":
    main()
