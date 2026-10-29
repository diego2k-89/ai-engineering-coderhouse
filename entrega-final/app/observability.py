"""
Observabilidad con LangSmith.

LangGraph y LangChain ya emiten trazas: con LANGSMITH_TRACING=true y una
LANGSMITH_API_KEY en el entorno, cada ejecución del grafo llega a LangSmith como
una traza con un span por nodo, por llamada al modelo y por herramienta. La
búsqueda híbrida también aparece desglosada (el EnsembleRetriever y sus dos
retrievers, BM25 y Pinecone), porque los tres son retrievers de LangChain. En la
pre-entrega 7 la búsqueda en Pinecone necesitaba un @traceable aparte.

Este módulo hace dos cosas:

- `verificar_observabilidad()`: al arrancar, informa si el tracing está activo y
  a qué proyecto va. Sin la key, la API funciona igual pero sin trazas, y es
  mejor saberlo en el log que descubrirlo en un dashboard vacío.
- `config_de_traza()`: arma la parte de la config de LangGraph que etiqueta cada
  ejecución, para poder filtrar en el dashboard por job o por versión de prompt.
"""

import logging
import os

logger = logging.getLogger("api.observabilidad")

# Versión de los prompts del supervisor, los especialistas y la síntesis. Se
# manda como metadato en cada traza: si cambia un prompt, se sube este número y
# en LangSmith se pueden comparar las corridas de una versión y de la otra.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "final-v1")


def verificar_observabilidad() -> bool:
    activo = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    tiene_key = bool(os.getenv("LANGSMITH_API_KEY", "").strip())
    proyecto = os.getenv("LANGSMITH_PROJECT", "default")

    if activo and tiene_key:
        logger.info("LangSmith activo: las trazas van al proyecto '%s'", proyecto)
        return True
    if activo:
        logger.warning("LANGSMITH_TRACING=true pero falta LANGSMITH_API_KEY: no se envían trazas")
    else:
        logger.warning("LangSmith desactivado (LANGSMITH_TRACING no es 'true')")
    return False


def config_de_traza(job_id: str, tipo: str) -> dict:
    """Nombre, etiquetas y metadatos que acompañan la traza de un job.

    `tipo` es "iniciar" o "reanudar": una tarea pausada por aprobación humana
    queda como dos trazas, una por cada tramo, unidas por el mismo job_id.
    """
    return {
        "run_name": f"tarea-{tipo}",
        "tags": ["entrega-final", tipo],
        "metadata": {"job_id": job_id, "prompt_version": PROMPT_VERSION},
    }
