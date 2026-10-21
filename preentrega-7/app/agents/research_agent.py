"""
Agente investigador: busca en la documentación de httpx indexada en Pinecone.

Reutiliza el índice de la pre-entrega 4 (83 fragmentos, embeddings de Google de
1536 dimensiones). Acá solo se consulta: no se indexa nada.

Su única herramienta es la búsqueda. No calcula: si la tarea pide una cuenta,
eso es trabajo del analista.
"""

import asyncio
from functools import lru_cache
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    INDEX_NAME,
    NAMESPACE,
    crear_modelo,
    requerir,
)

NOMBRE = "investigador"
TOP_K = 4

PROMPT = """Sos el investigador de un equipo. Tu trabajo es encontrar datos en \
la documentación oficial de httpx, la librería HTTP de Python.

Reglas:
- Usá buscar_documentacion para cada dato. No respondas de memoria.
- Si la primera búsqueda no trae el dato, reformulá la consulta (en inglés, que
  es el idioma de la documentación) antes de darlo por no encontrado.
- No hagas cuentas: reportá los datos tal como aparecen.
- Respondé con los datos encontrados y, para cada uno, el archivo de donde sale.
- Si no encontrás algo, decilo explícitamente. No lo completes con suposiciones."""


@lru_cache(maxsize=1)
def _recursos() -> tuple[Any, Any]:
    """Cliente de Pinecone y modelo de embeddings, creados una sola vez."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from pinecone import Pinecone

    indice = Pinecone(api_key=requerir("PINECONE_API_KEY")).Index(name=INDEX_NAME)
    requerir("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIM
    )
    return indice, embeddings


# Pinecone y el modelo de embeddings no son componentes de LangChain, así que
# LangSmith no los registra solo. @traceable crea un span propio para esta
# función, de tipo "retriever", con la consulta de entrada y los fragmentos de
# salida. Sin esto, en la traza el tiempo de la búsqueda quedaría escondido
# adentro de la herramienta.
@traceable(run_type="retriever", name="pinecone_busqueda")
def _buscar(consulta: str) -> list[dict[str, Any]]:
    indice, embeddings = _recursos()
    respuesta = indice.query(
        vector=embeddings.embed_query(consulta),
        top_k=TOP_K,
        namespace=NAMESPACE,
        include_metadata=True,
    )
    return [
        {
            "fuente": r.metadata.get("fuente"),
            "seccion": r.metadata.get("seccion"),
            "score": round(r.score, 3),
            "texto": r.metadata.get("texto"),
        }
        for r in respuesta.matches
    ]


@tool
async def buscar_documentacion(consulta: str) -> dict[str, Any]:
    """Busca en la documentación oficial de httpx y devuelve los fragmentos más
    relevantes para la consulta.

    Usar para cualquier dato sobre httpx: valores por defecto, parámetros,
    clases, excepciones, variables de entorno, configuración de clientes,
    timeouts, límites de conexiones, proxies, SSL, HTTP/2, etc.

    La documentación está en inglés: las consultas en inglés y con los nombres
    exactos (por ejemplo "max_connections default", "PoolTimeout") recuperan
    mejor.

    Devuelve {"resultados": [...]}, donde cada resultado tiene "fuente"
    (archivo), "seccion", "score" (similitud, de 0 a 1) y "texto". Si la
    búsqueda falla, devuelve {"error": "..."}.
    """
    try:
        # El SDK de Pinecone y el de embeddings son sincrónicos: se corren en
        # un hilo aparte para no bloquear el event loop.
        resultados = await asyncio.to_thread(_buscar, consulta)
    except Exception as exc:  # noqa: BLE001 - el error vuelve al modelo como dato
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"resultados": resultados}


def crear_investigador():
    return create_react_agent(
        crear_modelo(), [buscar_documentacion], prompt=PROMPT, name=NOMBRE
    )
