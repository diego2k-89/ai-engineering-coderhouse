"""
Configuración del sistema. Todo sale de variables de entorno.

Ningún nombre de índice, namespace, modelo ni URL está escrito en el código: si
cambia algo, se cambia el .env (o el environment de docker-compose). Los valores
por defecto de acá son los de desarrollo y están documentados en .env.example.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# --- Modelo de lenguaje (supervisor, especialistas y síntesis) ---
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# Vacía = no se manda. Los modelos más nuevos de Anthropic rechazan el parámetro
# (error 400 "temperature: Extra inputs are not permitted").
TEMPERATURA = os.getenv("ANTHROPIC_TEMPERATURE", "").strip()

# --- RAG: Pinecone y embeddings ---
INDEX_NAME = os.getenv("INDEX_NAME", "httpx-docs")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "httpx-docs")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Pesos del EnsembleRetriever (BM25, vectorial). Ver rag/recuperador.py.
PESO_BM25 = float(os.getenv("RAG_PESO_BM25", "0.3"))
PESO_VECTORIAL = float(os.getenv("RAG_PESO_VECTORIAL", "0.7"))

# --- Redis, API y worker ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PRESUPUESTO_DERIVACIONES = int(os.getenv("PRESUPUESTO_DERIVACIONES", "2"))
MAX_CONCURRENCIA = int(os.getenv("MAX_CONCURRENCIA", "5"))
TIMEOUT_JOB_SEGUNDOS = int(os.getenv("TIMEOUT_JOB_SEGUNDOS", "180"))


def requerir(variable: str) -> str:
    """Devuelve una variable del entorno o corta con un mensaje claro.

    El valor nunca se imprime: solo se informa qué variable falta.
    """
    valor = os.getenv(variable, "").strip()
    if not valor:
        raise RuntimeError(
            f"Falta {variable}. Copiar .env.example como .env y completar las keys."
        )
    return valor


def crear_modelo(max_tokens: int = 1024):
    """Modelo de Anthropic. Las llamadas se hacen con ainvoke: siempre async."""
    from langchain_anthropic import ChatAnthropic

    requerir("ANTHROPIC_API_KEY")
    opcionales = {"temperature": float(TEMPERATURA)} if TEMPERATURA else {}
    return ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=max_tokens, **opcionales)


def crear_embeddings():
    """Modelo de embeddings de Google. Se usa con aembed_query / aembed_documents."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    requerir("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIM
    )


def configurar_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "anthropic", "urllib3", "aiohttp"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
