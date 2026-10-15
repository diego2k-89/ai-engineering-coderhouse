"""
Configuración compartida por los scripts de la pre-entrega.

Lee el .env una sola vez y expone lo que usan setup_index.py, ingest.py, rag.py
y evaluate.py: rutas, nombres del índice, el modelo de embeddings y el cliente
de Pinecone.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# --- Pinecone ---
INDEX_NAME = os.getenv("INDEX_NAME", "httpx-docs")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "httpx-docs")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Embeddings ---
# gemini-embedding-001 produce 3072 dimensiones por defecto y admite reducirlas.
# El valor de acá es el que se le pide al modelo; setup_index.py mide la
# dimensión real antes de crear el índice.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


def configurar_logs() -> None:
    """Logs propios en INFO; el ruido de las librerías, en WARNING."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def requerir(variable: str) -> str:
    """Devuelve una variable del .env o corta con un mensaje claro.

    El valor nunca se imprime ni se loguea: solo se informa qué variable falta.
    """
    valor = os.getenv(variable, "").strip()
    if not valor:
        raise RuntimeError(
            f"Falta {variable} en el .env. Copiar .env.example como .env y "
            "completar las keys."
        )
    return valor


def get_embeddings():
    """Modelo de embeddings compartido por la ingesta y la consulta.

    Indexar y consultar con el mismo modelo y la misma dimensión no es
    opcional: por eso hay una sola función que lo construye y todos los scripts
    la usan.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    requerir("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIM,
    )


def get_pinecone():
    """Cliente de Pinecone autenticado con la key del .env."""
    from pinecone import Pinecone

    return Pinecone(api_key=requerir("PINECONE_API_KEY"))
