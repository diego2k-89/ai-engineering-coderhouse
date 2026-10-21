"""
Configuración compartida: lee el .env una vez y arma el modelo de lenguaje.

El supervisor, los especialistas y la síntesis usan el mismo modelo. Cambiarlo
es cambiar ANTHROPIC_MODEL en el .env.

Además de lo que ya tenía la pre-entrega 6, acá se leen la URL de Redis y los
dos números que controlan la API: el presupuesto de derivaciones antes de pedir
aprobación humana, y cuántas tareas procesa el worker al mismo tiempo.
"""

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- Pinecone (el índice que se armó en la pre-entrega 4) ---
INDEX_NAME = os.getenv("INDEX_NAME", "httpx-docs")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "httpx-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- API, worker y Redis ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Derivaciones a especialistas que una tarea puede hacer sin aprobación. Una
# pregunta típica usa 2 (investigador y analista). La tercera se pausa hasta
# que una persona la apruebe: ver hitl.py.
PRESUPUESTO_DERIVACIONES = int(os.getenv("PRESUPUESTO_DERIVACIONES", "2"))

# Tareas que el worker corre en paralelo. Con 5, las 5 peticiones de la prueba
# de carga arrancan juntas en vez de esperar en fila.
MAX_CONCURRENCIA = int(os.getenv("MAX_CONCURRENCIA", "5"))


def requerir(variable: str) -> str:
    """Devuelve una variable del .env o corta con un mensaje claro.

    El valor nunca se imprime: solo se informa qué variable falta.
    """
    valor = os.getenv(variable, "").strip()
    if not valor:
        raise RuntimeError(
            f"Falta {variable} en el .env. Copiar .env.example como .env y "
            "completar las keys."
        )
    return valor


def crear_modelo(max_tokens: int = 1024) -> ChatAnthropic:
    requerir("ANTHROPIC_API_KEY")
    return ChatAnthropic(model=ANTHROPIC_MODEL, temperature=0, max_tokens=max_tokens)
