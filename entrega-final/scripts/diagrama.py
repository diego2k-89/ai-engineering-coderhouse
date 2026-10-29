"""
Genera el diagrama del grafo de agentes en Mermaid, a partir del grafo real.

El diagrama del README sale de acá: si cambia un nodo o una arista, se vuelve a
correr y el README no queda desactualizado.

No llama al modelo ni a Pinecone ni a Redis: construir el grafo solo arma los
objetos. Por eso alcanza con una key de mentira si no hay .env.

Uso:
    python scripts/diagrama.py
"""

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("ANTHROPIC_API_KEY", "solo-para-dibujar")

from app.graph.grafo import construir_grafo  # noqa: E402

destino = RAIZ / "docs" / "grafo.mmd"
destino.parent.mkdir(exist_ok=True)
mermaid = construir_grafo(checkpointer=None, recuperador=None).get_graph().draw_mermaid()
destino.write_text(mermaid, encoding="utf-8")
print(mermaid)
print(f"Guardado en {destino.relative_to(RAIZ)}")
