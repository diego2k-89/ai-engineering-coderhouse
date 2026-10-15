"""
Herramientas del agente: consultas a una base de clientes y pedidos simulada.

El LLM decide qué herramienta usar leyendo solo el nombre y el docstring de cada
una. Por eso los docstrings son largos y concretos: dicen qué recibe, qué
devuelve, y qué hacer cuando la respuesta es un error.

Las herramientas no lanzan excepciones ante un dato inválido: devuelven el error
como resultado. ToolNode, con su configuración por defecto, vuelve a lanzar las
excepciones en vez de mostrárselas al modelo. Devolviendo el error como dato, el
modelo lo lee y puede corregir la llamada o pedir una aclaración, que es el
"ciclo de retorno" que pide la consigna.
"""

import asyncio
import difflib
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Base de datos simulada
# ---------------------------------------------------------------------------

CLIENTES: dict[int, dict[str, Any]] = {
    101: {"nombre": "Acme SRL", "ciudad": "Córdoba"},
    102: {"nombre": "Distribuidora Norte", "ciudad": "Salta"},
    103: {"nombre": "TecnoSur SA", "ciudad": "Bahía Blanca"},
    104: {"nombre": "Acme Logística", "ciudad": "Rosario"},
}

PEDIDOS: list[dict[str, Any]] = [
    {"pedido_id": 5001, "cliente_id": 102, "fecha": "2026-07-03", "total": 4200, "estado": "entregado"},
    {"pedido_id": 5007, "cliente_id": 102, "fecha": "2026-08-19", "total": 6300, "estado": "entregado"},
    {"pedido_id": 5012, "cliente_id": 102, "fecha": "2026-09-30", "total": 4000, "estado": "en camino"},
    {"pedido_id": 5003, "cliente_id": 103, "fecha": "2026-07-21", "total": 18900, "estado": "entregado"},
    {"pedido_id": 5010, "cliente_id": 103, "fecha": "2026-09-02", "total": 7350, "estado": "cancelado"},
    {"pedido_id": 5002, "cliente_id": 101, "fecha": "2026-07-10", "total": 2100, "estado": "entregado"},
    {"pedido_id": 5011, "cliente_id": 104, "fecha": "2026-09-15", "total": 9800, "estado": "entregado"},
]

# Latencia simulada de una consulta a la base, para que la ejecución sea
# realmente asíncrona y no un cálculo instantáneo.
LATENCIA_SEGUNDOS = 0.2


# ---------------------------------------------------------------------------
# Herramientas
# ---------------------------------------------------------------------------


@tool
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """Busca un cliente por su nombre y devuelve su cliente_id.

    Usar esta herramienta SIEMPRE que la pregunta nombre a un cliente por su
    nombre, antes de consultar sus pedidos: buscar_pedidos necesita el
    cliente_id numérico, no el nombre.

    Acepta el nombre completo o una parte ("Norte" encuentra a
    "Distribuidora Norte"). No distingue mayúsculas.

    Devuelve uno de estos tres resultados:
    - Un cliente: {"cliente_id": 102, "nombre": "...", "ciudad": "..."}.
    - Varios clientes que coinciden: {"error": "ambiguo", "coincidencias": [...]}.
      En ese caso NO elegir uno al azar: preguntarle al usuario a cuál se refiere.
    - Ninguno: {"error": "no_encontrado", "sugerencias": [...]}. Si hay una
      sugerencia que se parece claramente al nombre pedido (un error de tipeo),
      volver a llamar a esta herramienta con ese nombre. Si no hay sugerencias,
      decirle al usuario que el cliente no existe.
    """
    await asyncio.sleep(LATENCIA_SEGUNDOS)

    buscado = nombre.strip().lower()
    coincidencias = [
        {"cliente_id": cid, **datos}
        for cid, datos in CLIENTES.items()
        if buscado in datos["nombre"].lower()
    ]

    if len(coincidencias) == 1:
        return coincidencias[0]

    if len(coincidencias) > 1:
        return {"error": "ambiguo", "coincidencias": coincidencias}

    # Comparación en minúsculas: difflib distingue mayúsculas, y "tecnosurr" no
    # se parecería lo suficiente a "TecnoSur SA".
    por_minuscula = {datos["nombre"].lower(): datos["nombre"] for datos in CLIENTES.values()}
    parecidos = difflib.get_close_matches(buscado, list(por_minuscula), n=3, cutoff=0.6)
    sugerencias = [por_minuscula[p] for p in parecidos]
    return {"error": "no_encontrado", "buscado": nombre, "sugerencias": sugerencias}


@tool
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Devuelve todos los pedidos de un cliente, a partir de su cliente_id.

    Requiere el cliente_id numérico (por ejemplo 102). Si solo se conoce el
    nombre del cliente, llamar antes a buscar_cliente para obtenerlo.

    Devuelve {"cliente_id", "cantidad", "total", "pedidos"}, donde "total" es la
    suma en pesos de todos los pedidos, incluidos los cancelados, y "pedidos" es
    la lista ordenada del más viejo al más nuevo, con pedido_id, fecha, total y
    estado de cada uno. El último pedido es el último de la lista.

    Si el cliente_id no existe, devuelve {"error": "cliente_inexistente"}.
    """
    await asyncio.sleep(LATENCIA_SEGUNDOS)

    if cliente_id not in CLIENTES:
        return {"error": "cliente_inexistente", "cliente_id": cliente_id}

    pedidos = sorted(
        (p for p in PEDIDOS if p["cliente_id"] == cliente_id), key=lambda p: p["fecha"]
    )
    return {
        "cliente_id": cliente_id,
        "cantidad": len(pedidos),
        "total": sum(p["total"] for p in pedidos),
        "pedidos": pedidos,
    }


HERRAMIENTAS = [buscar_cliente, buscar_pedidos]
