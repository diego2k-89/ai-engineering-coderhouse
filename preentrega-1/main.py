"""
Script de prueba del cliente LLM.

Corre tres demostraciones:
  1. Modo normal, mostrando el fallback entre proveedores.
  2. Modo streaming, midiendo el tiempo hasta el primer token.
  3. Validación con Pydantic rechazando una configuración inválida.

Uso:
    python main.py
"""

import asyncio
import logging
import time

from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client import AsyncLLMManager, ChatMessage, LLMConfig, Provider, Role
from llm_client.manager import cadena_desde_entorno

PREGUNTA = [
    ChatMessage(
        role=Role.SYSTEM,
        content="Respondé en español, claro y breve: dos líneas como máximo.",
    ),
    ChatMessage(role=Role.USER, content="¿Qué es la entropía?"),
]


def titulo(texto: str) -> None:
    print(f"\n{'=' * 62}\n  {texto}\n{'=' * 62}")


async def demo_normal(manager: AsyncLLMManager) -> None:
    titulo("1. Modo normal")

    inicio = time.perf_counter()
    respuesta = await manager.generate(PREGUNTA)
    tardanza = time.perf_counter() - inicio

    if respuesta.ok:
        print(f"Respondió: {respuesta.provider.value} ({respuesta.model})")
        print(f"Latencia total: {tardanza:.2f}s\n")
        print(respuesta.content)
    else:
        print("Ningún proveedor pudo responder.")
        print(respuesta.error)


async def demo_streaming(manager: AsyncLLMManager) -> None:
    titulo("2. Modo streaming")

    inicio = time.perf_counter()
    ttft = None

    async for fragmento in manager.generate_stream(PREGUNTA):
        if ttft is None:
            ttft = time.perf_counter() - inicio
            print(f"(primer token a los {ttft:.2f}s)\n")
        print(fragmento, end="", flush=True)

    total = time.perf_counter() - inicio
    proveedor = manager.ultimo_proveedor
    print(f"\n\nRespondió: {proveedor.value if proveedor else '-'}")
    print(f"Latencia total: {total:.2f}s")


def demo_validacion() -> None:
    titulo("3. Validación con Pydantic")

    print("Intento crear una configuración con temperature=5 (el máximo es 2):\n")
    try:
        LLMConfig(
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-5",
            api_key="no-importa",
            temperature=5,
        )
        print("No debería llegar acá.")
    except ValidationError as e:
        print(e)
        print("\nRechazado antes de llamar a la API. Cero tokens gastados.")


async def main() -> None:
    load_dotenv()

    # El SDK de Gemini avisa por consola sobre "automatic function calling"
    # aunque no usemos tools. Es ruido: lo bajamos a nivel error.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)

    cadena = cadena_desde_entorno()
    print("Cadena de proveedores:", " -> ".join(p.value for p in cadena))
    print("El primero está sin API key a propósito, para ver el fallback.")

    manager = AsyncLLMManager()

    await demo_normal(manager)
    await demo_streaming(manager)
    demo_validacion()

    print()


if __name__ == "__main__":
    asyncio.run(main())
