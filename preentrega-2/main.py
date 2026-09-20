"""
Script de prueba del pipeline.

Corre cuatro pruebas:
  1. Extracción sobre un texto técnico, con los tres proveedores.
  2. Prueba de estrés: un texto ambiguo, sin contenido técnico.
  3. Truncamiento forzado: max_tokens ridículamente bajo, para ver la
     detección de respuesta incompleta y los reintentos.

Uso:
    python main.py
"""

import asyncio
import logging

from chain import build_chain, process_text
from schemas import EntidadesTecnicas

TEXTO_TECNICO = (
    "Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché "
    "en Redis parece saturarse en picos de tráfico y las conexiones a "
    "PostgreSQL se agotan porque el pool está mal dimensionado. Esto está "
    "afectando a usuarios en producción."
)

TEXTO_AMBIGUO = "El sistema anda medio raro últimamente, no sé bien qué está pasando."

PROVEEDORES = ("anthropic", "gemini", "openai")


def configurar_logs() -> None:
    """Logs del pipeline en INFO; el ruido de las librerías, en WARNING."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "google_genai.models"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    # LangChain también detecta el corte por max_tokens, pero lo reporta
    # imprimiendo un traceback completo en vez de lanzar. En la demo 3 eso
    # aparece tres veces y tapa lo que queremos ver, que es nuestra detección.
    logging.getLogger("langchain_core.output_parsers").setLevel(logging.CRITICAL)


def titulo(texto: str) -> None:
    print(f"\n{'=' * 66}\n  {texto}\n{'=' * 66}")


def mostrar(resultado: EntidadesTecnicas) -> None:
    print(resultado.model_dump_json(indent=2))


async def demo_proveedores() -> None:
    titulo("1. El mismo texto, los tres proveedores")
    print("La cadena es idéntica; solo cambia el modelo que hay detrás.\n")

    for proveedor in PROVEEDORES:
        print(f"--- {proveedor} ---")
        try:
            mostrar(await process_text(TEXTO_TECNICO, provider=proveedor))
        except Exception as e:
            print(f"Falló: {type(e).__name__}: {str(e)[:160]}")
        print()


async def demo_texto_ambiguo() -> None:
    titulo("2. Prueba de estrés: texto sin contenido técnico")
    print(f"Entrada: {TEXTO_AMBIGUO!r}\n")
    print(
        "El esquema exige al menos una tecnología, pero el texto no nombra\n"
        "ninguna. O el modelo inventa algo para cumplir, o el validador lo\n"
        "rechaza. Las dos salidas son informativas.\n"
    )

    try:
        resultado = await process_text(TEXTO_AMBIGUO)
        print("El modelo produjo una salida válida:")
        mostrar(resultado)
        print(
            "\nOjo: revisar si las tecnologías salen del texto o si el modelo\n"
            "las completó para cumplir con el esquema."
        )
    except Exception as e:
        print(f"Rechazado tras los reintentos: {type(e).__name__}")
        print(f"{str(e)[:300]}")
        print("\nEl pipeline prefirió no devolver nada antes que devolver algo inventado.")


async def demo_truncamiento() -> None:
    titulo("3. Truncamiento forzado (finish_reason)")
    print(
        "Armamos la cadena con max_tokens=25: el modelo no llega a completar\n"
        "la respuesta. Sin detección, un JSON cortado puede seguir siendo\n"
        "válido y pasar como bueno.\n"
    )

    cadena = build_chain(max_tokens=25)

    try:
        resultado = await cadena.ainvoke({"texto": TEXTO_TECNICO})
        print("Inesperado: el modelo entró en 25 tokens.")
        mostrar(resultado)
    except Exception as e:
        print(f"\nDetectado y rechazado: {type(e).__name__}")
        print(f"{str(e)[:300]}")
        print(
            "\nLos reintentos de arriba son .with_retry() haciendo su trabajo.\n"
            "Con el presupuesto de tokens fijo, los tres fallan igual: el\n"
            "problema no es pasajero."
        )


async def main() -> None:
    configurar_logs()

    await demo_proveedores()
    await demo_texto_ambiguo()
    await demo_truncamiento()

    print()


if __name__ == "__main__":
    asyncio.run(main())
