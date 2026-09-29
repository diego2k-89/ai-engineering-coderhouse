"""
Script de prueba del RAG.

Corre tres pruebas:
  1. Pregunta con respuesta en los documentos.
  2. Pregunta trampa: algo que las políticas no cubren, para verificar que el
     sistema lo admite en vez de inventar.
  3. La misma pregunta con los dos proveedores, en paralelo, para comparar cómo
     redacta cada uno sobre el mismo contexto.

Uso:
    python main.py                # las tres pruebas
    python main.py --interactivo  # las tres pruebas y después consola de preguntas

Requiere el índice ya construido. Si no existe, se construye en la primera
consulta (ver ingest.py).
"""

import argparse
import asyncio
import logging

from rag import get_rag_response, get_retriever
from schemas import RAGResponse

PREGUNTA_CON_RESPUESTA = (
    "¿Cuántos días de vacaciones le corresponden a un empleado con 7 años de "
    "antigüedad?"
)

# Ninguno de los cuatro documentos habla de bonos ni de remuneración variable.
# La pregunta está elegida para que el retriever igual traiga algo: la política
# de teletrabajo menciona un subsidio que se liquida con el sueldo. O sea que el
# modelo recibe contexto plausible y adyacente, y de todas formas tiene que
# reconocer que la respuesta no está ahí. Es una prueba más dura que preguntar
# algo sin ninguna relación con los documentos.
PREGUNTA_TRAMPA = (
    "¿Cuál es la política de bonos por rendimiento anual y cómo se calcula el "
    "porcentaje sobre el salario?"
)

PREGUNTA_COMPARACION = (
    "¿Cuántos días por semana se puede trabajar de forma remota y qué "
    "requisitos de conectividad hay?"
)

PROVEEDORES = ("anthropic", "gemini")


def configurar_logs() -> None:
    """Logs del RAG en INFO; el ruido de las librerías, más arriba.

    `httpx2` va en la lista además de `httpx`: el SDK de Anthropic loguea por
    ahí, y sin silenciarlo cada consulta imprime su propia línea de HTTP 200.

    `google_genai.models` va a ERROR y no a WARNING porque emite una
    recomendación sobre automatic function calling en cada llamada. Es una
    sugerencia de estilo de la librería, no un problema de este código.

    `google_genai._api_client` queda a propósito en INFO. Ahí aparecen los
    reintentos que el SDK de Google hace por su cuenta ante un 503, y esa
    información importa: explica de dónde sale un elapsed alto sin que haya
    fallado nada.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    for ruidoso in ("httpx", "httpx2", "httpcore", "chromadb"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n  {texto}\n{'=' * 70}")


def mostrar(r: RAGResponse, con_fragmentos: bool = False) -> None:
    """Imprime una respuesta con sus fuentes reales."""
    marca = "SÍ" if r.encontro_respuesta else "NO"
    print(f"\nRespuesta ({r.proveedor}):\n{r.respuesta}\n")
    print(f"¿Encontró la información en el contexto? {marca}")
    print(f"Fuentes: {', '.join(r.fuentes) if r.fuentes else '(ninguna)'}")
    print(f"Fragmentos usados: {r.cantidad_fragmentos}")

    if con_fragmentos:
        print("\nFragmentos que vio el modelo:")
        for i, f in enumerate(r.fragmentos, 1):
            print(f"  [{i}] {f.fuente}")
            print(f"      {f.extracto[:120]}...")


async def demo_con_respuesta(retriever) -> None:
    titulo("1. Pregunta con respuesta en los documentos")
    print(f"Pregunta: {PREGUNTA_CON_RESPUESTA}")

    # Se muestran los fragmentos recuperados para poder verificar a mano que la
    # respuesta sale del documento y no del modelo.
    mostrar(
        await get_rag_response(PREGUNTA_CON_RESPUESTA, retriever=retriever),
        con_fragmentos=True,
    )


async def demo_trampa(retriever) -> None:
    titulo("2. Pregunta trampa: información que no está en los documentos")
    print(f"Pregunta: {PREGUNTA_TRAMPA}")
    print(
        "\nNinguna de las cuatro políticas habla de bonos. El retriever va a\n"
        "traer fragmentos igual, porque siempre devuelve los k más parecidos:\n"
        "similitud no es lo mismo que pertinencia. Lo que se prueba acá es si\n"
        "el modelo admite que la respuesta no está."
    )

    resultado = await get_rag_response(PREGUNTA_TRAMPA, retriever=retriever)
    mostrar(resultado, con_fragmentos=True)

    if resultado.encontro_respuesta:
        print(
            "\nATENCIÓN: el modelo marcó que encontró la respuesta. Hay que\n"
            "revisar el texto: o los documentos cubren el tema más de lo\n"
            "esperado, o el anclaje al contexto falló."
        )
    else:
        print(
            "\nCorrecto: recuperó fragmentos pero reconoció que no alcanzan\n"
            "para responder. Eso es el comportamiento buscado."
        )


async def demo_comparacion(retriever) -> None:
    titulo("3. Los dos proveedores sobre el mismo contexto")
    print(f"Pregunta: {PREGUNTA_COMPARACION}")
    print(
        "\nMismo índice, mismo retriever y misma pregunta: los fragmentos que\n"
        "recibe cada modelo son idénticos. Lo único que cambia es quién redacta.\n"
        "Las dos consultas salen en paralelo con asyncio.gather, porque son\n"
        "espera de red y no hay razón para hacerlas una después de la otra.\n"
    )

    resultados = await asyncio.gather(
        *(
            get_rag_response(PREGUNTA_COMPARACION, provider=p, retriever=retriever)
            for p in PROVEEDORES
        ),
        # Sin esto, un proveedor caído cancelaría la otra consulta y la demo se
        # cortaría entera.
        return_exceptions=True,
    )

    for proveedor, resultado in zip(PROVEEDORES, resultados):
        print(f"\n--- {proveedor} ---")
        if isinstance(resultado, Exception):
            print(f"Falló: {type(resultado).__name__}: {str(resultado)[:200]}")
        else:
            mostrar(resultado)


async def modo_interactivo(retriever) -> None:
    titulo("Modo interactivo")
    print("Preguntá sobre las políticas de Vantia. 'salir' para terminar.\n")

    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break

        try:
            mostrar(await get_rag_response(pregunta, retriever=retriever))
        except Exception as e:
            print(f"Falló la consulta: {type(e).__name__}: {str(e)[:200]}")

        print("-" * 70)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Pruebas del RAG local")
    parser.add_argument(
        "--interactivo",
        action="store_true",
        help="Abre una consola de preguntas después de las tres pruebas",
    )
    args = parser.parse_args()

    configurar_logs()

    # Un solo retriever para todas las consultas: construirlo abre el índice de
    # Chroma, y no hace falta repetir eso en cada pregunta.
    retriever = get_retriever()

    await demo_con_respuesta(retriever)
    await demo_trampa(retriever)
    await demo_comparacion(retriever)

    if args.interactivo:
        await modo_interactivo(retriever)

    print()


if __name__ == "__main__":
    asyncio.run(main())
