"""
Ingesta de la documentación a Pinecone (de la pre-entrega 4, ahora async).

    .md -> limpiar -> fragmentar -> aembed_documents (lotes de 20) -> upsert async (lotes de 100)

Es lo primero que corre `docker compose up`: crea el índice si no existe y sube
los fragmentos si el namespace no los tiene todos. Si ya están, termina en un
segundo. Así el sistema arranca igual en una máquina limpia (índice vacío) que
en una donde ya se cargó.

Cada fragmento tiene un ID fijo (`<archivo>#<n>`): volver a subir pisa los
vectores en vez de duplicarlos.

Uso:
    python -m app.rag.ingesta            # crea el índice y sube lo que falte
    python -m app.rag.ingesta --reset    # vacía el namespace y sube todo de cero
"""

import argparse
import asyncio
import logging
import os
import time

from langchain_core.documents import Document

from app.config import (
    EMBEDDING_DIM,
    INDEX_NAME,
    NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    configurar_logs,
    crear_embeddings,
    requerir,
)
from app.rag.documentos import cargar_fragmentos

logger = logging.getLogger("rag.ingesta")

TAMANIO_LOTE = int(os.getenv("BATCH_SIZE", "100"))
# El plan gratuito de Google limita cuánto se vectoriza por minuto: mandar los
# 83 fragmentos juntos devolvió 429 en la pre-entrega 4.
EMBED_TAMANIO_LOTE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_PAUSA = float(os.getenv("EMBED_PAUSE_SECONDS", "15"))
EMBED_REINTENTOS = 4


async def asegurar_indice(pc, embeddings) -> None:
    """Crea el índice si no existe; si existe, valida su dimensión.

    La dimensión se mide con un embedding de prueba en vez de confiar en el
    número del .env: si no coincide con la del índice, el upsert fallaría.
    """
    from pinecone import ServerlessSpec

    dimension = len(await embeddings.aembed_query("dimension"))

    if await pc.has_index(INDEX_NAME):
        descripcion = await pc.describe_index(INDEX_NAME)
        if descripcion.dimension != dimension:
            raise SystemExit(
                f"El índice '{INDEX_NAME}' tiene dimensión {descripcion.dimension} y el "
                f"modelo produce {dimension}. Usar otro INDEX_NAME."
            )
        logger.info("El índice '%s' ya existe (dimensión %d)", INDEX_NAME, dimension)
        return

    if dimension != EMBEDDING_DIM:
        logger.warning("EMBEDDING_DIM=%d pero el modelo produce %d", EMBEDDING_DIM, dimension)
    logger.info("Creando el índice '%s' (dimensión %d, coseno)", INDEX_NAME, dimension)
    await pc.create_index(
        name=INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


def _es_limite_de_uso(error: Exception) -> bool:
    texto = str(error)
    return "429" in texto or "RESOURCE_EXHAUSTED" in texto


async def vectorizar_lote(embeddings, textos: list[str]) -> list[list[float]]:
    """Vectoriza un lote; ante un 429 espera 30, 60, 120 s y reintenta.

    Cualquier otro error se propaga: una key inválida no se arregla esperando.
    """
    espera = 30
    for intento in range(1, EMBED_REINTENTOS + 1):
        try:
            return await embeddings.aembed_documents(textos)
        except Exception as error:
            if not _es_limite_de_uso(error) or intento == EMBED_REINTENTOS:
                raise
            logger.warning("429 de la API de embeddings (intento %d); reintento en %ds", intento, espera)
            await asyncio.sleep(espera)
            espera *= 2
    raise AssertionError("inalcanzable")


async def subir(indice, embeddings, fragmentos: list[Document]) -> None:
    valores: list[list[float]] = []
    for desde in range(0, len(fragmentos), EMBED_TAMANIO_LOTE):
        if desde:
            await asyncio.sleep(EMBED_PAUSA)
        lote = fragmentos[desde : desde + EMBED_TAMANIO_LOTE]
        valores.extend(await vectorizar_lote(embeddings, [f.page_content for f in lote]))
        logger.info("Embeddings %d-%d de %d", desde + 1, desde + len(lote), len(fragmentos))

    vectores = [
        {"id": f.id, "values": v, "metadata": {**f.metadata, "texto": f.page_content}}
        for f, v in zip(fragmentos, valores)
    ]
    for desde in range(0, len(vectores), TAMANIO_LOTE):
        await indice.upsert(vectors=vectores[desde : desde + TAMANIO_LOTE], namespace=NAMESPACE)
        logger.info("Upsert %d-%d de %d", desde + 1, min(desde + TAMANIO_LOTE, len(vectores)), len(vectores))


async def vectores_en_namespace(indice) -> int:
    estadisticas = await indice.describe_index_stats()
    namespace = (estadisticas.namespaces or {}).get(NAMESPACE)
    return namespace.vector_count if namespace else 0


async def main(reset: bool) -> None:
    from pinecone import PineconeAsyncio

    inicio = time.perf_counter()
    fragmentos = cargar_fragmentos()
    embeddings = crear_embeddings()

    async with PineconeAsyncio(api_key=requerir("PINECONE_API_KEY")) as pc:
        await asegurar_indice(pc, embeddings)
        descripcion = await pc.describe_index(INDEX_NAME)

        async with pc.IndexAsyncio(host=descripcion.host) as indice:
            if reset:
                logger.warning("Vaciando el namespace '%s'", NAMESPACE)
                await indice.delete(delete_all=True, namespace=NAMESPACE)
            elif (cargados := await vectores_en_namespace(indice)) >= len(fragmentos):
                logger.info(
                    "El namespace '%s' ya tiene %d vectores (%d fragmentos): no hay nada que subir",
                    NAMESPACE, cargados, len(fragmentos),
                )
                return

            logger.info("Subiendo %d fragmentos a '%s/%s'", len(fragmentos), INDEX_NAME, NAMESPACE)
            await subir(indice, embeddings, fragmentos)

    logger.info("Ingesta completa en %.2fs", time.perf_counter() - inicio)


if __name__ == "__main__":
    configurar_logs()
    parser = argparse.ArgumentParser(description="Ingesta de la documentación a Pinecone")
    parser.add_argument("--reset", action="store_true", help="vacía el namespace antes de subir")
    asyncio.run(main(parser.parse_args().reset))
