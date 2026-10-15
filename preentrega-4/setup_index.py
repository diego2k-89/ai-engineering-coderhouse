"""
Inicialización del índice de Pinecone.

Verifica si el índice existe y, si no, lo crea en modo Serverless. Si ya existe,
controla que su dimensión coincida con la del modelo de embeddings.

Ese control es el "mismatch de dimensiones" que marca la consigna: la dimensión
de un índice se fija al crearlo y no se puede cambiar después. Subir vectores de
otro tamaño falla, así que conviene detectarlo acá y no en medio de la ingesta.

Uso:
    python setup_index.py
"""

import logging
import time

from pinecone import ServerlessSpec

from config import (
    CLOUD,
    EMBEDDING_MODEL,
    INDEX_NAME,
    REGION,
    configurar_logs,
    get_embeddings,
    get_pinecone,
)

logger = logging.getLogger("rag.setup")


def medir_dimension(embeddings) -> int:
    """Dimensión real de los vectores que produce el modelo.

    Se mide generando un embedding de prueba en vez de confiar en un número
    escrito a mano. Si el modelo ignorara la dimensión pedida, esto lo detecta
    antes de crear un índice que después no serviría.
    """
    return len(embeddings.embed_query("dimension"))


def asegurar_indice(pc, nombre: str, dimension: int) -> None:
    """Crea el índice si no existe; si existe, valida su dimensión."""
    if pc.has_index(nombre):
        descripcion = pc.describe_index(nombre)

        if descripcion.dimension != dimension:
            raise ValueError(
                f"El índice '{nombre}' existe con dimensión {descripcion.dimension}, "
                f"pero el modelo produce vectores de {dimension}. Usar otro "
                "INDEX_NAME o borrar el índice desde la consola de Pinecone."
            )

        logger.info(
            "El índice '%s' ya existe (dimensión %d, métrica %s): no se crea",
            nombre,
            descripcion.dimension,
            descripcion.metric,
        )
        return

    logger.info(
        "Creando el índice '%s' (dimensión %d, coseno, %s %s)...",
        nombre,
        dimension,
        CLOUD,
        REGION,
    )
    inicio = time.perf_counter()

    # Coseno por dos motivos. Es la métrica que recomienda Google para estos
    # embeddings. Y con gemini-embedding-001 solo los vectores de 3072 vienen
    # normalizados: los de 1536 no. El coseno divide por la longitud de cada
    # vector, así que compara dirección y no hace falta normalizar a mano. Con
    # producto punto, en cambio, la longitud distorsionaría el resultado.
    pc.create_index(
        name=nombre,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )

    logger.info("Índice creado en %.2fs", time.perf_counter() - inicio)


def main() -> None:
    configurar_logs()

    dimension = medir_dimension(get_embeddings())
    logger.info(
        "El modelo %s produce vectores de %d dimensiones", EMBEDDING_MODEL, dimension
    )

    asegurar_indice(get_pinecone(), INDEX_NAME, dimension)


if __name__ == "__main__":
    main()
