"""
Recuperación híbrida de la pre-entrega 4, ahora asíncrona de punta a punta.

    consulta -> BM25 (en memoria)                       --\\
                                                          > RRF (0,3 / 0,7) -> top-5
    consulta -> aembed_query -> Pinecone (IndexAsyncio) --/

- **BM25** busca coincidencias exactas de palabras: nombres de clases,
  parámetros, variables de entorno. Vive en memoria y se arma al arrancar con
  los mismos fragmentos que están en Pinecone.
- **Vectorial**: el embedding de la consulta sale de `aembed_query` (el cliente
  async de Google) y la búsqueda de `IndexAsyncio` (el cliente async de
  Pinecone). En las pre-entregas 6 y 7 estas dos llamadas eran sincrónicas y
  corrían en un hilo aparte con `asyncio.to_thread`.
- **RRF**: el `EnsembleRetriever` combina los dos rankings por posición. Los
  pesos 0,3 / 0,7 salen de la evaluación de la pre-entrega 4: con 0,5 / 0,5 los
  empates exactos se resolvían a favor de BM25 y el fragmento correcto bajaba
  al segundo lugar.

Las dos búsquedas corren a la vez: `EnsembleRetriever.ainvoke` las lanza con
`asyncio.gather`.
"""

import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import (
    INDEX_NAME,
    NAMESPACE,
    PESO_BM25,
    PESO_VECTORIAL,
    TOP_K,
    crear_embeddings,
    requerir,
)
from app.rag.documentos import cargar_fragmentos

logger = logging.getLogger("rag.recuperador")


def tokenizar(texto: str) -> list[str]:
    """Minúsculas y palabras, para BM25.

    El tokenizador por defecto de BM25Retriever es `texto.split()`: separa solo
    por espacios. Una pregunta que dice "`trust_env`?" no coincidiría con
    "trust_env". `\\w+` conserva el guion bajo, así que `trust_env` queda entero.
    """
    return re.findall(r"\w+", texto.lower())


class PineconeRetriever(BaseRetriever):
    """Retriever vectorial sobre el cliente async de Pinecone.

    Solo implementa la versión async. La sincrónica lanza un error a propósito:
    si alguna parte del sistema la llamara, bloquearía el event loop.
    """

    indice: Any
    embeddings: Any
    namespace: str
    k: int = TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise NotImplementedError("PineconeRetriever es solo async: usar ainvoke()")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = await self.embeddings.aembed_query(query)
        respuesta = await self.indice.query(
            vector=vector, top_k=self.k, namespace=self.namespace, include_metadata=True
        )

        documentos = []
        for resultado in respuesta.matches:
            metadata = dict(resultado.metadata)
            # El texto viaja en la metadata (así lo sube la ingesta). Tiene que
            # quedar idéntico al de BM25 para que el ensemble los reconozca
            # como el mismo fragmento.
            texto = metadata.pop("texto")
            metadata["fragmento"] = int(metadata.get("fragmento", 0))
            metadata["score"] = round(resultado.score, 4)
            documentos.append(Document(page_content=texto, metadata=metadata, id=resultado.id))
        return documentos


class RecuperadorHibrido:
    """El EnsembleRetriever de BM25 + Pinecone, cortado en top-k."""

    def __init__(self, bm25: BaseRetriever, vectorial: BaseRetriever, k: int = TOP_K):
        self.k = k
        # El orden importa: ante un empate de puntaje, gana el primero de la
        # lista. Con pesos distintos no hay empates.
        self.ensemble = EnsembleRetriever(
            retrievers=[bm25, vectorial], weights=[PESO_BM25, PESO_VECTORIAL]
        )

    async def buscar(self, consulta: str) -> list[Document]:
        """Top-k fragmentos. El ensemble no corta: devuelve todo lo fusionado."""
        inicio = time.perf_counter()
        documentos = (await self.ensemble.ainvoke(consulta))[: self.k]
        logger.info("Búsqueda híbrida: %d fragmentos en %.2fs", len(documentos), time.perf_counter() - inicio)
        return documentos


@asynccontextmanager
async def abrir_recuperador() -> AsyncIterator[RecuperadorHibrido]:
    """Abre las conexiones a Pinecone y arma el recuperador.

    Los clientes async de Pinecone abren una sesión HTTP (aiohttp) que hay que
    cerrar: por eso es un context manager que vive lo mismo que la API.
    """
    from pinecone import PineconeAsyncio

    fragmentos = cargar_fragmentos()
    bm25 = BM25Retriever.from_documents(fragmentos, preprocess_func=tokenizar)
    bm25.k = TOP_K

    async with PineconeAsyncio(api_key=requerir("PINECONE_API_KEY")) as pc:
        descripcion = await pc.describe_index(INDEX_NAME)
        async with pc.IndexAsyncio(host=descripcion.host) as indice:
            vectorial = PineconeRetriever(
                indice=indice, embeddings=crear_embeddings(), namespace=NAMESPACE, k=TOP_K
            )
            logger.info(
                "Recuperador listo: BM25 con %d fragmentos + Pinecone '%s/%s'",
                len(fragmentos), INDEX_NAME, NAMESPACE,
            )
            yield RecuperadorHibrido(bm25, vectorial)
