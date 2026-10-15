"""
Recuperador híbrido: combina búsqueda léxica (BM25) con búsqueda vectorial
(Pinecone) en un EnsembleRetriever, y devuelve el top-5.

    pregunta -> BM25 (en memoria)   --\\
                                        > RRF -> top-5
    pregunta -> Pinecone (vectorial) --/

BM25 busca coincidencias exactas de palabras: es bueno con nombres de clases,
parámetros y variables de entorno. La búsqueda vectorial busca por significado:
es buena con preguntas formuladas con otras palabras. El EnsembleRetriever fusiona
los dos rankings con Reciprocal Rank Fusion, que combina por posición y no por
puntaje: los puntajes de BM25 y los de similitud coseno están en escalas
distintas y sumarlos directamente no tiene sentido.

Uso:
    python rag.py "¿Para qué sirve trust_env?"
"""

import logging
import re
import sys
import time
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import INDEX_NAME, NAMESPACE, configurar_logs, get_embeddings, get_pinecone
from ingest import cargar_fragmentos

logger = logging.getLogger("rag.consulta")

TOP_K = 5

# Pesos del EnsembleRetriever (BM25, vectorial). No son 50/50 a propósito.
#
# Con pesos iguales, el puesto n de BM25 y el puesto n del vectorial suman
# exactamente lo mismo en RRF, 0.5/(n+60), y el EnsembleRetriever desempata por
# orden de aparición: gana el retriever que está primero en la lista. El
# resultado queda decidido por un desempate, no por la calidad de cada uno.
# En la evaluación, eso mandaba el documento correcto del primer al segundo
# puesto en las tres preguntas semánticas.
#
# Con 0.3/0.7 no hay empates y el vectorial, que en la evaluación fue el más
# preciso, define el orden. BM25 sigue sumando cuando coincide con el vectorial.
PESOS = (0.3, 0.7)


# ---------------------------------------------------------------------------
# Retriever léxico
# ---------------------------------------------------------------------------


def tokenizar(texto: str) -> list[str]:
    """Pasa a minúsculas y separa en palabras, para BM25.

    El tokenizador por defecto de BM25Retriever es `texto.split()`: separa solo
    por espacios y distingue mayúsculas. Con eso, una pregunta que escribe
    "`trust_env`?" no coincide con "trust_env" en el documento, por los
    backticks y el signo de pregunta pegados. Los nombres exactos son justo el
    caso donde BM25 tiene que aportar, así que el tokenizador importa.

    `\\w+` conserva el guion bajo: `trust_env` y `SSL_CERT_FILE` quedan como un
    solo término. En cambio `httpx.Client` se separa en "httpx" y "client".
    """
    return re.findall(r"\w+", texto.lower())


def crear_bm25(fragmentos: list[Document], k: int = TOP_K) -> BM25Retriever:
    """Arma el índice BM25 en memoria a partir de los fragmentos locales.

    BM25 no vive en Pinecone: se reconstruye en cada corrida con los mismos
    fragmentos que se subieron. Para 83 fragmentos tarda milisegundos.
    """
    retriever = BM25Retriever.from_documents(fragmentos, preprocess_func=tokenizar)
    retriever.k = k
    return retriever


# ---------------------------------------------------------------------------
# Retriever vectorial
# ---------------------------------------------------------------------------


class PineconeRetriever(BaseRetriever):
    """Retriever de LangChain sobre el SDK de Pinecone.

    Hace lo mismo que `PineconeVectorStore.as_retriever()` de la librería
    langchain-pinecone, que no se puede usar porque su última versión no
    soporta Python 3.14. Son tres pasos: vectorizar la pregunta, consultar a
    Pinecone y convertir cada resultado en un Document.

    Al heredar de BaseRetriever, el EnsembleRetriever lo puede combinar con BM25
    igual que a cualquier retriever de LangChain.
    """

    indice: Any
    embeddings: Any
    namespace: str
    k: int = TOP_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = self.embeddings.embed_query(query)

        respuesta = self.indice.query(
            vector=vector,
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
        )

        documentos = []
        for resultado in respuesta.matches:
            metadata = dict(resultado.metadata)

            # El texto viaja dentro de la metadata (así se subió en ingest.py).
            # Tiene que quedar idéntico al page_content de BM25: el ensemble
            # reconoce que dos resultados son el mismo fragmento comparando ese
            # texto, y recién ahí suma sus puntajes.
            texto = metadata.pop("texto")

            # Pinecone guarda los números como float: 3 vuelve como 3.0.
            metadata["fragmento"] = int(metadata.get("fragmento", 0))
            metadata["score"] = round(resultado.score, 4)

            documentos.append(
                Document(page_content=texto, metadata=metadata, id=resultado.id)
            )

        return documentos


# ---------------------------------------------------------------------------
# El sistema
# ---------------------------------------------------------------------------


class RAGSystem:
    """Encapsula el EnsembleRetriever y devuelve los top-5 fragmentos.

    Expone también los dos retrievers por separado, para que evaluate.py pueda
    comparar el híbrido contra cada uno solo.
    """

    def __init__(self, k: int = TOP_K, pesos: tuple[float, float] = PESOS):
        inicio = time.perf_counter()
        self.k = k

        self.bm25 = crear_bm25(cargar_fragmentos(), k=k)
        self.vectorial = PineconeRetriever(
            indice=get_pinecone().Index(name=INDEX_NAME),
            embeddings=get_embeddings(),
            namespace=NAMESPACE,
            k=k,
        )

        # El orden de la lista importa: ante un empate de puntaje, gana el
        # resultado del primer retriever. Ver el comentario de PESOS.
        self.hibrido = EnsembleRetriever(
            retrievers=[self.bm25, self.vectorial],
            weights=list(pesos),
        )

        self.retrievers = {
            "bm25": self.bm25,
            "vectorial": self.vectorial,
            "hibrido": self.hibrido,
        }

        logger.info(
            "RAGSystem listo en %.2fs (top-%d, pesos BM25/vectorial %s)",
            time.perf_counter() - inicio,
            k,
            pesos,
        )

    def buscar(self, consulta: str, modo: str = "hibrido") -> list[Document]:
        """Devuelve los top-k fragmentos para una consulta.

        El EnsembleRetriever no corta en k: devuelve todo lo que fusionó, que
        con dos retrievers de 5 resultados puede llegar a 10. El corte se hace
        acá.
        """
        if modo not in self.retrievers:
            raise ValueError(f"Modo '{modo}' no soportado: {list(self.retrievers)}")

        inicio = time.perf_counter()
        documentos = self.retrievers[modo].invoke(consulta)[: self.k]

        logger.info(
            "[%s] %d fragmentos en %.2fs",
            modo,
            len(documentos),
            time.perf_counter() - inicio,
        )
        return documentos


# ---------------------------------------------------------------------------


def mostrar(documentos: list[Document]) -> None:
    for posicion, doc in enumerate(documentos, 1):
        m = doc.metadata
        print(f"\n{posicion}. {m['fuente']}  ·  {m['seccion']}  ·  {m['categoria']}")
        print(f"   {doc.page_content[:200].strip()!r}...")


def main() -> None:
    configurar_logs()

    if len(sys.argv) < 2:
        raise SystemExit('Uso: python rag.py "pregunta"')

    consulta = " ".join(sys.argv[1:])
    sistema = RAGSystem()

    print(f"\nConsulta: {consulta}")
    mostrar(sistema.buscar(consulta))


if __name__ == "__main__":
    main()
