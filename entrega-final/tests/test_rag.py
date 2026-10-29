"""
Tests del RAG híbrido sin Pinecone: la fusión RRF con retrievers falsos y el
tokenizador de BM25.
"""

import asyncio

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.rag.recuperador import RecuperadorHibrido, tokenizar


class RetrieverFijo(BaseRetriever):
    """Devuelve siempre los mismos documentos, en el mismo orden."""

    textos: list[str]

    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(t) for t in self.textos]

    async def _aget_relevant_documents(self, query, *, run_manager):
        return self._get_relevant_documents(query, run_manager=run_manager)


def test_rrf_premia_lo_que_aparece_en_los_dos_y_pesa_mas_lo_vectorial():
    # BM25 trae [A, B] y Pinecone [B, C]. Con pesos 0,3 / 0,7:
    #   B aparece en los dos       -> primero
    #   C es 2.º en el vectorial    -> le gana a A, que es 1.º en BM25
    recuperador = RecuperadorHibrido(
        bm25=RetrieverFijo(textos=["A", "B"]),
        vectorial=RetrieverFijo(textos=["B", "C"]),
        k=3,
    )
    documentos = asyncio.run(recuperador.buscar("cualquier cosa"))
    assert [d.page_content for d in documentos] == ["B", "C", "A"]


def test_buscar_corta_en_top_k():
    recuperador = RecuperadorHibrido(
        bm25=RetrieverFijo(textos=["A", "B", "C"]),
        vectorial=RetrieverFijo(textos=["D", "E", "F"]),
        k=2,
    )
    assert len(asyncio.run(recuperador.buscar("x"))) == 2


def test_tokenizar_conserva_nombres_con_guion_bajo():
    assert tokenizar("¿Qué hace `trust_env`?") == ["qué", "hace", "trust_env"]
