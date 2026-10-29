"""
Tests de la lógica del grafo que no necesita al modelo: el ruteo con el
presupuesto de aprobación, las herramientas de los agentes y los contratos de
Pydantic.
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.agents import research_agent
from app.agents.analyst_agent import calcular, comparar_con_limite
from app.config import PRESUPUESTO_DERIVACIONES
from app.graph.grafo import construir_grafo, despues_de_aprobacion, enrutar
from app.graph.hitl import requiere_aprobacion
from app.schemas import Contribucion, Decision, EstadoJob, Job


def estado(siguiente: str, derivaciones: int, aprobado: bool = False) -> dict:
    return {
        "siguiente_agente": siguiente,
        "contribuciones": [{"agente": "investigador"}] * derivaciones,
        "presupuesto_aprobado": aprobado,
    }


# --- Ruteo y aprobación humana ---------------------------------------------


def test_dentro_del_presupuesto_va_directo_al_especialista():
    assert enrutar(estado("analista", PRESUPUESTO_DERIVACIONES - 1)) == "analista"


def test_superar_el_presupuesto_pasa_por_aprobacion():
    assert enrutar(estado("investigador", PRESUPUESTO_DERIVACIONES)) == "aprobacion"


def test_con_presupuesto_aprobado_no_vuelve_a_pausar():
    assert enrutar(estado("investigador", PRESUPUESTO_DERIVACIONES + 1, aprobado=True)) == "investigador"


def test_finalizar_nunca_pide_aprobacion():
    assert enrutar(estado("FINALIZAR", PRESUPUESTO_DERIVACIONES + 3)) == "sintesis"
    assert not requiere_aprobacion(estado("investigador", 0))


def test_rechazo_lleva_a_la_sintesis():
    assert despues_de_aprobacion(estado("FINALIZAR", 3)) == "sintesis"
    assert despues_de_aprobacion(estado("analista", 3, aprobado=True)) == "analista"


def test_el_grafo_tiene_los_cinco_nodos(monkeypatch):
    # Construir el grafo no llama al modelo: alcanza con una key de mentira.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-de-prueba")
    grafo = construir_grafo(checkpointer=None, recuperador=None)
    nodos = set(grafo.get_graph().nodes) - {"__start__", "__end__"}
    assert nodos == {"supervisor", "investigador", "analista", "aprobacion", "sintesis"}


# --- Herramientas del analista ------------------------------------------------


def test_calcular_resuelve_aritmetica_y_rechaza_codigo():
    assert calcular.invoke({"expresion": "100 / 250 * 100"})["resultado"] == 40.0
    assert calcular.invoke({"expresion": "__import__('os').getcwd()"})["error"]
    assert calcular.invoke({"expresion": "9 ** 9 ** 9"})["error"]
    assert calcular.invoke({"expresion": "1 / 0"})["error"] == "división por cero"


def test_comparar_con_limite():
    r = comparar_con_limite.invoke({"valor": 250, "limite": 100, "descripcion": "conexiones"})
    assert r["supera_el_limite"] is True
    assert r["diferencia"] == 150
    assert r["porcentaje_del_limite"] == 250


def test_herramientas_validan_su_entrada():
    # args_schema: un argumento con el tipo equivocado no llega a la función.
    with pytest.raises(ValidationError):
        calcular.invoke({"expresion": ""})
    with pytest.raises(ValidationError):
        comparar_con_limite.invoke({"valor": "muchas", "limite": 100, "descripcion": "x"})


# --- Herramienta del investigador ---------------------------------------------


class RecuperadorQueFalla:
    """Falla las primeras `fallas` búsquedas y después devuelve un fragmento."""

    def __init__(self, fallas: int):
        self.fallas = fallas
        self.llamadas = 0

    async def buscar(self, consulta: str) -> list[Document]:
        self.llamadas += 1
        if self.llamadas <= self.fallas:
            raise ConnectionError("Pinecone no responde")
        return [Document("max_connections=100", metadata={"fuente": "advanced/resource-limits.md", "seccion": "Pool"})]


async def _sin_espera(segundos):
    return None


def test_busqueda_reintenta_y_devuelve_fragmentos(monkeypatch):
    monkeypatch.setattr(research_agent, "asyncio", SimpleNamespace(sleep=_sin_espera))
    recuperador = RecuperadorQueFalla(fallas=2)
    herramienta = research_agent.crear_herramienta_busqueda(recuperador)

    resultado = asyncio.run(herramienta.ainvoke({"consulta": "max_connections default"}))

    assert recuperador.llamadas == 3
    assert resultado["error"] is None
    assert resultado["resultados"][0]["fuente"] == "advanced/resource-limits.md"


def test_busqueda_que_falla_siempre_devuelve_el_error_como_dato(monkeypatch):
    monkeypatch.setattr(research_agent, "asyncio", SimpleNamespace(sleep=_sin_espera))
    herramienta = research_agent.crear_herramienta_busqueda(RecuperadorQueFalla(fallas=99))

    resultado = asyncio.run(herramienta.ainvoke({"consulta": "max_connections default"}))

    assert resultado["resultados"] == []
    assert "Pinecone no responde" in resultado["error"]


# --- Contratos de Pydantic ----------------------------------------------------


def test_decision_solo_acepta_agentes_conocidos():
    with pytest.raises(ValidationError):
        Decision(razon="x", siguiente="programador")


def test_contribucion_valida_agente_y_paso():
    with pytest.raises(ValidationError):
        Contribucion(paso=0, agente="investigador", instruccion="x", resultado="x")
    with pytest.raises(ValidationError):
        Contribucion(paso=1, agente="supervisor", instruccion="x", resultado="x")


def test_job_ida_y_vuelta_por_json():
    job = Job(job_id="1", estado=EstadoJob.DONE, pregunta="¿x?", creado=1.0, actualizado=2.0)
    assert Job.model_validate_json(job.model_dump_json()) == job
