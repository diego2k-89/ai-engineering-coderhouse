"""
Tests de la lógica del grafo que no necesita al modelo: el ruteo con el
presupuesto de aprobación y las herramientas del analista.
"""

from app.agents.analyst_agent import calcular, comparar_con_limite
from app.config import PRESUPUESTO_DERIVACIONES
from app.graph import despues_de_aprobacion, enrutar
from app.hitl import requiere_aprobacion


def estado(siguiente: str, derivaciones: int, aprobado: bool = False) -> dict:
    return {
        "siguiente_agente": siguiente,
        "contribuciones": [{"agente": "investigador"}] * derivaciones,
        "presupuesto_aprobado": aprobado,
    }


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


def test_calcular_resuelve_aritmetica_y_rechaza_codigo():
    assert calcular.invoke({"expresion": "100 / 250 * 100"})["resultado"] == 40.0
    assert "error" in calcular.invoke({"expresion": "__import__('os').getcwd()"})
    assert "error" in calcular.invoke({"expresion": "9 ** 9 ** 9"})
    assert calcular.invoke({"expresion": "1 / 0"})["error"] == "división por cero"


def test_comparar_con_limite():
    r = comparar_con_limite.invoke({"valor": 250, "limite": 100, "descripcion": "conexiones"})
    assert r["supera_el_limite"] is True
    assert r["diferencia"] == 150
