"""
Tests de la API y del worker: estados del job, errores, validación con Pydantic
y la pausa de aprobación humana. Corren sin API keys: el grafo y Redis son falsos.
"""

import time

from langgraph.types import Command

from tests.conftest import PAUSA_DE_PRUEBA, GrafoFalso


def esperar_estado(cliente, job_id: str, estados: set[str], limite: float = 5.0) -> dict:
    """Hace polling hasta que el job llega a uno de los estados esperados."""
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        job = cliente.get(f"/tasks/{job_id}").json()
        if job["estado"] in estados:
            return job
        time.sleep(0.05)
    raise AssertionError(f"El job no llegó a {estados}; quedó en {job['estado']}")


def crear_tarea(cliente, pregunta: str = "¿Qué es trust_env?") -> str:
    return cliente.post("/tasks", json={"pregunta": pregunta}).json()["job_id"]


def test_crear_tarea_responde_sin_esperar_al_grafo(armar_cliente):
    # El grafo tarda 2 s: si el endpoint lo esperara, la respuesta tardaría eso.
    cliente = armar_cliente(GrafoFalso(demora=2.0))

    inicio = time.perf_counter()
    respuesta = cliente.post("/tasks", json={"pregunta": "¿Qué es trust_env?"})
    segundos = time.perf_counter() - inicio

    assert respuesta.status_code == 202
    assert respuesta.json()["estado"] == "PENDING"
    assert respuesta.json()["job_id"]
    assert segundos < 0.5


def test_job_termina_en_done_con_respuesta(armar_cliente):
    cliente = armar_cliente(GrafoFalso())
    job = esperar_estado(cliente, crear_tarea(cliente), {"DONE", "FAILED"})

    assert job["estado"] == "DONE"
    assert job["respuesta"] == "respuesta de prueba"
    assert job["agentes"] == ["investigador", "analista"]
    assert job["segundos"] is not None


def test_excepcion_en_el_grafo_marca_failed(armar_cliente):
    cliente = armar_cliente(GrafoFalso(modo="falla"))
    job = esperar_estado(cliente, crear_tarea(cliente), {"DONE", "FAILED"})

    assert job["estado"] == "FAILED"
    assert "Pinecone no responde" in job["error"]


def test_job_inexistente_da_404(armar_cliente):
    cliente = armar_cliente(GrafoFalso())
    assert cliente.get("/tasks/no-existe").status_code == 404


def test_pedidos_mal_formados_dan_422(armar_cliente):
    # Pydantic valida antes de que el endpoint corra.
    cliente = armar_cliente(GrafoFalso())
    assert cliente.post("/tasks", json={"pregunta": ""}).status_code == 422
    assert cliente.post("/tasks", json={"pregunta": "x" * 2001}).status_code == 422
    assert cliente.post("/tasks", json={"otra_cosa": "hola"}).status_code == 422


def test_pausa_y_aprobacion(armar_cliente):
    grafo = GrafoFalso(modo="pausa")
    cliente = armar_cliente(grafo)
    job_id = crear_tarea(cliente, "pregunta larga")

    job = esperar_estado(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
    assert job["estado"] == "WAITING_APPROVAL"
    assert job["aprobacion"] == PAUSA_DE_PRUEBA

    respuesta = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert respuesta.status_code == 202

    job = esperar_estado(cliente, job_id, {"DONE", "FAILED"})
    assert job["estado"] == "DONE"
    assert job["aprobacion"] is None
    # El segundo tramo se corrió con la decisión del humano.
    reanudacion = grafo.entradas[-1]
    assert isinstance(reanudacion, Command)
    assert reanudacion.resume == {"aprobado": True}


def test_aprobar_un_job_que_no_esta_pausado_da_409(armar_cliente):
    cliente = armar_cliente(GrafoFalso())
    job_id = crear_tarea(cliente)
    esperar_estado(cliente, job_id, {"DONE"})

    respuesta = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert respuesta.status_code == 409


def test_health_informa_redis_y_worker(armar_cliente):
    cliente = armar_cliente(GrafoFalso())
    assert cliente.get("/health").json() == {"redis": True, "worker": True}
