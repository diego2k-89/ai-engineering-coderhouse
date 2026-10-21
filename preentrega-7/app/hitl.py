"""
Human-in-the-loop: la pausa para aprobación humana.

Qué se considera crítico: el **costo**. Cada derivación a un especialista es
una ejecución de un agente ReAct completo, con llamadas al modelo y, en el caso
del investigador, a Pinecone y a la API de embeddings. Una pregunta típica usa
dos derivaciones (investigador y analista). Si el supervisor quiere una más de
las que permite PRESUPUESTO_DERIVACIONES, el grafo se detiene y espera a que una
persona autorice el gasto.

    supervisor --(dentro del presupuesto)--> especialista
               --(lo supera, sin aprobar)--> aprobacion --(sí)--> especialista
                                                        --(no)--> sintesis

Dos piezas:

- `nodo_aprobacion`: llama a `interrupt()`, que detiene el grafo y guarda el
  estado en el checkpointer de Redis bajo el thread_id del job. El proceso no
  queda esperando: el worker libera el job y lo marca WAITING_APPROVAL.
- `router` con POST /tasks/{id}/approve: recibe la decisión y la encola como
  un pedido de "reanudar". El worker sigue el grafo con
  `Command(resume=decision)`, y ese valor es lo que devuelve `interrupt()`.

Como el estado vive en Redis y no en memoria, la API se puede reiniciar entre la
pausa y la aprobación.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from langgraph.types import interrupt
from pydantic import BaseModel

from app import jobs
from app.config import PRESUPUESTO_DERIVACIONES
from app.state import EstadoEquipo

logger = logging.getLogger("api.hitl")


def requiere_aprobacion(estado: EstadoEquipo) -> bool:
    """Indica si la próxima derivación supera el presupuesto sin aprobación.

    La cuenta son las contribuciones ya hechas: cada vuelta de un especialista
    agrega una. Una aprobación vale para el resto de la tarea, no para una sola
    derivación más.
    """
    if estado.get("presupuesto_aprobado"):
        return False
    return len(estado.get("contribuciones", [])) >= PRESUPUESTO_DERIVACIONES


async def nodo_aprobacion(estado: EstadoEquipo) -> dict:
    """Pausa el grafo hasta que una persona apruebe o rechace seguir gastando.

    Al reanudar, LangGraph vuelve a ejecutar este nodo desde el principio y
    `interrupt()` devuelve la decisión en lugar de pausar. Por eso no hay nada
    con efectos antes del `interrupt()`: se ejecutaría dos veces.
    """
    decision = interrupt(
        {
            "motivo": (
                f"La tarea ya hizo {len(estado['contribuciones'])} derivaciones a "
                f"especialistas (presupuesto: {PRESUPUESTO_DERIVACIONES}) y el "
                "supervisor quiere una más."
            ),
            "siguiente_agente": estado["siguiente_agente"],
            "instruccion": estado["instruccion"],
        }
    )

    aprobado = bool(decision.get("aprobado")) if isinstance(decision, dict) else False
    logger.info("Decisión humana: %s", "aprobado" if aprobado else "rechazado")

    registro = {
        "paso": estado["pasos"],
        "razon": f"Aprobación humana: {'aprobado' if aprobado else 'rechazado'}",
        "siguiente": estado["siguiente_agente"] if aprobado else "FINALIZAR",
        "instruccion": estado["instruccion"] if aprobado else "",
    }
    if aprobado:
        return {"presupuesto_aprobado": True, "decisiones": [registro]}
    # Rechazado: la síntesis cierra con lo que ya aportaron los especialistas.
    return {"siguiente_agente": "FINALIZAR", "decisiones": [registro]}


# ---------------------------------------------------------------------------
# Endpoint de aprobación
# ---------------------------------------------------------------------------

router = APIRouter()


class DecisionHumana(BaseModel):
    aprobado: bool


@router.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
async def aprobar(job_id: str, decision: DecisionHumana, request: Request) -> dict:
    """Registra la decisión humana sobre un job pausado y lo devuelve a la cola."""
    redis = request.app.state.redis
    job = await jobs.leer(redis, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    if job["estado"] != jobs.Estado.WAITING_APPROVAL:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"El job está en {job['estado']}, no esperando aprobación",
        )

    await jobs.actualizar(redis, job_id, estado=jobs.Estado.PENDING, aprobacion=None)
    await jobs.encolar(redis, job_id, "reanudar", aprobado=decision.aprobado)
    return {"job_id": job_id, "estado": jobs.Estado.PENDING, "aprobado": decision.aprobado}
