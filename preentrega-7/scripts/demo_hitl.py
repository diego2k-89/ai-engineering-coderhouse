"""
Demo del flujo human-in-the-loop, en dos pasos separados.

Paso 1: crea una tarea que necesita más derivaciones que el presupuesto y espera
a que quede pausada.

    python scripts/demo_hitl.py

Paso 2: aprueba o rechaza. Entre el paso 1 y el 2 se puede reiniciar la API: el
grafo pausado está guardado en Redis.

    python scripts/demo_hitl.py --aprobar JOB_ID
    python scripts/demo_hitl.py --rechazar JOB_ID

Cada paso agrega su resultado a trazas/hitl.json.
"""

import argparse
import json
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TRAZAS = Path(__file__).resolve().parent.parent / "trazas"

# Pide dos datos de archivos distintos y una cuenta con los dos: lo habitual es
# que el supervisor derive dos veces al investigador y una al analista, una más
# que el presupuesto por defecto (2).
PREGUNTA = (
    "Necesito dos datos de httpx, buscalos por separado: el timeout por defecto y "
    "la cantidad máxima de conexiones keep-alive por defecto. Después calculá "
    "cuántos segundos suman 3 intentos que agoten ese timeout y cuántas conexiones "
    "keep-alive faltan para llegar a 50."
)


def esperar(cliente: httpx.Client, job_id: str, estados: set[str]) -> dict:
    while True:
        job = cliente.get(f"/tasks/{job_id}").json()
        print(f"  {job['estado']}")
        if job["estado"] in estados:
            return job
        time.sleep(2)


def registrar(evento: dict) -> None:
    TRAZAS.mkdir(exist_ok=True)
    archivo = TRAZAS / "hitl.json"
    eventos = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else []
    eventos.append(evento)
    archivo.write_text(json.dumps(eventos, ensure_ascii=False, indent=2), encoding="utf-8")


def iniciar(cliente: httpx.Client) -> None:
    job_id = cliente.post("/tasks", json={"pregunta": PREGUNTA}).json()["job_id"]
    print(f"Job {job_id} creado. Esperando...")
    job = esperar(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
    registrar({"paso": "iniciar", **job})

    if job["estado"] == "WAITING_APPROVAL":
        print(f"\nPausado: {job['aprobacion']['motivo']}")
        print(f"Quiere derivar a: {job['aprobacion']['siguiente_agente']}")
        print(f"\nPara seguir:\n  python scripts/demo_hitl.py --aprobar {job_id}")
        print(f"  python scripts/demo_hitl.py --rechazar {job_id}")
    else:
        print(f"\nTerminó en {job['estado']} sin pausarse: no superó el presupuesto.")
        print("Para forzar la pausa, reiniciar la API con PRESUPUESTO_DERIVACIONES=1.")


def decidir(cliente: httpx.Client, job_id: str, aprobado: bool) -> None:
    respuesta = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": aprobado})
    print(f"{respuesta.status_code} {respuesta.json()}")
    respuesta.raise_for_status()
    job = esperar(cliente, job_id, {"DONE", "FAILED", "WAITING_APPROVAL"})
    registrar({"paso": "aprobar" if aprobado else "rechazar", **job})
    print(f"\n{job.get('respuesta') or job.get('error')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo de aprobación humana")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--aprobar", metavar="JOB_ID")
    grupo.add_argument("--rechazar", metavar="JOB_ID")
    parser.add_argument("--url", default=BASE_URL)
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as cliente:
        if args.aprobar:
            decidir(cliente, args.aprobar, True)
        elif args.rechazar:
            decidir(cliente, args.rechazar, False)
        else:
            iniciar(cliente)


if __name__ == "__main__":
    main()
