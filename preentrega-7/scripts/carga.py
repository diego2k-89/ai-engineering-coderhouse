"""
Prueba de carga: 5 peticiones concurrentes contra la API.

Manda las 5 preguntas al mismo tiempo, hace polling de cada job hasta que
termina y mide dos cosas:

- Cuánto tardó cada POST en responder: muestra que la API no espera al grafo.
- Cuánto tardó cada job de punta a punta (desde el POST hasta DONE), con su
  p50 y p95.

El costo por ejecución y la latencia p95 que pide la consigna se toman del
dashboard de LangSmith. Las latencias de acá sirven para contrastar.

Uso, con la API corriendo:
    python scripts/carga.py
"""

import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
TRAZAS = Path(__file__).resolve().parent.parent / "trazas"
ESPERA_POLLING = 1.0
LIMITE_SEGUNDOS = 300

PREGUNTAS = [
    "Un servicio necesita 250 conexiones simultáneas con un httpx.Client creado sin "
    "parámetros. ¿Alcanza el límite por defecto? Si no, ¿cuántas faltan?",
    "¿Qué excepción lanza httpx cuando no consigue una conexión del pool a tiempo?",
    "¿Cuál es el timeout por defecto de httpx y cuántos segundos suman 3 intentos "
    "que agoten ese timeout?",
    "¿Cuántas conexiones keep-alive mantiene httpx por defecto? ¿Qué porcentaje de "
    "50 representan?",
    "¿Qué variable de entorno usa httpx para indicar un archivo de certificados SSL?",
]


def percentil(valores: list[float], p: float) -> float:
    """Percentil por el método del rango más cercano.

    Con 5 valores, el p95 cae en la posición ceil(0,95 * 5) = 5: es el más lento.
    """
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


async def una_tarea(cliente: httpx.AsyncClient, numero: int, pregunta: str) -> dict:
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"pregunta": pregunta})
    respuesta.raise_for_status()
    segundos_post = time.perf_counter() - inicio
    job_id = respuesta.json()["job_id"]
    print(f"[{numero}] encolado en {segundos_post * 1000:.0f} ms -> {job_id}")

    while time.perf_counter() - inicio < LIMITE_SEGUNDOS:
        await asyncio.sleep(ESPERA_POLLING)
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in ("DONE", "FAILED", "WAITING_APPROVAL"):
            break

    total = time.perf_counter() - inicio
    print(f"[{numero}] {job['estado']} en {total:.1f} s")
    return {
        "numero": numero,
        "job_id": job_id,
        "pregunta": pregunta,
        "estado": job["estado"],
        "segundos_post": round(segundos_post, 3),
        "segundos_total": round(total, 2),
        "segundos_worker": job.get("segundos"),
        "agentes": job.get("agentes"),
        "respuesta": job.get("respuesta"),
        "error": job.get("error"),
    }


async def main(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as cliente:
        inicio = time.perf_counter()
        resultados = await asyncio.gather(
            *(una_tarea(cliente, n, p) for n, p in enumerate(PREGUNTAS, 1))
        )
        pared = time.perf_counter() - inicio

    terminados = [r["segundos_total"] for r in resultados if r["estado"] == "DONE"]
    posts = [r["segundos_post"] for r in resultados]
    resumen = {
        "peticiones": len(resultados),
        "done": len(terminados),
        "tiempo_total_pared_s": round(pared, 2),
        "post_max_ms": round(max(posts) * 1000),
        "latencia_p50_s": round(statistics.median(terminados), 2) if terminados else None,
        "latencia_p95_s": round(percentil(terminados, 95), 2) if terminados else None,
    }

    print("\nResumen:")
    for clave, valor in resumen.items():
        print(f"  {clave:22} {valor}")

    TRAZAS.mkdir(exist_ok=True)
    destino = TRAZAS / "carga.json"
    destino.write_text(
        json.dumps({"resumen": resumen, "resultados": resultados}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGuardado en {destino.relative_to(TRAZAS.parent)}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else BASE_URL))
