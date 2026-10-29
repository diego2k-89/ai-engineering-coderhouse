"""
Las 5 pruebas de punta a punta, contra la API levantada y con las trazas en
LangSmith.

Cada prueba es una pregunta con un dato que la respuesta tiene que contener
(sale de la documentación de httpx o de una cuenta sobre ella). Las 5 se mandan
a la vez, así que también es una prueba de carga: la API tiene que responder
cada POST en milisegundos mientras el worker corre los 5 grafos en paralelo.

Al final muestra cuáles pasaron, las latencias (p50 y p95) y guarda todo en
trazas/pruebas.json. Cada job queda en LangSmith con su job_id como metadato.

Uso, con la API corriendo (docker compose up):
    python scripts/pruebas.py
    python scripts/pruebas.py http://otra-url:8000
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

# (nombre, pregunta, texto que la respuesta tiene que contener)
PRUEBAS = [
    (
        "limite_de_conexiones",
        "Un servicio necesita 250 conexiones simultáneas con un httpx.Client creado sin "
        "parámetros. ¿Alcanza el límite por defecto? Si no, ¿cuántas faltan?",
        "150",  # 250 - 100 (max_connections)
    ),
    (
        "excepcion_del_pool",
        "¿Qué excepción lanza httpx cuando no consigue una conexión del pool a tiempo?",
        "PoolTimeout",
    ),
    (
        "timeout_por_reintentos",
        "¿Cuál es el timeout por defecto de httpx y cuántos segundos suman 3 intentos "
        "que agoten ese timeout?",
        "15",  # 3 x 5 s
    ),
    (
        "porcentaje_keepalive",
        "¿Cuántas conexiones keep-alive mantiene httpx por defecto? ¿Qué porcentaje de "
        "50 representan?",
        "40",  # 20 / 50
    ),
    (
        "variable_de_entorno_ssl",
        "¿Qué variable de entorno usa httpx para indicar un archivo de certificados SSL?",
        "SSL_CERT_FILE",
    ),
]


def percentil(valores: list[float], p: float) -> float:
    """Percentil por el método del rango más cercano.

    Con 5 valores, el p95 cae en la posición ceil(0,95 * 5) = 5: es el más lento.
    """
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


async def una_prueba(cliente: httpx.AsyncClient, nombre: str, pregunta: str, esperado: str) -> dict:
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"pregunta": pregunta})
    respuesta.raise_for_status()
    segundos_post = time.perf_counter() - inicio
    job_id = respuesta.json()["job_id"]
    print(f"[{nombre}] encolado en {segundos_post * 1000:.0f} ms -> {job_id}")

    job = {"estado": "PENDING"}
    while time.perf_counter() - inicio < LIMITE_SEGUNDOS:
        await asyncio.sleep(ESPERA_POLLING)
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["estado"] in ("DONE", "FAILED", "WAITING_APPROVAL"):
            break

    total = time.perf_counter() - inicio
    paso = job["estado"] == "DONE" and esperado in (job.get("respuesta") or "")
    print(f"[{nombre}] {job['estado']} en {total:.1f} s -> {'OK' if paso else 'FALLA'}")
    return {
        "prueba": nombre,
        "job_id": job_id,
        "pregunta": pregunta,
        "esperado": esperado,
        "paso": paso,
        "estado": job["estado"],
        "segundos_post": round(segundos_post, 3),
        "segundos_total": round(total, 2),
        "agentes": job.get("agentes"),
        "respuesta": job.get("respuesta"),
        "error": job.get("error"),
    }


async def main(base_url: str) -> int:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as cliente:
        inicio = time.perf_counter()
        resultados = await asyncio.gather(*(una_prueba(cliente, *p) for p in PRUEBAS))
        pared = time.perf_counter() - inicio

    terminados = [r["segundos_total"] for r in resultados if r["estado"] == "DONE"]
    resumen = {
        "pruebas": len(resultados),
        "pasaron": sum(r["paso"] for r in resultados),
        "tiempo_total_pared_s": round(pared, 2),
        "post_max_ms": round(max(r["segundos_post"] for r in resultados) * 1000),
        "latencia_p50_s": round(statistics.median(terminados), 2) if terminados else None,
        "latencia_p95_s": round(percentil(terminados, 95), 2) if terminados else None,
    }

    print("\nResumen:")
    for clave, valor in resumen.items():
        print(f"  {clave:22} {valor}")

    TRAZAS.mkdir(exist_ok=True)
    destino = TRAZAS / "pruebas.json"
    destino.write_text(
        json.dumps({"resumen": resumen, "resultados": resultados}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGuardado en {destino.relative_to(TRAZAS.parent)}")
    return 0 if resumen["pasaron"] == resumen["pruebas"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else BASE_URL)))
