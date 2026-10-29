"""
Chequeos sobre el código fuente, para los dos errores que la consigna marca:

- llamadas sincrónicas que bloquean el event loop dentro del sistema async;
- nombres de modelos, índices o URLs escritos fuera de config.py.
"""

import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def archivos(excepto: set[str] = frozenset()) -> list[Path]:
    return [p for p in APP.rglob("*.py") if p.name not in excepto]


def buscar(patron: str, excepto: set[str] = frozenset()) -> list[str]:
    hallazgos = []
    for archivo in archivos(excepto):
        for numero, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(patron, linea):
                hallazgos.append(f"{archivo.relative_to(APP)}:{numero}: {linea.strip()}")
    return hallazgos


def test_no_hay_llamadas_que_bloqueen_el_event_loop():
    bloqueantes = r"^\s*(import requests|from requests)|time\.sleep\(|asyncio\.to_thread\("
    assert buscar(bloqueantes) == []


def test_no_hay_modelos_ni_indices_fuera_de_config():
    escritos = r"claude-(sonnet|opus|haiku)|gemini-embedding|httpx-docs|redis://"
    assert buscar(escritos, excepto={"config.py"}) == []
