"""
Agente investigador: busca en la documentación de httpx con el RAG híbrido.

Su única herramienta es la búsqueda. No calcula: si la tarea pide una cuenta,
eso es trabajo del analista.
"""

import asyncio
import logging

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.config import crear_modelo
from app.rag.recuperador import RecuperadorHibrido
from app.schemas import BusquedaEntrada, Fragmento, ResultadoBusqueda, a_dict

logger = logging.getLogger("agentes.investigador")

NOMBRE = "investigador"

# Reintentos de la búsqueda ante un error de red de Pinecone o de la API de
# embeddings: 1 s, después 2 s. Lo pidió la devolución de la pre-entrega 6.
REINTENTOS_BUSQUEDA = 3

PROMPT = """Sos el investigador de un equipo. Tu trabajo es encontrar datos en \
la documentación oficial de httpx, la librería HTTP de Python.

Reglas:
- Usá buscar_documentacion para cada dato. No respondas de memoria.
- Si la primera búsqueda no trae el dato, reformulá la consulta (en inglés, que
  es el idioma de la documentación) antes de darlo por no encontrado.
- No hagas cuentas: reportá los datos tal como aparecen.
- Respondé con los datos encontrados y, para cada uno, el archivo de donde sale.
- Si no encontrás algo, decilo explícitamente. No lo completes con suposiciones.
- Sé breve: los datos y su fuente, sin tablas ni recomendaciones."""


def crear_herramienta_busqueda(recuperador: RecuperadorHibrido):
    """Arma la herramienta sobre un recuperador ya abierto.

    El recuperador se abre una vez al arrancar la API (tiene conexiones a
    Pinecone que hay que cerrar al apagar), por eso la herramienta se crea acá
    y no a nivel de módulo.
    """

    @tool("buscar_documentacion", args_schema=BusquedaEntrada)
    async def buscar_documentacion(consulta: str) -> dict:
        """Busca en la documentación oficial de httpx y devuelve los fragmentos
        más relevantes para la consulta.

        Usar para cualquier dato sobre httpx: valores por defecto, parámetros,
        clases, excepciones, variables de entorno, configuración de clientes,
        timeouts, límites de conexiones, proxies, SSL, HTTP/2, etc.

        La documentación está en inglés: las consultas en inglés y con los
        nombres exactos (por ejemplo "max_connections default", "PoolTimeout")
        recuperan mejor.

        Devuelve {"consulta", "resultados": [{"fuente", "seccion", "texto"}]}.
        Si la búsqueda falla, "resultados" viene vacío y "error" dice por qué.
        """
        for intento in range(1, REINTENTOS_BUSQUEDA + 1):
            try:
                documentos = await recuperador.buscar(consulta)
                break
            except Exception as exc:  # noqa: BLE001 - el error vuelve al modelo como dato
                if intento == REINTENTOS_BUSQUEDA:
                    resultado = ResultadoBusqueda(consulta=consulta, error=f"{type(exc).__name__}: {exc}")
                    return a_dict(resultado)
                logger.warning("Búsqueda fallida (intento %d): %s", intento, exc)
                await asyncio.sleep(intento)

        resultado = ResultadoBusqueda(
            consulta=consulta,
            resultados=[
                Fragmento(
                    fuente=d.metadata.get("fuente", "?"),
                    seccion=d.metadata.get("seccion", "?"),
                    texto=d.page_content,
                )
                for d in documentos
            ],
        )
        return a_dict(resultado)

    return buscar_documentacion


def crear_investigador(recuperador: RecuperadorHibrido):
    return create_react_agent(
        crear_modelo(), [crear_herramienta_busqueda(recuperador)], prompt=PROMPT, name=NOMBRE
    )
