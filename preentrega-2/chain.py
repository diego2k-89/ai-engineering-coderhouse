"""
La cadena LCEL del pipeline.

    prompt | modelo_estructurado | verificacion

Los tres pasos se componen con el operador `|`. Cada uno es un Runnable, así
que la cadena entera también lo es: se puede invocar, reintentar y ejecutar de
forma asíncrona sin escribir código de pegamento.
"""

import logging
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from schemas import EntidadesTecnicas

load_dotenv()

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# ChatPromptTemplate y no una f-string: LangChain gestiona la variable {texto},
# el prompt queda versionable por separado y se puede inspeccionar sin ejecutar
# la cadena.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un analista técnico senior. Extraés información estructurada de "
            "textos técnicos: logs de error, descripciones de arquitectura y "
            "reportes de incidentes.\n"
            "\n"
            "Reglas:\n"
            "- Extraé únicamente tecnologías mencionadas de forma explícita en el "
            "texto. No infieras ni completes con lo que suele acompañarlas.\n"
            "- El nivel de criticidad se juzga por el impacto descripto, no por el "
            "tono del texto. Que alguien escriba nervioso no lo vuelve crítico.\n"
            "- El resumen es técnico y descriptivo, no una recomendación.\n"
            "- Si el texto es vago o no tiene contenido técnico, decilo en el "
            "resumen en vez de completar con suposiciones.",
        ),
        ("human", "{texto}"),
    ]
)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

# (variable de entorno del modelo, modelo por defecto)
_MODELOS: dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "gemini": ("GEMINI_MODEL", "gemini-flash-latest"),
}


def get_model(provider: str, max_tokens: int | None = None):
    """Fábrica de modelos: mismo criterio que el factory del Módulo 1.

    `temperature=0` es intencional. Para extracción estructurada queremos
    determinismo: el mismo texto debería dar el mismo resultado. La
    creatividad acá es un defecto, no una virtud.

    `max_tokens` se puede forzar desde afuera. Sirve para la demo de
    truncamiento: con un presupuesto ridículamente bajo, el modelo corta la
    respuesta a mitad de camino y se puede ver la detección funcionando.
    """
    provider = provider.strip().lower()
    if provider not in _MODELOS:
        raise ValueError(f"Proveedor no soportado: {provider}")

    var_modelo, modelo_default = _MODELOS[provider]
    modelo = os.getenv(var_modelo, modelo_default)
    temperatura = float(os.getenv("LLM_TEMPERATURE", "0"))
    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    if provider == "openai":
        return ChatOpenAI(model=modelo, temperature=temperatura, max_tokens=max_tokens)

    if provider == "anthropic":
        return ChatAnthropic(
            model=modelo, temperature=temperatura, max_tokens=max_tokens
        )

    return ChatGoogleGenerativeAI(
        model=modelo, temperature=temperatura, max_output_tokens=max_tokens
    )


# ---------------------------------------------------------------------------
# Verificación de la respuesta cruda
# ---------------------------------------------------------------------------


class RespuestaIncompleta(Exception):
    """El modelo cortó la respuesta antes de terminar de escribirla."""


# Cada proveedor nombra distinto el motivo de corte, y con valores distintos.
# OpenAI: finish_reason="length" | Anthropic: stop_reason="max_tokens"
# Gemini: finish_reason="MAX_TOKENS"
_CLAVES_DE_CORTE = ("finish_reason", "stop_reason")
_MOTIVOS_DE_CORTE = {"length", "max_tokens"}


def _fue_truncada(mensaje) -> bool:
    """¿El modelo se quedó sin tokens a mitad de camino?"""
    metadata = getattr(mensaje, "response_metadata", None) or {}

    for clave in _CLAVES_DE_CORTE:
        valor = metadata.get(clave)
        if isinstance(valor, str) and valor.lower() in _MOTIVOS_DE_CORTE:
            return True

    return False


def verificar(salida: dict) -> EntidadesTecnicas:
    """Revisa la respuesta cruda antes de dar el objeto por bueno.

    Con `include_raw=True`, el modelo devuelve un diccionario con tres claves:
    `raw` (el mensaje completo, con sus metadatos), `parsed` (el objeto
    Pydantic, o None) y `parsing_error`.

    Ese acceso al mensaje crudo es lo que permite detectar una respuesta
    truncada. Un JSON cortado a mitad de camino a veces sigue siendo válido:
    si el modelo iba a listar cinco tecnologías y alcanzó a escribir tres, el
    esquema lo acepta y el objeto parece correcto aunque esté incompleto.
    """
    mensaje = salida.get("raw")
    objeto = salida.get("parsed")
    error_de_parseo = salida.get("parsing_error")

    if _fue_truncada(mensaje):
        logger.warning("Respuesta truncada por límite de tokens. Reintentando...")
        raise RespuestaIncompleta(
            "El modelo cortó la respuesta por falta de tokens; el objeto puede "
            "estar incompleto"
        )

    if error_de_parseo is not None:
        logger.warning("La salida no cumplió el esquema: %s", error_de_parseo)
        raise error_de_parseo

    if objeto is None:
        raise RespuestaIncompleta("El modelo no devolvió una salida parseable")

    logger.info("Salida validada: %s", objeto.model_dump())
    return objeto


# ---------------------------------------------------------------------------
# La cadena
# ---------------------------------------------------------------------------


def build_chain(provider: str | None = None, max_tokens: int | None = None):
    """Arma la cadena LCEL completa para un proveedor.

    `.with_retry()` va sobre la cadena entera y no solo sobre el modelo: así
    también reintenta cuando el que falla es el paso de verificación, que es
    justamente donde detectamos la respuesta truncada.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")

    modelo = get_model(provider, max_tokens=max_tokens)
    modelo_estructurado = modelo.with_structured_output(
        EntidadesTecnicas, include_raw=True
    )

    return (PROMPT | modelo_estructurado | RunnableLambda(verificar)).with_retry(
        stop_after_attempt=int(os.getenv("LLM_MAX_RETRIES", "3")),
        wait_exponential_jitter=True,
    )


async def process_text(text: str, provider: str | None = None) -> EntidadesTecnicas:
    """Procesa un texto y devuelve el objeto validado.

    Si tras todos los reintentos no se logra una salida válida, la excepción
    se propaga: quien llama decide qué hacer. A diferencia del Módulo 1, acá
    no hay un objeto de error que devolver — la consigna pide que el pipeline
    devuelva siempre un objeto validado o nada.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
    logger.info("[%s] Procesando texto de %d caracteres", provider, len(text))

    cadena = build_chain(provider)

    try:
        resultado = await cadena.ainvoke({"texto": text})
    except Exception as e:
        logger.error("[%s] Falló tras los reintentos: %s", provider, e)
        raise

    logger.info("[%s] Listo", provider)
    return resultado
