"""
AsyncLLMManager: el único punto de entrada del cliente.

Hace tres cosas que el resto del código no debería tener que saber hacer:
  1. Elegir qué cliente concreto instanciar, según configuración (Factory).
  2. Reintentar con espera creciente cuando el fallo es pasajero.
  3. Pasar al siguiente proveedor cuando el primero no se recupera (Fallback).
"""

import asyncio
import os
from collections.abc import AsyncGenerator

from pydantic import SecretStr

from .base import BaseLLMClient
from .providers import AnthropicClient, GeminiClient, LLMStreamError, OpenAIClient
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider

# Qué clase le corresponde a cada proveedor.
# Sumar un proveedor nuevo es agregar una línea acá, nada más.
_CLIENTES: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}

# De qué variables de entorno sale la configuración de cada uno.
# (variable de la key, variable del modelo, modelo por defecto)
_VARIABLES: dict[Provider, tuple[str, str, str]] = {
    Provider.OPENAI: ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    Provider.ANTHROPIC: ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-5"),
    Provider.GEMINI: ("GOOGLE_API_KEY", "GEMINI_MODEL", "gemini-flash-latest"),
}


def config_desde_entorno(provider: Provider) -> LLMConfig:
    """Arma la configuración de un proveedor leyendo el entorno.

    Las keys nunca se escriben en el código: salen del .env, que está
    excluido del repositorio.
    """
    var_key, var_modelo, modelo_default = _VARIABLES[provider]
    api_key = os.getenv(var_key, "").strip()

    if not api_key:
        # Sin key, el SDK ni siquiera deja construir el cliente y el programa
        # se cortaría acá. Con un valor de relleno, el cliente arranca y falla
        # recién al llamar a la API, con un 401 — que es justo lo que queremos
        # para ver el fallback funcionando.
        api_key = "sin-configurar"

    return LLMConfig(
        provider=provider,
        model=os.getenv(var_modelo, modelo_default),
        api_key=SecretStr(api_key),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    )


def crear_cliente(config: LLMConfig) -> BaseLLMClient:
    """Factory: dada una configuración, devuelve el cliente concreto.

    Este es el único lugar del proyecto que sabe que OpenAIClient existe.
    """
    clase = _CLIENTES.get(config.provider)
    if clase is None:
        raise ValueError(f"Proveedor no soportado: {config.provider}")
    return clase(config)


def cadena_desde_entorno() -> list[Provider]:
    """Lee LLM_PROVIDER y LLM_FALLBACK_CHAIN y arma el orden de intento."""
    nombres = [os.getenv("LLM_PROVIDER", "anthropic")]
    nombres += os.getenv("LLM_FALLBACK_CHAIN", "").split(",")

    cadena: list[Provider] = []
    for nombre in nombres:
        nombre = nombre.strip().lower()
        if not nombre:
            continue
        provider = Provider(nombre)  # si el nombre no existe, falla acá y se ve claro
        if provider not in cadena:
            cadena.append(provider)
    return cadena


class AsyncLLMManager:
    """Orquesta reintentos y fallback sobre una cadena de proveedores."""

    def __init__(
        self,
        cadena: list[Provider] | None = None,
        max_reintentos: int | None = None,
    ):
        self.cadena = cadena if cadena is not None else cadena_desde_entorno()
        if not self.cadena:
            raise ValueError("La cadena de proveedores quedó vacía")

        if max_reintentos is None:
            max_reintentos = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.max_reintentos = max_reintentos

        # Qué proveedor respondió en la última llamada. Sirve para mostrarlo
        # en streaming, donde el generador solo puede emitir texto.
        self.ultimo_proveedor: Provider | None = None

    # ---------- modo normal ----------

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la primera respuesta exitosa de la cadena.

        Si todos fallan, devuelve un ModelResponse con el detalle de cada
        fallo. Nunca levanta una excepción hacia afuera.
        """
        errores: list[str] = []

        for provider in self.cadena:
            respuesta = await self._intentar(provider, messages)
            if respuesta.ok:
                self.ultimo_proveedor = provider
                return respuesta
            errores.append(f"{provider.value}: {respuesta.error}")

        return ModelResponse(
            provider=self.cadena[-1],
            model="-",
            error="Todos los proveedores fallaron || " + " || ".join(errores),
        )

    async def _intentar(
        self, provider: Provider, messages: list[ChatMessage]
    ) -> ModelResponse:
        """Llama a un proveedor, reintentando con backoff exponencial."""
        try:
            cliente = crear_cliente(config_desde_entorno(provider))
        except Exception as e:
            return ModelResponse(
                provider=provider, model="-", error=f"No se pudo crear el cliente: {e}"
            )

        respuesta = await self._llamar(cliente, provider, messages)
        intento = 0

        # Solo se reintenta lo que puede mejorar esperando: rate limits y
        # caídas de red. Un 401 devuelve retryable=False y sale de una.
        while (
            not respuesta.ok
            and respuesta.retryable
            and intento < self.max_reintentos - 1
        ):
            espera = 2**intento  # 1s, 2s, 4s, ...
            await asyncio.sleep(espera)
            intento += 1
            respuesta = await self._llamar(cliente, provider, messages)

        return respuesta

    async def _llamar(
        self, cliente: BaseLLMClient, provider: Provider, messages: list[ChatMessage]
    ) -> ModelResponse:
        """Blinda la llamada a un cliente contra cualquier excepción.

        Los clientes ya atajan los errores conocidos de su API. Esto cubre lo
        que no previmos: un SDK que cambió una firma, un bug nuestro, lo que
        sea. Un proveedor roto degrada a un fallback; no tira abajo la cadena.
        """
        try:
            return await cliente.generate(messages)
        except Exception as e:
            return ModelResponse(
                provider=provider,
                model="-",
                error=f"Error inesperado: {type(e).__name__}: {e}",
            )

    # ---------- modo streaming ----------

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Emite los fragmentos del primer proveedor que logre responder."""
        errores: list[str] = []

        for provider in self.cadena:
            try:
                cliente = crear_cliente(config_desde_entorno(provider))
            except Exception as e:
                errores.append(f"{provider.value}: no se pudo crear el cliente: {e}")
                continue

            emitio_algo = False
            try:
                async for fragmento in cliente.generate_stream(messages):
                    if not emitio_algo:
                        self.ultimo_proveedor = provider
                        emitio_algo = True
                    yield fragmento
                return  # terminó bien, no hace falta seguir la cadena
            except Exception as e:
                if emitio_algo:
                    # Ya le mostramos texto al usuario. Cambiar de proveedor
                    # ahora mezclaría dos respuestas distintas en la misma
                    # pantalla, así que preferimos cortar.
                    raise
                errores.append(f"{provider.value}: {e}")

        raise LLMStreamError(
            "Todos los proveedores fallaron || " + " || ".join(errores)
        )
