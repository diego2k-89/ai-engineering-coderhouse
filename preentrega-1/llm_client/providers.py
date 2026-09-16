"""
Implementaciones concretas, una por proveedor.

Cada clase traduce entre nuestro formato interno (ChatMessage, ModelResponse)
y el formato particular de su SDK. Toda la fealdad de "cada API es distinta"
vive acá adentro y no sale de este archivo.
"""

from collections.abc import AsyncGenerator

from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APIError as AnthropicAPIError
from anthropic import AsyncAnthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from google import genai
from google.genai import types
from openai import APIConnectionError as OpenAIConnectionError
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI
from openai import RateLimitError as OpenAIRateLimitError

from .base import BaseLLMClient
from .schemas import ChatMessage, LLMConfig, ModelResponse, Provider, Role


# Códigos HTTP que representan un problema pasajero: tiene sentido reintentar.
# 429 = demasiadas peticiones; 5xx = el servidor del proveedor está en problemas.
_CODIGOS_REINTENTABLES = {408, 429, 500, 502, 503, 504}


class LLMStreamError(Exception):
    """Un proveedor falló mientras enviaba el stream.

    En modo normal el error viaja adentro del ModelResponse, pero un generador
    no puede devolver un objeto: solo puede emitir fragmentos o cortarse.
    Esta excepción es la señal que el manager escucha para probar el siguiente
    proveedor de la cadena.
    """


def _separar_system(messages: list[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
    """Separa el mensaje de sistema del resto.

    OpenAI acepta el system como un mensaje más dentro de la lista.
    Anthropic y Gemini lo quieren aparte, en un parámetro propio.
    Esta función existe para que esa diferencia no se note desde afuera.
    """
    system = None
    resto = []
    for m in messages:
        if m.role == Role.SYSTEM:
            system = m.content
        else:
            resto.append(m)
    return system, resto


class OpenAIClient(BaseLLMClient):
    """Cliente de OpenAI.

    En este proyecto se usa a propósito sin API key válida, para simular una
    caída de proveedor y poder ver el fallback en acción.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
        )

    def _a_dicts(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    def _fallo(self, detalle: str, *, retryable: bool = False) -> ModelResponse:
        return ModelResponse(
            provider=Provider.OPENAI,
            model=self.config.model,
            error=detalle,
            retryable=retryable,
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            respuesta = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._a_dicts(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.config.model,
                content=respuesta.choices[0].message.content or "",
            )
        except OpenAIRateLimitError as e:
            return self._fallo(f"Límite de cuota excedido: {e}", retryable=True)
        except OpenAIConnectionError as e:
            return self._fallo(f"Error de conexión: {e}", retryable=True)
        except OpenAIAPIError as e:
            return self._fallo(f"Error de la API de OpenAI: {e}")

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._a_dicts(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                fragmento = chunk.choices[0].delta.content
                if fragmento:
                    yield fragmento
        except (OpenAIRateLimitError, OpenAIConnectionError, OpenAIAPIError) as e:
            raise LLMStreamError(f"OpenAI falló durante el streaming: {e}") from e


class AnthropicClient(BaseLLMClient):
    """Cliente de Anthropic.

    Diferencias contra OpenAI: el system va en un parámetro aparte,
    max_tokens es obligatorio, y el texto viene en respuesta.content[0].text.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncAnthropic(
            api_key=config.api_key.get_secret_value(),
            timeout=config.timeout_seconds,
        )

    def _armar_kwargs(self, messages: list[ChatMessage]) -> dict:
        # Ojo con temperature: el SDK 1.x de Anthropic la sacó de la firma de
        # messages.create(), y los modelos actuales (Sonnet 5, Opus 5) devuelven
        # 400 ante cualquier valor de sampling que no sea el default. Así que
        # este cliente directamente no la manda.
        # La temperature sigue validada en LLMConfig y sigue aplicando a OpenAI
        # y a Gemini: absorber esta clase de diferencias es para lo que existe
        # la capa de abstracción.
        system, resto = _separar_system(messages)
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": m.role.value, "content": m.content} for m in resto],
        }
        if system:
            kwargs["system"] = system
        return kwargs

    def _fallo(self, detalle: str, *, retryable: bool = False) -> ModelResponse:
        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=self.config.model,
            error=detalle,
            retryable=retryable,
        )

    @staticmethod
    def _extraer_texto(respuesta) -> str:
        """Junta el texto de la respuesta, salteando los bloques que no lo son.

        La respuesta de Anthropic es una lista de bloques, no un string. Los
        modelos con razonamiento extendido ponen primero un ThinkingBlock, que
        no tiene .text. Agarrar content[0] a ciegas se rompe con esos modelos.
        """
        return "".join(
            bloque.text
            for bloque in respuesta.content
            if getattr(bloque, "type", None) == "text"
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            respuesta = await self._client.messages.create(**self._armar_kwargs(messages))
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=self.config.model,
                content=self._extraer_texto(respuesta),
            )
        except AnthropicRateLimitError as e:
            return self._fallo(f"Límite de cuota excedido: {e}", retryable=True)
        except AnthropicConnectionError as e:
            return self._fallo(f"Error de conexión: {e}", retryable=True)
        except AnthropicAPIError as e:
            return self._fallo(f"Error de la API de Anthropic: {e}")

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._armar_kwargs(messages)) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except (
            AnthropicRateLimitError,
            AnthropicConnectionError,
            AnthropicAPIError,
        ) as e:
            raise LLMStreamError(f"Anthropic falló durante el streaming: {e}") from e


class GeminiClient(BaseLLMClient):
    """Cliente de Google Gemini.

    Es el más distinto de los tres: al rol del asistente lo llama "model",
    los mensajes son objetos types.Content en vez de diccionarios, y el
    cliente asíncrono se alcanza por la propiedad .aio.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = genai.Client(api_key=config.api_key.get_secret_value())

    def _armar_entrada(self, messages: list[ChatMessage]):
        system, resto = _separar_system(messages)
        contents = [
            types.Content(
                role="model" if m.role == Role.ASSISTANT else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in resto
        ]
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            system_instruction=system,
        )
        return contents, config

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        contents, config = self._armar_entrada(messages)
        try:
            respuesta = await self._client.aio.models.generate_content(
                model=self.config.model, contents=contents, config=config
            )
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.config.model,
                content=respuesta.text or "",
            )
        except Exception as e:
            # El SDK de Gemini no expone una jerarquía de excepciones tan clara
            # como los otros dos, así que acá sí atajamos de forma amplia y
            # miramos el código HTTP para decidir si vale la pena reintentar.
            codigo = getattr(e, "code", None)
            return ModelResponse(
                provider=Provider.GEMINI,
                model=self.config.model,
                error=f"Error de la API de Gemini: {e}",
                retryable=codigo in _CODIGOS_REINTENTABLES,
            )

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        contents, config = self._armar_entrada(messages)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.config.model, contents=contents, config=config
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMStreamError(f"Gemini falló durante el streaming: {e}") from e
