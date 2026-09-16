"""
El contrato que todo cliente de LLM tiene que cumplir.

Este archivo no hace nada: solo declara qué métodos existen. La lógica real
vive en cada proveedor. Esa separación es la que permite cambiar de OpenAI a
Anthropic sin tocar una línea del código que los usa.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from .schemas import ChatMessage, ModelResponse


class BaseLLMClient(ABC):
    """Interfaz común a todos los proveedores.

    Heredar de ABC y marcar métodos con @abstractmethod hace que Python
    rechace instanciar una subclase que se olvidó de implementar alguno.
    El contrato se verifica solo, no depende de que uno se acuerde.
    """

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa, en una sola pieza.

        Nunca debe dejar escapar una excepción del proveedor: si algo falla,
        se devuelve un ModelResponse con el motivo en el campo `error`.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        """Devuelve la respuesta de a fragmentos, a medida que el modelo la genera."""
        raise NotImplementedError
        yield  # inalcanzable: solo le avisa a Python que esto es un generador
