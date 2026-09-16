"""
Esquemas de datos del cliente LLM.

Todo lo que entra y sale del cliente pasa por acá. La idea es que un dato
mal formado se detecte en este archivo y no doce capas más abajo, cuando la
API devuelve un 400 y el mensaje de error no dice nada útil.
"""

from enum import Enum

from pydantic import BaseModel, Field, SecretStr


class Provider(str, Enum):
    """Proveedores soportados."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Role(str, Enum):
    """Roles válidos en una conversación.

    Gemini le dice "model" al asistente en vez de "assistant". Esa traducción
    la hace el cliente de Gemini, no este esquema: acá adentro siempre usamos
    los mismos tres nombres.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje de la conversación.

    Es el reemplazo del diccionario suelto {"role": ..., "content": ...}.
    Con un diccionario, un error de tipeo en la clave se descubre en runtime;
    acá lo levanta Pydantic al construir el objeto.
    """

    role: Role
    content: str = Field(min_length=1)


class LLMConfig(BaseModel):
    """Configuración de una llamada al modelo.

    Los rangos salen de la consigna: temperature entre 0 y 2, max_tokens positivo.
    """

    provider: Provider
    model: str = Field(min_length=1)
    api_key: SecretStr
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=8192)


class ModelResponse(BaseModel):
    """Respuesta del modelo, o el error que impidió obtenerla.

    Nunca dejamos escapar una excepción del proveedor hacia el código que llama.
    Un fallo se devuelve acá adentro, en `error`, y el programa sigue vivo.
    """

    provider: Provider
    model: str
    content: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None
