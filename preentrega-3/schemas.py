"""
Los contratos de datos del sistema RAG.

Hay una separación deliberada entre lo que genera el LLM y lo que devuelve el
sistema. El modelo solo escribe el texto de la respuesta; las fuentes las arma
el código a partir de los metadatos reales de los fragmentos recuperados.

El motivo: si además se le pide al LLM que cite de dónde sacó la información,
inventa referencias. Nombres de archivo que no existen, números de sección
plausibles. Y una cita falsa es peor que ninguna, porque da una sensación de
verificabilidad que no existe.
"""

from pydantic import BaseModel, Field


class RespuestaLLM(BaseModel):
    """Lo único que el LLM tiene permitido generar.

    Este esquema es el que viaja al modelo a través del PydanticOutputParser.
    """

    respuesta: str = Field(
        ...,
        min_length=1,
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el "
            "CONTEXTO provisto. Si la información no está en el contexto, "
            "decirlo de forma explícita en vez de completar con conocimiento "
            "general."
        ),
    )

    encontro_respuesta: bool = Field(
        ...,
        description=(
            "true si el CONTEXTO contiene la información necesaria para "
            "responder la pregunta. false si no está y la respuesta es una "
            "declaración de que no se cuenta con ese dato."
        ),
    )


class FragmentoRecuperado(BaseModel):
    """Un fragmento que el retriever trajo de la base vectorial.

    Se incluye en la salida para poder auditar qué vio el modelo. Sin esto, un
    RAG es una caja negra: si la respuesta sale mal, no hay forma de saber si
    falló la recuperación o la generación.
    """

    fuente: str = Field(..., description="Archivo del que proviene el fragmento")
    extracto: str = Field(..., description="Primeros caracteres, para inspección")


class RAGResponse(BaseModel):
    """Lo que devuelve get_rag_response().

    Combina el texto del LLM con metadata verificable que arma el código.
    """

    pregunta: str
    respuesta: str
    encontro_respuesta: bool
    fuentes: list[str] = Field(
        default_factory=list,
        description="Archivos de origen de los fragmentos usados como contexto",
    )
    fragmentos: list[FragmentoRecuperado] = Field(default_factory=list)
    proveedor: str = Field(..., description="Qué LLM generó la respuesta")

    @property
    def cantidad_fragmentos(self) -> int:
        return len(self.fragmentos)
