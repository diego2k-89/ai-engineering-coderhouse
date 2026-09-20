"""
El contrato de salida del pipeline.

Este archivo cumple una función distinta a la del módulo 1. Allá los esquemas
validaban lo que NOSOTROS le mandábamos a la API. Acá validan lo que el MODELO
nos devuelve, y además viajan hasta el modelo: LangChain convierte esta clase
en un JSON Schema y se lo pasa al proveedor como definición de herramienta.

Por eso los `description` de cada campo importan tanto. No son comentarios para
quien lee el código: son parte de la instrucción que recibe el modelo.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class NivelCriticidad(str, Enum):
    """Valores permitidos para la criticidad.

    Un Enum y no un string libre: si fuera str, el modelo podría devolver
    "high", "ALTA", "crítica" o "media-alta" y todos pasarían la validación.
    Con el Enum, el esquema que ve el modelo dice exactamente estas tres
    opciones y cualquier otra cosa se rechaza.
    """

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    """Lo que el pipeline devuelve para cualquier texto de entrada."""

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Tecnologías, frameworks, lenguajes o herramientas mencionadas "
            "explícitamente en el texto. Usar el nombre propio tal como se "
            "escribe habitualmente (por ejemplo 'PostgreSQL', no 'postgres'). "
            "No inventar tecnologías que el texto no menciona."
        ),
    )

    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "Gravedad del problema descripto o relevancia de la arquitectura. "
            "'alta' si hay usuarios afectados o servicio caído, 'media' si hay "
            "degradación sin impacto directo, 'baja' si es informativo."
        ),
    )

    resumen_tecnico: str = Field(
        ...,
        min_length=10,
        description=(
            "Resumen técnico de una o dos oraciones sobre lo que describe el "
            "texto. Mencionar el componente afectado y la causa, si se deduce."
        ),
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Normaliza la lista después de que el modelo la devolvió.

        Dos cosas que los LLMs hacen seguido: dejar strings vacíos o con
        espacios sobrantes, y repetir la misma tecnología con distinta
        capitalización ("FastAPI" y "fastapi" en la misma lista).
        """
        limpias: list[str] = []
        vistas: set[str] = set()

        for valor in valores:
            valor = valor.strip()
            if not valor or valor.lower() in vistas:
                continue
            vistas.add(valor.lower())
            limpias.append(valor)

        if not limpias:
            raise ValueError("La lista de tecnologías quedó vacía después de limpiarla")

        return limpias
