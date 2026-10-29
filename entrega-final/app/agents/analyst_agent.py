"""
Agente analista: hace cuentas y verifica valores contra límites.

No tiene acceso a la documentación. Trabaja solo con los datos que le pasa el
supervisor, así que si le falta uno lo tiene que decir en vez de inventarlo.

Las dos herramientas validan su entrada con Pydantic (`args_schema`) y arman la
salida con un modelo. Son cálculos locales, sin red: no tienen nada que esperar,
así que son funciones comunes y no async.
"""

import ast
import operator

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.config import crear_modelo
from app.schemas import (
    CalculoEntrada,
    ComparacionEntrada,
    ResultadoCalculo,
    ResultadoComparacion,
    a_dict,
)

NOMBRE = "analista"

PROMPT = """Sos el analista de un equipo. Tu trabajo es hacer cuentas y \
verificar valores, a partir de los datos que te pasan.

Reglas:
- Toda cuenta se hace con la herramienta calcular. No calcules de cabeza.
- Para decidir si un valor supera un límite, usá comparar_con_limite: el valor
  es lo que se necesita y el límite es el máximo permitido.
- Usá solo los datos que te dieron. Si falta un dato para la cuenta, decilo y
  no lo supongas.
- Respondé con cada resultado y la cuenta que lo produjo. Sé breve: sin tablas
  ni recomendaciones."""

# Operaciones permitidas. Cualquier otra cosa (nombres, llamadas, atributos) se
# rechaza: la expresión la escribe un modelo, y eval() ejecutaría cualquier código.
_OPERADORES = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluar(nodo: ast.AST) -> float:
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, (int, float)):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES:
        izquierda, derecha = _evaluar(nodo.left), _evaluar(nodo.right)
        # Un exponente enorme (9 ** 9 ** 9) dejaría el proceso colgado calculando.
        if isinstance(nodo.op, ast.Pow) and abs(derecha) > 100:
            raise ValueError("exponente mayor a 100")
        return _OPERADORES[type(nodo.op)](izquierda, derecha)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES:
        return _OPERADORES[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError(f"elemento no permitido: {type(nodo).__name__}")


@tool("calcular", args_schema=CalculoEntrada)
def calcular(expresion: str) -> dict:
    """Evalúa una expresión aritmética y devuelve el resultado.

    Acepta números, paréntesis y los operadores + - * / // % **. No acepta
    variables ni funciones: los valores tienen que estar escritos en la
    expresión. Ejemplos: "250 - 100", "100 / 250 * 100", "(20 + 5) * 3".

    Devuelve {"expresion", "resultado"}. Si la expresión no es válida, devuelve
    "error" con el motivo: corregirla y volver a llamar.
    """
    try:
        valor = _evaluar(ast.parse(expresion, mode="eval").body)
        resultado = ResultadoCalculo(expresion=expresion, resultado=round(valor, 4))
    except ZeroDivisionError:
        resultado = ResultadoCalculo(expresion=expresion, error="división por cero")
    except (SyntaxError, ValueError) as exc:
        resultado = ResultadoCalculo(expresion=expresion, error=f"expresión inválida ({exc})")
    return a_dict(resultado)


@tool("comparar_con_limite", args_schema=ComparacionEntrada)
def comparar_con_limite(valor: float, limite: float, descripcion: str) -> dict:
    """Compara un valor contra un límite e informa si lo supera.

    Usar cuando la pregunta es si algo "alcanza", "entra", "supera" o "excede"
    un máximo. `valor` es lo que se necesita o se quiere usar; `limite` es el
    máximo permitido. `descripcion` dice qué se compara, por ejemplo
    "conexiones necesarias vs max_connections".

    Devuelve {"descripcion", "valor", "limite", "supera_el_limite",
    "diferencia" (valor - limite), "porcentaje_del_limite" (valor / limite * 100)}.
    """
    resultado = ResultadoComparacion(
        descripcion=descripcion,
        valor=valor,
        limite=limite,
        supera_el_limite=valor > limite,
        diferencia=round(valor - limite, 4),
        porcentaje_del_limite=round(valor / limite * 100, 2) if limite else None,
    )
    return a_dict(resultado)


def crear_analista():
    return create_react_agent(
        crear_modelo(), [calcular, comparar_con_limite], prompt=PROMPT, name=NOMBRE
    )
