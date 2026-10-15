"""
Evaluación del recuperador con un Golden Set.

Cada pregunta de golden_set.json tiene un documento que la responde, conocido de
antemano. Se corre cada pregunta contra los tres modos del RAGSystem (BM25 solo,
vectorial solo e híbrido) y se calcula:

- Recall@5: ¿aparece el documento esperado entre los 5 fragmentos? (0 o 1)
- Precision@5: ¿qué fracción de los 5 fragmentos viene del documento esperado?
- MRR: 1 / posición del primer fragmento del documento esperado (0 si no está).

La consigna pide las dos primeras para el híbrido. La comparación contra cada
retriever por separado es lo que permite ver si la combinación aporta algo.

Uso:
    python evaluate.py                    # pesos por defecto, 0.3 / 0.7
    python evaluate.py --pesos 0.5 0.5    # pesos BM25 / vectorial
"""

import argparse
import json
import logging
import time
from collections import Counter

from config import BASE_DIR, configurar_logs
from rag import PESOS, TOP_K, RAGSystem

logger = logging.getLogger("rag.evaluacion")

GOLDEN_SET = BASE_DIR / "golden_set.json"
MODOS = ("bm25", "vectorial", "hibrido")


def cargar_golden_set(fuentes_validas: set[str]) -> list[dict]:
    """Lee el Golden Set y valida que cada documento esperado exista.

    Un nombre mal escrito no daría error: daría Recall 0, y se leería como que
    el retriever falló. Por eso se valida antes de evaluar.
    """
    casos = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))

    for n, caso in enumerate(casos, 1):
        faltan = {"pregunta", "documento_id_esperado"} - caso.keys()
        if faltan:
            raise ValueError(f"Pregunta {n} del Golden Set: faltan {sorted(faltan)}")
        if caso["documento_id_esperado"] not in fuentes_validas:
            raise ValueError(
                f"Pregunta {n}: '{caso['documento_id_esperado']}' no es ninguno de "
                "los documentos indexados"
            )

    return casos


def medir(fuentes: list[str], esperado: str, k: int) -> dict:
    """Recall@k, Precision@k y MRR de un resultado contra el documento esperado.

    Precision@k se divide por k y no por la cantidad devuelta, como pide la
    consigna: qué porcentaje de los 5 recuperados es útil.
    """
    posiciones = [i for i, fuente in enumerate(fuentes) if fuente == esperado]

    return {
        "recall": 1.0 if posiciones else 0.0,
        "precision": len(posiciones) / k,
        "mrr": 1 / (posiciones[0] + 1) if posiciones else 0.0,
    }


def techo_de_precision(fragmentos_por_fuente: Counter, esperado: str, k: int) -> float:
    """Precision@k máxima posible para una pregunta.

    Si el documento esperado tiene un solo fragmento, ni el retriever perfecto
    puede pasar de 1/k: los otros 4 lugares los ocupan fragmentos de otros
    documentos. Sin este número, una precisión de 20% parece mala y puede ser
    la máxima alcanzable.
    """
    return min(k, fragmentos_por_fuente[esperado]) / k


def promedio(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else 0.0


def main() -> None:
    configurar_logs()

    parser = argparse.ArgumentParser(description="Evaluación del recuperador")
    parser.add_argument(
        "--pesos",
        nargs=2,
        type=float,
        default=PESOS,
        metavar=("BM25", "VECTORIAL"),
        help=f"Pesos del EnsembleRetriever. Por defecto {PESOS[0]} {PESOS[1]}",
    )
    args = parser.parse_args()

    inicio = time.perf_counter()
    sistema = RAGSystem(k=TOP_K, pesos=tuple(args.pesos))
    fragmentos_por_fuente = Counter(d.metadata["fuente"] for d in sistema.bm25.docs)
    casos = cargar_golden_set(set(fragmentos_por_fuente))

    logger.info("Evaluando %d preguntas en %d modos", len(casos), len(MODOS))

    resultados = {modo: [] for modo in MODOS}
    techos = []

    print("\n" + "=" * 78)
    print("  DETALLE POR PREGUNTA")
    print("=" * 78)

    for caso in casos:
        esperado = caso["documento_id_esperado"]
        techo = techo_de_precision(fragmentos_por_fuente, esperado, TOP_K)
        techos.append(techo)

        print(f"\n[{caso.get('tipo', '-')}] {caso['pregunta']}")
        print(f"  esperado: {esperado}  (techo de Precision@{TOP_K}: {techo:.0%})")

        for modo in MODOS:
            documentos = sistema.buscar(caso["pregunta"], modo)
            fuentes = [d.metadata["fuente"] for d in documentos]
            metrica = medir(fuentes, esperado, TOP_K)
            resultados[modo].append({**metrica, "tipo": caso.get("tipo", "-")})

            marcas = " ".join("●" if f == esperado else "·" for f in fuentes)
            print(
                f"  {modo:10} R@{TOP_K} {metrica['recall']:.0f}   "
                f"P@{TOP_K} {metrica['precision']:4.0%}   "
                f"MRR {metrica['mrr']:.2f}   {marcas}"
            )

    print("\n" + "=" * 78)
    print("  RESUMEN")
    print("=" * 78)
    print(f"\n  {'modo':10} {'Recall@5':>10} {'Precision@5':>13} {'MRR':>7}")
    for modo in MODOS:
        filas = resultados[modo]
        print(
            f"  {modo:10} {promedio([f['recall'] for f in filas]):>10.0%} "
            f"{promedio([f['precision'] for f in filas]):>13.0%} "
            f"{promedio([f['mrr'] for f in filas]):>7.2f}"
        )
    print(f"  {'techo':10} {'100%':>10} {promedio(techos):>13.0%} {'1.00':>7}")

    tipos = sorted({f["tipo"] for f in resultados["hibrido"]})
    if len(tipos) > 1:
        print("\n  Por tipo de pregunta (Recall@5 / MRR):")
        print(f"  {'modo':10}" + "".join(f"{t:>22}" for t in tipos))
        for modo in MODOS:
            celdas = []
            for tipo in tipos:
                filas = [f for f in resultados[modo] if f["tipo"] == tipo]
                celdas.append(
                    f"{promedio([f['recall'] for f in filas]):.0%} / "
                    f"{promedio([f['mrr'] for f in filas]):.2f} ({len(filas)})"
                )
            print(f"  {modo:10}" + "".join(f"{c:>22}" for c in celdas))

    print("\n  ● fragmento del documento esperado   · fragmento de otro documento")
    logger.info("Evaluación completa en %.2fs", time.perf_counter() - inicio)


if __name__ == "__main__":
    main()
