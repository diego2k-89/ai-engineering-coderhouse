"""
Pipeline de ingesta: carga la documentación de httpx, la limpia, la fragmenta y
la sube a Pinecone en lotes.

    .md -> limpiar -> fragmentar -> metadatos -> embeddings -> upsert

Hay dos tamaños de lote distintos, porque hay dos límites distintos. Los
embeddings van de a pocos fragmentos y con pausa: el plan gratuito de Google
limita cuánto se puede vectorizar por minuto. El upsert a Pinecone va en lotes
grandes, como recomienda la clase. El cuello de botella de la ingesta es la API
de embeddings, no la base vectorial.

Cada fragmento se sube con un ID fijo (`<archivo>#<n° de fragmento>`). Volver a
correr la ingesta pisa los vectores existentes en vez de duplicarlos.

Las funciones de carga y fragmentación también las usa rag.py: el retriever BM25
vive en memoria y se reconstruye en cada corrida a partir de los mismos
fragmentos que están en Pinecone. Si los dos fragmentaran distinto, el
EnsembleRetriever no podría combinar sus resultados.

Uso:
    python ingest.py                   # sube o actualiza los fragmentos
    python ingest.py --solo-fragmentar # muestra los fragmentos, sin tocar Pinecone
    python ingest.py --reset           # vacía el namespace y sube todo de cero
"""

import argparse
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    INDEX_NAME,
    NAMESPACE,
    configurar_logs,
    get_embeddings,
    get_pinecone,
)

logger = logging.getLogger("rag.ingesta")

# Medidos en tokens. La consigna sugiere entre 500 y 800.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Vectores por upsert. La clase recomienda lotes de 100 a 200.
TAMANIO_LOTE = int(os.getenv("BATCH_SIZE", "100"))

# Fragmentos por llamada a la API de embeddings y segundos de pausa entre
# llamadas. Mandar los 83 fragmentos en una sola llamada devolvió 429.
EMBED_TAMANIO_LOTE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_PAUSA = float(os.getenv("EMBED_PAUSE_SECONDS", "15"))
EMBED_REINTENTOS = 4

# Categoría de cada archivo de primer nivel, tomada de la navegación del sitio
# de httpx (mkdocs.yml). Todo lo que está en advanced/ es "avanzado".
CATEGORIAS = {
    "index.md": "introduccion",
    "quickstart.md": "introduccion",
    "async.md": "guias",
    "http2.md": "guias",
    "logging.md": "guias",
    "compatibility.md": "guias",
    "troubleshooting.md": "guias",
    "api.md": "referencia",
    "exceptions.md": "referencia",
    "environment_variables.md": "referencia",
}

_TITULO = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------


def limpiar_markdown(texto: str) -> str:
    """Saca el marcado propio de MkDocs y normaliza el texto.

    La documentación de httpx está escrita para MkDocs. En el sitio web ese
    marcado se reemplaza por contenido; en el .md crudo es ruido:

    - `::: httpx.Client` inserta la documentación generada desde el código. Se
      conserva el nombre (`httpx.Client`), que sirve para la búsqueda léxica.
    - `:docstring:` es una opción de esa directiva y no aporta nada.
    - `:members: get post ...` lista métodos: se conservan los nombres.
    - `!!! note "Título"` abre un recuadro de aviso. Se conserva el título si
      lo tiene; el contenido del recuadro queda como está.

    Además normaliza los finales de línea: los archivos clonados en Windows
    vienen con CRLF.
    """
    lineas = []

    for linea in texto.replace("\r\n", "\n").split("\n"):
        limpia = linea.strip()

        if limpia == ":docstring:":
            continue
        if limpia.startswith(":::"):
            linea = limpia.removeprefix(":::").strip()
        elif limpia.startswith(":members:"):
            linea = limpia.removeprefix(":members:").strip()
        elif limpia.startswith("!!!"):
            titulo = re.search(r'"([^"]+)"', limpia)
            if not titulo:
                continue
            linea = titulo.group(1)

        lineas.append(linea.rstrip())

    texto = "\n".join(lineas)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def _limpiar_titulo(titulo: str) -> str:
    """Saca links y backticks de un título de Markdown."""
    return _LINK.sub(r"\1", titulo).replace("`", "").strip()


def titulos_fuera_de_codigo(texto: str) -> list[tuple[int, int, str]]:
    """Posición, nivel y texto de cada título (#, ##, ###) del documento.

    Saltea los bloques de código: adentro de ``` una línea que empieza con #
    es un comentario de Python (`# Using the top-level API:`), no un título.
    Sin este control, la mitad de las "secciones" serían comentarios.
    """
    titulos = []
    posicion = 0
    en_codigo = False

    for linea in texto.split("\n"):
        if linea.lstrip().startswith("```"):
            en_codigo = not en_codigo
        elif not en_codigo:
            coincidencia = _TITULO.match(linea)
            if coincidencia:
                nivel = len(coincidencia.group(1))
                texto_titulo = _limpiar_titulo(coincidencia.group(2))
                titulos.append((posicion, nivel, texto_titulo))
        posicion += len(linea) + 1

    return titulos


def cargar_documentos(carpeta: Path = DATA_DIR) -> list[Document]:
    """Lee los .md de la carpeta (y subcarpetas) y los devuelve limpios.

    La `fuente` es la ruta relativa con barras normales (`advanced/ssl.md`),
    igual en Windows y en Linux, porque es lo que se compara contra el Golden
    Set.
    """
    archivos = sorted(p for p in carpeta.rglob("*.md") if p.name != "LICENSE.md")
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .md en {carpeta}")

    documentos = []
    for archivo in archivos:
        fuente = archivo.relative_to(carpeta).as_posix()
        contenido = limpiar_markdown(archivo.read_text(encoding="utf-8"))

        # El título del documento es su primer título de nivel 1. Varios
        # archivos no tienen (arrancan con texto o con un ##); ahí el nombre del
        # archivo hace de título: timeouts.md -> "Timeouts".
        principales = [
            t for _, nivel, t in titulos_fuera_de_codigo(contenido) if nivel == 1
        ]
        titulo = (
            principales[0]
            if principales
            else archivo.stem.replace("-", " ").replace("_", " ").capitalize()
        )

        categoria = "avanzado" if fuente.startswith("advanced/") else CATEGORIAS.get(
            fuente, "otros"
        )

        documentos.append(
            Document(
                page_content=contenido,
                metadata={"fuente": fuente, "categoria": categoria, "titulo": titulo},
            )
        )

    logger.info("Cargados %d documentos desde %s", len(documentos), carpeta)
    return documentos


# ---------------------------------------------------------------------------
# Fragmentación
# ---------------------------------------------------------------------------


def _seccion_en(
    titulos: list[tuple[int, int, str]], posicion: int, por_defecto: str
) -> str:
    """Último título que aparece antes de una posición del texto."""
    seccion = por_defecto
    for inicio, _, titulo in titulos:
        if inicio > posicion:
            break
        seccion = titulo
    return seccion


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Parte los documentos en fragmentos con metadatos e ID fijo.

    El splitter combina dos cosas:

    - `from_tiktoken_encoder`: el tamaño se mide en tokens, no en caracteres.
    - Los separadores de Markdown: antes de cortar por párrafo, prueba cortar
      por títulos y por bloques de código. Es el chunking "Markdown aware" de la
      clase: una sección o un bloque de código quedan enteros siempre que
      entren en el tamaño.

    Los separadores de Markdown son expresiones regulares (`\\n#{1,6} `), así
    que `is_separator_regex=True` es obligatorio. Sin eso se buscarían como
    texto literal, nunca coincidirían, y el corte sería el genérico sin avisar.

    La consigna pide "página" como metadato. Un .md no tiene páginas: el
    equivalente es la sección, el último título antes de cada fragmento.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(
            Language.MARKDOWN
        ),
        is_separator_regex=True,
        add_start_index=True,
    )

    fragmentos = []
    for documento in documentos:
        titulos = titulos_fuera_de_codigo(documento.page_content)

        for n, fragmento in enumerate(splitter.split_documents([documento])):
            inicio = fragmento.metadata.pop("start_index", 0)
            fragmento.metadata["seccion"] = _seccion_en(
                titulos, inicio, documento.metadata["titulo"]
            )
            fragmento.metadata["fragmento"] = n
            fragmento.id = f"{documento.metadata['fuente']}#{n}"
            fragmentos.append(fragmento)

    logger.info(
        "%d documentos -> %d fragmentos (chunk_size=%d, overlap=%d)",
        len(documentos),
        len(fragmentos),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return fragmentos


def cargar_fragmentos() -> list[Document]:
    """Carga y fragmenta en un paso. Es lo que usa rag.py para armar BM25."""
    return fragmentar(cargar_documentos())


# ---------------------------------------------------------------------------
# Subida a Pinecone
# ---------------------------------------------------------------------------


def a_vector(fragmento: Document, valores: list[float]) -> dict:
    """Arma el registro que se sube a Pinecone.

    El texto del fragmento va dentro de la metadata, como pide la consigna: una
    sola consulta a Pinecone devuelve el vector, el texto y la fuente, sin
    necesidad de otra base de datos para recuperar el contenido.
    """
    return {
        "id": fragmento.id,
        "values": valores,
        "metadata": {**fragmento.metadata, "texto": fragmento.page_content},
    }


def _es_limite_de_uso(error: Exception) -> bool:
    """Indica si el error es un 429 de la API de embeddings."""
    texto = str(error)
    return "429" in texto or "RESOURCE_EXHAUSTED" in texto


def vectorizar_lote(embeddings, textos: list[str]) -> list[list[float]]:
    """Vectoriza un lote de textos, reintentando si la API responde 429.

    Un 429 significa que se superó un límite de uso. Si es el límite por
    minuto, alcanza con esperar a que se renueve la ventana: la espera arranca
    en 30 s y se duplica en cada intento, como el backoff del módulo 1. Si
    después de todos los intentos sigue fallando, lo más probable es que se haya
    agotado la cuota diaria, y ahí esperar no sirve.

    Cualquier otro error se propaga sin reintentar: una key inválida no se
    arregla esperando.
    """
    espera = 30

    for intento in range(1, EMBED_REINTENTOS + 1):
        try:
            return embeddings.embed_documents(textos)
        except Exception as error:
            if not _es_limite_de_uso(error):
                raise
            if intento == EMBED_REINTENTOS:
                raise RuntimeError(
                    "La API de embeddings sigue respondiendo 429 después de "
                    f"{EMBED_REINTENTOS} intentos. Probablemente se agotó la cuota "
                    "diaria del plan gratuito. Los límites del proyecto se ven en "
                    "https://ai.dev/rate-limit"
                ) from error

            logger.warning(
                "Límite de uso de la API de embeddings (429), intento %d de %d. "
                "Reintentando en %ds",
                intento,
                EMBED_REINTENTOS,
                espera,
            )
            time.sleep(espera)
            espera *= 2


def generar_embeddings(fragmentos: list[Document], embeddings) -> list[list[float]]:
    """Vectoriza todos los fragmentos en lotes chicos, con pausa entre lotes."""
    valores = []
    total = len(fragmentos)

    for desde in range(0, total, EMBED_TAMANIO_LOTE):
        if desde:
            time.sleep(EMBED_PAUSA)

        lote = fragmentos[desde : desde + EMBED_TAMANIO_LOTE]
        inicio = time.perf_counter()
        valores.extend(vectorizar_lote(embeddings, [f.page_content for f in lote]))

        logger.info(
            "Embeddings %d-%d de %d en %.2fs",
            desde + 1,
            desde + len(lote),
            total,
            time.perf_counter() - inicio,
        )

    return valores


def subir_en_lotes(
    indice, fragmentos: list[Document], valores: list[list[float]]
) -> None:
    """Hace upsert de a TAMANIO_LOTE vectores.

    Subir de a un vector es lento por el costo de red de cada llamada; subir
    todo junto puede superar el tamaño máximo de una petición. Los lotes son el
    punto medio.
    """
    vectores = [a_vector(f, v) for f, v in zip(fragmentos, valores)]
    total = len(vectores)

    for desde in range(0, total, TAMANIO_LOTE):
        lote = vectores[desde : desde + TAMANIO_LOTE]
        inicio = time.perf_counter()

        indice.upsert(vectors=lote, namespace=NAMESPACE)

        logger.info(
            "Upsert %d-%d de %d en %.2fs",
            desde + 1,
            desde + len(lote),
            total,
            time.perf_counter() - inicio,
        )


def vaciar_namespace(indice) -> None:
    """Borra todos los vectores del namespace, si existe."""
    estadisticas = indice.describe_index_stats()
    if NAMESPACE in (estadisticas.namespaces or {}):
        logger.warning("Vaciando el namespace '%s'", NAMESPACE)
        indice.delete(delete_all=True, namespace=NAMESPACE)


def mostrar_resumen(fragmentos: list[Document]) -> None:
    """Cuenta fragmentos por categoría y por archivo, para revisar el chunking."""
    por_categoria = Counter(f.metadata["categoria"] for f in fragmentos)
    por_archivo = Counter(f.metadata["fuente"] for f in fragmentos)

    print("\nFragmentos por categoría:")
    for categoria, cantidad in sorted(por_categoria.items()):
        print(f"  {categoria:14} {cantidad}")

    print("\nFragmentos por archivo:")
    for fuente, cantidad in sorted(por_archivo.items()):
        print(f"  {fuente:32} {cantidad}")

    ejemplo = fragmentos[0]
    print(f"\nEjemplo, {ejemplo.id}:")
    for clave, valor in ejemplo.metadata.items():
        print(f"  {clave:10} {valor}")
    print(f"  {'texto':10} {ejemplo.page_content[:120]!r}...")


# ---------------------------------------------------------------------------


def main() -> None:
    configurar_logs()

    parser = argparse.ArgumentParser(description="Ingesta de documentos a Pinecone")
    parser.add_argument(
        "--solo-fragmentar",
        action="store_true",
        help="Muestra los fragmentos sin generar embeddings ni tocar Pinecone",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vacía el namespace antes de subir",
    )
    args = parser.parse_args()

    fragmentos = cargar_fragmentos()
    mostrar_resumen(fragmentos)

    if args.solo_fragmentar:
        return

    pc = get_pinecone()
    if not pc.has_index(INDEX_NAME):
        raise SystemExit(
            f"No existe el índice '{INDEX_NAME}'. "
            "Correr primero: python setup_index.py"
        )

    indice = pc.Index(name=INDEX_NAME)
    if args.reset:
        vaciar_namespace(indice)

    inicio = time.perf_counter()
    logger.info(
        "Subiendo %d fragmentos al namespace '%s' del índice '%s'",
        len(fragmentos),
        NAMESPACE,
        INDEX_NAME,
    )
    valores = generar_embeddings(fragmentos, get_embeddings())
    subir_en_lotes(indice, fragmentos, valores)
    logger.info("Ingesta completa en %.2fs", time.perf_counter() - inicio)

    # En Serverless las estadísticas se actualizan con unos segundos de demora:
    # justo después del upsert el conteo puede mostrar menos vectores.
    estadisticas = indice.describe_index_stats()
    en_namespace = estadisticas.namespaces.get(NAMESPACE)
    logger.info(
        "Vectores en el namespace '%s': %s (total del índice: %s)",
        NAMESPACE,
        en_namespace.vector_count if en_namespace else 0,
        estadisticas.total_vector_count,
    )


if __name__ == "__main__":
    main()
