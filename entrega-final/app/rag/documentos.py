"""
Carga, limpieza y fragmentación de la documentación de httpx (de la pre-entrega 4).

Lo usan dos piezas, y tienen que fragmentar exactamente igual:

- `ingesta.py`, que sube los fragmentos a Pinecone.
- `recuperador.py`, que arma el índice BM25 en memoria con los mismos fragmentos.

El EnsembleRetriever reconoce que dos resultados son el mismo fragmento
comparando el texto. Si BM25 y Pinecone fragmentaran distinto, no podría
combinarlos.

Leer los archivos es una operación local de disco que se hace una sola vez al
arrancar, no una llamada de red: no pasa por el event loop.
"""

import logging
import os
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.config import DATA_DIR

logger = logging.getLogger("rag.documentos")

# Medidos en tokens. La consigna del módulo 4 sugería entre 500 y 800.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

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
    """Carga y fragmenta en un paso. Lo usan la ingesta y el retriever BM25."""
    return fragmentar(cargar_documentos())
