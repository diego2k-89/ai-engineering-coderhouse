"""
Ingesta: carga los documentos de /data, los limpia, los fragmenta y los
persiste en ChromaDB.

El índice se guarda en una carpeta distinta por cada modelo de embeddings:

    vectorstore/
    ├── google__gemini-embedding-001/
    │   ├── chroma/         <- la base de Chroma
    │   └── manifest.json   <- cómo se construyó este índice
    └── hf__all-MiniLM-L6-v2/
        └── ...

Ese namespacing resuelve el error #1 que marca la consigna: indexar con un
modelo y consultar con otro. No falla con una excepción: devuelve fragmentos
al azar, y eso es muy difícil de detectar. Con una carpeta por modelo, cambiar
de embeddings nunca contamina un índice existente, y se puede volver atrás sin
reindexar.

Uso:
    python ingest.py             # indexa si no existe el índice
    python ingest.py --reindex   # borra y vuelve a indexar desde cero
"""

import argparse
import json
import logging
import os
import re
import shutil
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Chroma manda telemetría anónima por defecto. Acá queda desactivada.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger("rag.ingesta")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

COLLECTION_NAME = "politicas_vantia"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _slug(texto: str) -> str:
    """Convierte un nombre de modelo en algo usable como nombre de carpeta."""
    texto = texto.strip().lower().replace("/", "-")
    return re.sub(r"[^a-z0-9._-]+", "-", texto).strip("-")


def get_embeddings() -> tuple[object, str, str]:
    """Devuelve (objeto_embeddings, provider, model_id) según el .env.

    Los imports van adentro de cada rama a propósito: así, no tener instalado
    langchain-huggingface no rompe la rama de Google, que es la predeterminada.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "google").strip().lower()

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        modelo = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(model=modelo), provider, modelo

    if provider in ("huggingface", "hf"):
        from langchain_huggingface import HuggingFaceEmbeddings

        modelo = os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        return HuggingFaceEmbeddings(model_name=modelo), "hf", modelo

    raise ValueError(
        f"EMBEDDING_PROVIDER='{provider}' no soportado. Usar 'google' o 'huggingface'."
    )


def ruta_del_indice(provider: str, modelo: str) -> Path:
    """Carpeta propia para este par (proveedor, modelo)."""
    return VECTORSTORE_DIR / f"{_slug(provider)}__{_slug(modelo)}"


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------


def limpiar_texto(texto: str) -> str:
    """Normaliza el texto antes de fragmentarlo.

    Los documentos reales traen ruido: espacios dobles, saltos de línea de más,
    caracteres de una extracción mal hecha. Ese ruido entra al embedding y lo
    ensucia, así que se limpia antes de vectorizar y no después.
    """
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)          # espacios repetidos
    texto = re.sub(r"\n{3,}", "\n\n", texto)       # más de un renglón en blanco
    texto = re.sub(r" +\n", "\n", texto)           # espacios al final de línea
    return texto.strip()


def cargar_documentos(carpeta: Path = DATA_DIR) -> list[Document]:
    """Lee los .txt y .md de la carpeta y los devuelve como Documents.

    Se usa pathlib en vez de DirectoryLoader de langchain-community: son seis
    líneas, el resultado es idéntico y evita sumar un paquete grande solo para
    leer cuatro archivos.
    """
    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {carpeta}")

    archivos = sorted([*carpeta.glob("*.txt"), *carpeta.glob("*.md")])
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .txt ni .md en {carpeta}")

    documentos = []
    for archivo in archivos:
        contenido = limpiar_texto(archivo.read_text(encoding="utf-8"))
        documentos.append(
            Document(
                page_content=contenido,
                # El metadato 'source' es lo que después se convierte en la
                # cita. Se guarda solo el nombre del archivo, no la ruta
                # absoluta, para no filtrar la estructura del disco.
                metadata={"source": archivo.name},
            )
        )
        logger.info("Cargado %s (%d caracteres)", archivo.name, len(contenido))

    return documentos


# ---------------------------------------------------------------------------
# Fragmentación
# ---------------------------------------------------------------------------


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Parte los documentos en chunks de 500 tokens con 50 de overlap.

    `from_tiktoken_encoder` es lo que hace que chunk_size=500 signifique
    literalmente 500 tokens. Sin eso, el splitter cuenta caracteres, y 500
    caracteres de español pueden ser 90 o 160 tokens según la puntuación y los
    acentos. La consigna pide tokens, no caracteres.

    RecursiveCharacterTextSplitter prueba una jerarquía de cortes antes de
    partir por la fuerza: párrafos, saltos de línea, espacios y recién al final
    caracteres sueltos. Así un corte casi nunca cae en medio de una oración.
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documentos)

    # Los fragmentos se numeran dentro de cada archivo, para poder citar con
    # más precisión que solo el nombre del documento.
    contador: dict[str, int] = {}
    for chunk in chunks:
        fuente = chunk.metadata.get("source", "desconocida")
        contador[fuente] = contador.get(fuente, 0) + 1
        chunk.metadata["chunk"] = contador[fuente]

    logger.info(
        "%d documentos -> %d fragmentos (chunk_size=%d, overlap=%d)",
        len(documentos),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def _escribir_manifiesto(
    ruta: Path, provider: str, modelo: str, n_chunks: int, segundos: float
) -> None:
    """Deja registrado cómo se construyó este índice.

    Si más adelante un índice devuelve resultados raros, el manifiesto dice con
    qué modelo de embeddings y con qué tamaño de chunk se armó.

    Guarda también cuánto tardó el indexado. Eso permite comparar el costo de un
    proveedor de embeddings contra otro sin volver a medir: los índices quedan
    lado a lado en disco, cada uno con su duración.
    """
    manifiesto = {
        "embedding_provider": provider,
        "embedding_model": modelo,
        "collection": COLLECTION_NAME,
        "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "fragmentos": n_chunks,
        "segundos_indexado": round(segundos, 2),
        "indexado": date.today().isoformat(),
    }
    (ruta / "manifest.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def leer_manifiesto(ruta: Path) -> dict | None:
    archivo = ruta / "manifest.json"
    if not archivo.exists():
        return None
    return json.loads(archivo.read_text(encoding="utf-8"))


def get_vectorstore(reindexar: bool = False) -> Chroma:
    """Devuelve la base vectorial, indexando solo si hace falta.

    La consigna marca la falta de persistencia como error común: reindexar en
    cada corrida cuesta tiempo y, con embeddings de pago, dinero.
    """
    embeddings, provider, modelo = get_embeddings()
    ruta = ruta_del_indice(provider, modelo)
    ruta_chroma = ruta / "chroma"

    if reindexar and ruta.exists():
        logger.warning("Borrando el índice existente en %s", ruta)
        shutil.rmtree(ruta)

    ya_existe = ruta_chroma.exists() and any(ruta_chroma.iterdir())

    if ya_existe:
        manifiesto = leer_manifiesto(ruta) or {}
        logger.info(
            "Índice existente (%s / %s, %s fragmentos): no se reindexa",
            manifiesto.get("embedding_provider", "?"),
            manifiesto.get("embedding_model", "?"),
            manifiesto.get("fragmentos", "?"),
        )
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(ruta_chroma),
        )

    logger.info("No hay índice para %s / %s: indexando", provider, modelo)

    # El indexado es la operación cara del módulo: una llamada de embeddings por
    # fragmento. Medirla con perf_counter (monótono, inmune a un ajuste del reloj
    # del sistema) es lo que da la referencia para saber cuánto ahorra la
    # persistencia en las corridas siguientes.
    inicio = time.perf_counter()

    chunks = fragmentar(cargar_documentos())

    ruta_chroma.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(ruta_chroma),
    )

    segundos = time.perf_counter() - inicio

    _escribir_manifiesto(ruta, provider, modelo, len(chunks), segundos)
    logger.info(
        "Indexados %d fragmentos en %s (%.2fs)", len(chunks), ruta_chroma, segundos
    )

    return vectorstore


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-8s %(name)s | %(message)s"
    )

    parser = argparse.ArgumentParser(description="Ingesta de documentos a ChromaDB")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Borra el índice de este modelo y lo reconstruye desde cero",
    )
    args = parser.parse_args()

    vectorstore = get_vectorstore(reindexar=args.reindex)
    total = vectorstore._collection.count()

    print(f"\nFragmentos en la colección: {total}")
    print("Índices disponibles en disco:")
    if VECTORSTORE_DIR.exists():
        for carpeta in sorted(VECTORSTORE_DIR.iterdir()):
            manifiesto = leer_manifiesto(carpeta) or {}
            print(
                f"  {carpeta.name:44} {manifiesto.get('fragmentos', '?')} fragmentos"
                f"  en {manifiesto.get('segundos_indexado', '?')}s"
                f"  ({manifiesto.get('indexado', 'fecha desconocida')})"
            )


if __name__ == "__main__":
    main()
