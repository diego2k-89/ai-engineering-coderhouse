# Pre-entrega 4 - Sistema RAG escalable en la nube con Pinecone

Módulo de recuperación sobre la documentación oficial de
[httpx](https://www.python-httpx.org/) (20 páginas en Markdown), indexada en
Pinecone Serverless. Combina búsqueda léxica (BM25) con búsqueda vectorial en un
`EnsembleRetriever`, y se evalúa con un Golden Set de preguntas cuya respuesta se
conoce de antemano.

El objetivo no es que el recuperador parezca funcionar, es **medir si recupera
bien**, y comparar el híbrido contra cada retriever por separado.

## Requisitos

- Python 3.12 o superior.
- Una cuenta de [Pinecone](https://app.pinecone.io). El plan gratuito alcanza;
  solo permite AWS `us-east-1`, que es la región que usa el proyecto.
- Una API key de [Google AI Studio](https://aistudio.google.com/apikey), para
  los embeddings.

## Instalación

Desde la raíz del repo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r preentrega-4/requirements.txt
```

En Linux o Mac la activación es `source .venv/bin/activate`.

## Configuración

```bash
cd preentrega-4
copy .env.example .env
```

Y completar las dos keys. El `.env` está en el `.gitignore`.

| Variable | Descripción |
|---|---|
| `PINECONE_API_KEY` | Key de Pinecone |
| `GOOGLE_API_KEY` | Key de Google AI Studio |
| `INDEX_NAME` | Nombre del índice. Default `httpx-docs` |
| `PINECONE_NAMESPACE` | Default `httpx-docs` |
| `EMBEDDING_MODEL` | Default `models/gemini-embedding-001` |
| `EMBEDDING_DIM` | Default 1536 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | En tokens. Default 600 / 100 |
| `BATCH_SIZE` | Vectores por upsert. Default 100 |
| `EMBED_BATCH_SIZE` / `EMBED_PAUSE_SECONDS` | Lote y pausa para la API de embeddings. Default 20 / 15 |

La consigna menciona `OPENAI_API_KEY`; acá la reemplaza la de Google, que es la
que genera los embeddings. La de Anthropic no serviría: Anthropic no ofrece
modelos de embeddings.

## Replicar el índice

Dos pasos, desde `preentrega-4/`:

```bash
python setup_index.py
python ingest.py
```

1. **`setup_index.py`** mide la dimensión real de los embeddings y crea el índice
   Serverless si no existe. Si ya existe, verifica que su dimensión coincida con
   la del modelo y, si no coincide, corta antes de la ingesta.
2. **`ingest.py`** carga los 20 documentos, los fragmenta en 83 pedazos y los sube
   a Pinecone. Tarda alrededor de un minuto y medio, por las pausas entre lotes.

Para verificar que quedó todo arriba:

```bash
python -c "from config import *; print(get_pinecone().Index(name=INDEX_NAME).describe_index_stats())"
```

El namespace `httpx-docs` tiene que mostrar 83 vectores.

Opciones de la ingesta:

```bash
python ingest.py --solo-fragmentar   # muestra los fragmentos sin tocar ninguna API
python ingest.py --reset             # vacía el namespace y sube todo de cero
```

Volver a correr la ingesta sin `--reset` no duplica nada: cada fragmento tiene un
ID fijo (`advanced/ssl.md#0`) y el upsert pisa el vector existente.

## Consultar y evaluar

```bash
python rag.py "¿Para qué sirve trust_env?"
python evaluate.py
python evaluate.py --pesos 0.5 0.5
```

`rag.py` devuelve los 5 fragmentos del híbrido para una pregunta. `evaluate.py`
corre el Golden Set contra BM25 solo, vectorial solo e híbrido, e imprime las
métricas. `--pesos` cambia los pesos del ensemble (BM25, vectorial); por defecto
son 0.3 / 0.7, y el motivo está en "Qué observé".

## Archivos

```
data/              documentación de httpx (.md) y su LICENSE.md
golden_set.json    las 5 preguntas con su documento esperado
config.py          .env, modelo de embeddings y cliente de Pinecone compartidos
setup_index.py     verificación y creación del índice
ingest.py          limpieza, fragmentación, metadatos y subida en lotes
rag.py             PineconeRetriever, BM25 y RAGSystem
evaluate.py        Recall@5, Precision@5 y MRR
.env.example       plantilla de configuración
```

## Cómo funciona

### Ingesta

```
.md -> limpiar -> fragmentar -> metadatos -> embeddings -> upsert
```

**Limpieza.** La documentación de httpx está escrita para MkDocs, y el `.md`
crudo trae marcado que en el sitio web se reemplaza por contenido: directivas
como `::: httpx.Client` / `:docstring:` y recuadros `!!! note`. Se saca el
marcado y se conserva lo útil, como los nombres de clases y métodos.

**Fragmentación.** `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con los
separadores de Markdown: mide en tokens y, antes de partir por párrafo, prueba
cortar por títulos y bloques de código. Esos separadores son expresiones
regulares, así que `is_separator_regex=True` es obligatorio: sin eso se buscan
como texto literal, no coinciden nunca, y el corte vuelve al genérico sin avisar.

**Metadatos** de cada fragmento:

| Campo | Ejemplo |
|---|---|
| `fuente` | `advanced/timeouts.md` |
| `categoria` | `avanzado` (de la navegación del sitio: introducción, guías, avanzado, referencia) |
| `titulo` | `Timeouts` |
| `seccion` | `Fine tuning the configuration` |
| `fragmento` | `1` |
| `texto` | el contenido del fragmento |

La consigna pide "página" como metadato, pero un `.md` no tiene páginas. El
equivalente es la **sección**: el último título antes de cada fragmento. Para
calcularla hay que saltear los bloques de código, porque adentro de ellos una
línea con `#` es un comentario de Python y no un título.

El **texto va dentro de la metadata**, como pide la consigna: una consulta a
Pinecone devuelve el vector, el texto y la fuente, sin otra base de datos.

**Subida en lotes.** Hay dos tamaños de lote porque hay dos límites distintos:

| Paso | Lote | Motivo |
|---|---|---|
| Embeddings (Google) | 20, con 15 s de pausa | El plan gratuito limita cuánto se vectoriza por minuto |
| Upsert (Pinecone) | 100 | La recomendación de la clase |

Si la API de embeddings igual responde 429, se reintenta con backoff exponencial
(30 s, 60 s, 120 s). Cualquier otro error corta sin reintentar.

### Recuperación híbrida

```
pregunta -> BM25 (en memoria)   --\
                                    > RRF -> top-5
pregunta -> Pinecone (vectorial) --/
```

**BM25** se arma en memoria en cada corrida, con los mismos 83 fragmentos que
están en Pinecone. Usa un tokenizador propio: el de `BM25Retriever` por defecto
es `texto.split()`, que separa solo por espacios y distingue mayúsculas. Con eso,
una pregunta que dice `` `trust_env`? `` no coincide con `trust_env` en el
documento, por los backticks y el signo pegados.

**El retriever vectorial** es una clase propia, `PineconeRetriever`, sobre el SDK
de Pinecone. Hace lo que haría `PineconeVectorStore` de `langchain-pinecone`, que
no se puede usar en Python 3.14 (ver más abajo).

**`RAGSystem`** combina los dos en un `EnsembleRetriever`. La fusión es
**Reciprocal Rank Fusion**: suma `peso / (posición + 60)` por cada retriever, o
sea que combina por posición y no por puntaje. Los puntajes de BM25 y los de
similitud coseno están en escalas distintas y sumarlos directamente no tiene
sentido.

El ensemble no corta en top-k: devuelve todo lo que fusionó, hasta 10 resultados
con dos retrievers de 5. El top-5 lo aplica `RAGSystem`.

Para que el ensemble reconozca que dos resultados son el mismo fragmento, compara
el texto. Por eso BM25 y la ingesta usan la misma función de fragmentación: si se
edita algo en `data/`, hay que volver a correr `ingest.py`.

### Evaluación

El Golden Set tiene 5 preguntas en español, de dos tipos:

- **Léxicas** (2): nombran un término exacto que aparece en un solo archivo
  (`trust_env`, `MockTransport`). Es el terreno donde se espera que BM25 ayude.
- **Semánticas** (3): dicen lo mismo con otras palabras. "¿Cómo limito la
  cantidad máxima de conexiones?" cuando el documento dice `max_connections`.

Métricas, a nivel documento:

- **Recall@5**: ¿aparece el documento esperado entre los 5 fragmentos?
- **Precision@5**: ¿qué fracción de los 5 fragmentos viene del documento esperado?
- **MRR**: 1 / posición del primer fragmento correcto. Mide si el correcto sale
  primero o queda abajo.

**Precision@5 tiene un techo distinto en cada pregunta.** `resource-limits.md`
tiene un solo fragmento: aunque el retriever lo traiga primero, los otros cuatro
lugares son de otros documentos y la precisión no puede pasar de 20%. El script
calcula y muestra ese techo, para no leer como malo un número que es el máximo
posible.

## Resultados

Con los pesos por defecto (BM25 0.3, vectorial 0.7):

```
  modo         Recall@5   Precision@5     MRR
  bm25              60%           20%    0.47
  vectorial        100%           44%    1.00
  hibrido          100%           44%    1.00
  techo            100%           48%    1.00

  Por tipo de pregunta (Recall@5 / MRR):
  modo                      lexica             semantica
  bm25             100% / 1.00 (2)        33% / 0.11 (3)
  vectorial        100% / 1.00 (2)       100% / 1.00 (3)
  hibrido          100% / 1.00 (2)       100% / 1.00 (3)
```

El híbrido recupera el documento correcto en las 5 preguntas, siempre en el
primer lugar, con Precision@5 de 44% sobre un máximo posible de 48%.

Con pesos iguales (0.5 / 0.5), que es lo que usa la pista, el híbrido empeoraba:

```
  modo         Recall@5   Precision@5     MRR
  hibrido          100%           44%    0.70

  hibrido          100% / 1.00 (2)       100% / 0.50 (3)
```

En las tres preguntas semánticas el documento correcto caía del primer al
segundo lugar. La explicación está abajo.

## Qué observé al probarlo

**Con pesos iguales, el orden de la lista de retrievers decide el resultado.**
RRF suma `peso / (posición + 60)` por cada retriever. Con 0.5 / 0.5, el primer
resultado de BM25 y el primero del vectorial suman exactamente lo mismo:
`0.5/61`. Lo mismo pasa con los segundos, los terceros, y así. Cada vez que las
dos listas traen fragmentos distintos en la misma posición, hay un empate exacto.

El `EnsembleRetriever` desempata por orden de aparición, y BM25 es el primero de
la lista. Así que **BM25 gana todos los empates**. En las preguntas semánticas,
el primer resultado de BM25 es ruido: las preguntas están en español, los
documentos en inglés, y casi no comparten palabras. Ese ruido le ganaba por
desempate al fragmento correcto que el vectorial había puesto primero.

Con cualquier par de pesos distintos los empates desaparecen. Con 0.3 / 0.7, el
primero del vectorial suma `0.7/61 = 0.0115` y el primero de BM25 `0.3/61 =
0.0049`: manda el vectorial. BM25 igual aporta cuando coincide con él, porque un
fragmento que traen los dos suma los dos puntajes.

Para confirmarlo simulé la fórmula de LangChain con los rankings de cada
retriever, y reproduce las posiciones de las dos corridas en las 10
combinaciones de pregunta y pesos.

Antes de esa corrida había escrito otra explicación: que un fragmento presente
en las dos listas siempre le gana al que aparece en una sola, y que por eso hacía
falta un peso de 0.94 para el vectorial. La corrida con 0.3 / 0.7 la desmintió:
en esas preguntas casi no había fragmentos compartidos, y el problema era el
empate.

**El híbrido iguala al vectorial, no lo supera.** Con 0.3 / 0.7 los dos dan los
mismos números. En este Golden Set, BM25 no agrega nada medible. Hay que
aclarar que los pesos se eligieron mirando estas mismas 5 preguntas: es lo que
sugiere la clase ("ajustá según los resultados de tu propia evaluación"), pero
con un conjunto tan chico el valor exacto no generaliza.

**La ventaja esperada de BM25 con nombres exactos no apareció.** La clase plantea
que la búsqueda semántica es débil con términos exactos y que BM25 los cubre. En
las dos preguntas léxicas, el vectorial también encontró `trust_env` y
`MockTransport` en el primer lugar. Los embeddings de Google manejaron bien esos
identificadores.

**El Golden Set tiene un sesgo, y es chico.** Las preguntas están en español
sobre documentos en inglés, lo que deja a BM25 en desventaja de entrada en las
semánticas. Y con 5 preguntas los números orientan pero no alcanzan para una
conclusión general. Una prueba más justa para BM25 tendría preguntas en inglés y
muchas más.

**El cuello de botella de la ingesta fue la API de embeddings, no Pinecone.** La
primera versión mandaba los 83 fragmentos en una sola llamada y Google devolvió
429. Con lotes de 20 y pausas, el cuarto lote igual superó el límite por minuto:
el reintento esperó 30 segundos y siguió. Que esperar alcanzara confirma que era
el límite por minuto y no la cuota diaria. La ingesta completa tardó 105 s; el
upsert de los 83 vectores a Pinecone, 3,5 s.

**Las librerías de la pista no funcionan en Python 3.14 tal cual.** Tres cosas:

- `langchain-pinecone` declara `requires-python = "<3.14"`. pip retrocede hasta
  una versión de 2023 que pide un `numpy` viejo, y falla al compilarlo. Por eso el
  retriever vectorial es propio, sobre el SDK de Pinecone; la consigna lo permite.
- El SDK de Pinecone cambió de API en la v9 y otra vez en la v10:
  `create_index(spec=ServerlessSpec(...))`, `Index.upsert` e `Index.query`, que
  son los de la clase, en la v10 quedaron como compatibilidad y marcados como
  deprecated. Está fijado en la v8, donde son la API principal.
- `langchain-community`, de donde sale `BM25Retriever`, imprime un
  `DeprecationWarning` porque está siendo discontinuado. En la pista aparece el
  mismo aviso.

**Un detalle de los embeddings con dimensión reducida.** `gemini-embedding-001`
produce 3072 dimensiones y solo esas vienen normalizadas; las de 1536 no. El
índice usa métrica coseno, que divide por la longitud de cada vector, así que no
hace falta normalizar a mano. Con producto punto sí haría falta.

## Limitaciones conocidas

**BM25 vive en memoria.** Se reconstruye en cada corrida desde los archivos
locales. Para 83 fragmentos son milisegundos, pero no escala a millones de
documentos. Pinecone admite vectores dispersos (`sparse_vector` en la consulta)
para hacer la búsqueda híbrida del lado del servidor; queda como alternativa.

**RRF solo mira posiciones.** No sabe si el primer resultado de un retriever es
bueno o es lo menos malo que encontró. Los pesos ayudan, pero son fijos para
todas las preguntas. Lo que evalúa cada fragmento contra la pregunta es una
segunda etapa de re-ranking con un cross-encoder sobre el top-k fusionado, que es
lo que la clase propone como paso siguiente.

**La evaluación es a nivel documento.** Un fragmento cuenta como acierto si viene
del archivo esperado, aunque no sea el que contiene la respuesta. Evaluar a nivel
fragmento sería más exigente, pero requiere armar el Golden Set sabiendo qué
fragmento responde cada pregunta.

## Versiones usadas

`pinecone` 8.1.2, `langchain-core` 1.6.3, `langchain-classic` 1.0.8,
`langchain-community` 0.4.2, `langchain-google-genai` 4.4.0,
`langchain-text-splitters` 1.1.2, `rank_bm25` 0.2.2, `tiktoken` 0.14.0.

## Créditos

La documentación de `data/` es la de httpx, © 2019 Encode OSS Ltd, publicada bajo
licencia BSD-3-Clause. El texto completo de la licencia está en
[`data/LICENSE.md`](./data/LICENSE.md).
