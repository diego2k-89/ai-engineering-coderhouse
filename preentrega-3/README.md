# Pre-entrega 3 - Sistema de recuperación semántica local (RAG)

RAG sobre un conjunto de políticas internas de una empresa ficticia (Vantia
Software S.A.). Los documentos se fragmentan, se vectorizan con embeddings y se
guardan en ChromaDB. Una consulta recupera los fragmentos más parecidos y un LLM
responde usando solo eso como fuente.

El objetivo del ejercicio no es que el modelo responda bien, es que **no responda
cuando no tiene la información**.

## Requisitos

Python 3.12 o superior. Una API key de Google AI Studio para los embeddings, y
una de Anthropic o Google para la generación.

## Instalación

Desde la raíz del repo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r preentrega-3/requirements.txt
```

En Linux o Mac la activación es `source .venv/bin/activate`.

## Configuración

```bash
cd preentrega-3
copy .env.example .env
```

Y completar las keys. El `.env` está en el `.gitignore`.

| Variable | Descripción |
|---|---|
| `GOOGLE_API_KEY` | Key de Google AI Studio (aistudio.google.com/apikey) |
| `ANTHROPIC_API_KEY` | Key de Anthropic (console.anthropic.com) |
| `EMBEDDING_PROVIDER` | `google` o `huggingface`. Default `google` |
| `GOOGLE_EMBEDDING_MODEL` | Default `models/gemini-embedding-001` |
| `CHUNK_SIZE` | Tamaño de fragmento en tokens. Default 500 |
| `CHUNK_OVERLAP` | Solapamiento en tokens. Default 50 |
| `TOP_K` | Fragmentos a recuperar por consulta. Default 4 |
| `LLM_PROVIDER` | `anthropic` o `gemini`. Default `anthropic` |
| `ANTHROPIC_MODEL` | Default `claude-sonnet-4-6` |
| `GEMINI_MODEL` | Default `gemini-flash-latest` |
| `LLM_TEMPERATURE` | Default 0 |
| `LLM_MAX_TOKENS` | Default 1024 |
| `LLM_MAX_RETRIES` | Intentos antes de rendirse. Default 3 |

## Ingesta

Antes de consultar hay que construir el índice:

```bash
cd preentrega-3
python ingest.py
```

Corriéndolo una segunda vez el log dice `Índice existente (...): no se reindexa`.
Eso confirma que la persistencia funciona: el índice no se reconstruye en cada
corrida.

Para forzar la reconstrucción:

```bash
python ingest.py --reindex
```

## Ejecutar

```bash
python main.py                # las tres pruebas
python main.py --interactivo  # las tres pruebas y después consola de preguntas
```

Las tres pruebas son: una pregunta con respuesta en los documentos, una pregunta
trampa sobre un tema que las políticas no cubren, y la misma pregunta contra los
dos proveedores para comparar la redacción sobre un contexto idéntico.

## Archivos

```
data/             las cuatro políticas en .txt
schemas.py        modelos Pydantic: RespuestaLLM, FragmentoRecuperado, RAGResponse
ingest.py         carga, limpieza, fragmentación, embeddings y persistencia
rag.py            retriever, prompt, cadena LCEL y get_rag_response()
main.py           script de prueba
.env.example      plantilla de configuración
vectorstore/      índice de Chroma (generado, no versionado)
```

## Cómo funciona

Dos etapas separadas, y la separación es el punto.

**Ingesta** (`ingest.py`), una sola vez:

```
.txt -> limpiar -> fragmentar en tokens -> embeddings -> ChromaDB en disco
```

**Consulta** (`rag.py`), en cada pregunta:

```
pregunta -> retriever -> contexto -> (prompt | llm | parser) -> RAGResponse
```

### Fragmentación en tokens, no en caracteres

`RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)` corta en
500 tokens reales. Sin `from_tiktoken_encoder` el splitter cuenta caracteres, y
500 caracteres de español pueden ser 90 o 160 tokens según los acentos y la
puntuación. Como el límite de contexto del modelo se mide en tokens, contar
caracteres es estimar.

El splitter además prueba una jerarquía de cortes antes de partir por la fuerza:
párrafos, saltos de línea, espacios, y recién al final caracteres sueltos. Así un
corte casi nunca cae en medio de una oración.

Los 50 tokens de solapamiento existen para que una idea que queda justo en el
borde entre dos fragmentos aparezca completa en al menos uno de los dos.

### Un índice por modelo de embeddings

El índice no se guarda en una carpeta única, sino en una por cada par proveedor +
modelo:

```
vectorstore/
├── google__gemini-embedding-001/
│   ├── chroma/
│   └── manifest.json
└── hf__all-MiniLM-L6-v2/
    └── ...
```

Esto resuelve el error que la consigna marca como el más común: indexar con un
modelo de embeddings y consultar con otro. Ese error no lanza ninguna excepción.
Los vectores tienen dimensiones distintas o viven en espacios distintos, la
búsqueda devuelve los fragmentos "más cercanos" de un espacio que no corresponde,
y el resultado son fragmentos al azar que parecen válidos. Con una carpeta por
modelo es imposible mezclarlos, y cambiar de embeddings no destruye el índice
anterior.

El `manifest.json` de cada índice registra con qué modelo, con qué tamaño de
chunk y en cuánto tiempo se construyó.

### Las fuentes las arma el código, no el modelo

El LLM genera únicamente el texto de la respuesta y un booleano que dice si
encontró la información. Los nombres de archivo salen de los metadatos reales de
los fragmentos recuperados.

El motivo es que un modelo al que se le pide citar sus fuentes inventa
referencias plausibles: archivos que no existen, números de sección que suenan
bien. Y una cita falsa es peor que ninguna, porque da una sensación de
verificabilidad que no existe.

### El retriever queda afuera de la cadena LCEL

La cadena es `prompt | llm | parser`. La recuperación se hace antes, en
`get_rag_response()`, y no encadenada.

Es deliberado: recuperar y generar fallan distinto. Si la respuesta sale mal hay
que poder mirar los fragmentos recuperados por separado para saber si el problema
fue la búsqueda o la redacción. Por eso `RAGResponse` devuelve también los
fragmentos con un extracto de cada uno. Sin eso, un RAG es una caja negra.

### top_k entre 3 y 5

Más contexto no es mejor contexto. Pasado cierto punto los modelos pierden
precisión sobre lo que está en el medio del prompt, y cada fragmento extra son
tokens que se pagan en cada consulta.

### Logs

Cada consulta registra la cantidad de fragmentos recuperados y la duración,
desglosada en recuperación y generación:

```
INFO  rag.ingesta  | Índice existente (google / models/gemini-embedding-001, 12 fragmentos): no se reindexa
INFO  rag.consulta | [anthropic] Consulta: '¿Cuántos días de vacaciones le corresponden a un empleado con 7 años de antigüedad?'
INFO  rag.consulta | [anthropic] Recuperados 4 fragmentos en 0.44s
INFO  rag.consulta | [anthropic] Listo en 4.00s (recuperación 0.44s + generación 2.83s), encontro_respuesta=True
```

El desglose importa porque las dos etapas no cuestan lo mismo ni por los mismos
motivos. Conviene aclarar algo que no es obvio: la recuperación tampoco es
puramente local. La comparación de vectores sí lo es, pero antes hay que convertir
la pregunta en un vector, y eso es una llamada a la API de embeddings. Por eso la
recuperación varía entre 0,36 s y 1,22 s según la red, y no es un costo fijo.

La medición usa `time.perf_counter()`, que es monótono, en vez de `time.time()`,
que puede saltar si el reloj del sistema se ajusta en medio de la llamada.

En la primera consulta de cada proceso los dos parciales no suman el total: falta
un hueco de unos 0,73 s. No es ruido, es reproducible, y no aparece nunca más en
esa misma corrida. El motivo es el import diferido de `get_llm()`: la librería del
proveedor se importa recién cuando se la necesita, y ese primer import se paga una
vez. Python después la tiene en `sys.modules` y las consultas siguientes cierran
con el hueco en cero.

Es el precio del import adentro de la función, y se paga a cambio de que no tener
instalado un proveedor no impida usar el otro.

## Salida de ejemplo

Los cuatro documentos generan 12 fragmentos. La ingesta completa tarda unos 9,5 s
con los embeddings de Google.

**Pregunta con respuesta en los documentos.** El caso está elegido para que el
modelo tenga que interpretar un rango y no copiar un número: la política define
tramos de antigüedad, y 7 años cae dentro de "desde 5 años y hasta 10 inclusive".

```
Pregunta: ¿Cuántos días de vacaciones le corresponden a un empleado con 7 años de antigüedad?

Respuesta (anthropic):
A un empleado con 7 años de antigüedad le corresponden 21 días corridos de
vacaciones anuales. Esto se debe a que se encuentra en el tramo de antigüedad
que va desde 5 años hasta 10 años inclusive.

¿Encontró la información en el contexto? SÍ
Fuentes: politica_gastos_y_viajes.txt, politica_vacaciones.txt
Fragmentos usados: 4
```

**Pregunta trampa.** Ninguna de las cuatro políticas cubre bonos ni remuneración
variable:

```
Pregunta: ¿Cuál es la política de bonos por rendimiento anual y cómo se calcula el porcentaje sobre el salario?

Respuesta (anthropic):
La información sobre bonos por rendimiento anual y su cálculo no figura en las
políticas disponibles. Los documentos con los que cuento corresponden a la
política de vacaciones y la política de teletrabajo de Vantia Software S.A.,
ninguna de las cuales aborda ese tema.

¿Encontró la información en el contexto? NO
Fuentes: politica_teletrabajo.txt, politica_vacaciones.txt
Fragmentos usados: 4
```

**Los dos proveedores sobre el mismo contexto.** Misma pregunta, mismos cuatro
fragmentos, distinto generador. Los dos coinciden en los datos: 3 días remotos y
20 Mbps de bajada con 5 de subida.

## Qué observé al probarlo

**El retriever devuelve fragmentos que no tienen nada que ver, y eso es esperable.**
En la pregunta de vacaciones, uno de los cuatro fragmentos recuperados fue la
sección "Capacitación y conferencias" de la política de gastos. Una búsqueda por
similitud siempre devuelve los `k` más cercanos: si hay menos de `k` fragmentos
pertinentes, completa con lo que sigue en el ranking. El modelo lo ignoró y
respondió solo con lo que servía, que es el comportamiento correcto, pero el
fragmento igual se pagó en tokens.

**`fuentes` dice qué se le mostró al modelo, no qué usó.** Consecuencia directa de
lo anterior: la respuesta sobre vacaciones salió enteramente de
`politica_vacaciones.txt`, pero `fuentes` también lista
`politica_gastos_y_viajes.txt`, porque ese archivo aportó uno de los fragmentos
recuperados. En la pregunta trampa pasa lo mismo y se nota más: la respuesta es
que no hay información, y aun así hay dos fuentes listadas.

Se podría estrechar pidiéndole al modelo que indique cuáles de los fragmentos usó,
pero eso reintroduce justamente el problema que la separación evita: que el modelo
se invente la atribución. La alternativa honesta es entender el campo por lo que
es, un registro de qué se consultó.

**Hay dos capas de reintento y solo escribí una.** En la comparación entre
proveedores, Gemini devolvió un 503 y su propio SDK reintentó antes de que
`.with_retry()` se enterara:

```
INFO  google_genai._api_client | Retrying ... in 1.1 seconds as it raised ServerError: 503 UNAVAILABLE
INFO  rag.consulta | [anthropic] Listo en 4.08s (recuperación 0.36s + generación 3.72s)
INFO  rag.consulta | [gemini]    Listo en 17.47s (recuperación 1.12s + generación 16.33s)
```

Los 17 s de Gemini contra los 4 de Anthropic no miden qué modelo es más rápido:
miden un 503 y la espera del reintento interno del SDK. Sin el log de
`google_genai._api_client` visible, ese número quedaría inexplicado. Por eso ese
logger no está silenciado.

En una segunda corrida, sin 503, la misma comparación dio Gemini 4,09 s y Anthropic
5,87 s: el orden se invirtió. Sirve de advertencia sobre qué se puede concluir de
un elapsed. Con una sola medición por proveedor no se compara nada; lo que el
número informa es cuánto tardó esa llamada, no cuánto tarda ese modelo.

**La recuperación varía más de lo que esperaba, y no por concurrencia.** Los
tiempos de recuperación fueron 0,36 / 0,38 / 0,42 / 0,44 / 0,49 y 1,22 s. Al
principio atribuí el pico a que las dos consultas de `asyncio.gather` comparten un
retriever y el acceso a Chroma se serializaría. La segunda corrida lo descartó: ahí
las dos consultas concurrentes dieron 0,42 y 0,49 s, prácticamente iguales, y el
1,22 s apareció en una consulta que corrió sola.

La explicación que sí se sostiene es la de más arriba: cada recuperación incluye una
llamada de red para vectorizar la pregunta, así que la variación es latencia de API,
no contención local.

**El modelo enumera solo los documentos que vio.** En la respuesta a la pregunta
trampa dice que cuenta con "la política de vacaciones y la política de
teletrabajo". Es cierto respecto de su contexto, pero el sistema tiene cuatro
documentos indexados. Alguien leyendo esa respuesta podría concluir que faltan
archivos. Es una consecuencia de que el modelo solo conoce su ventana, no el
índice completo.

## Limitaciones conocidas

**El retriever siempre devuelve `k` fragmentos.** Una búsqueda por similitud no
tiene el concepto de "ningún resultado relevante": devuelve los `k` más cercanos
aunque estén lejos. Por eso el anclaje al contexto tiene que estar en el prompt,
y por eso `RespuestaLLM` incluye `encontro_respuesta` en vez de confiar en que la
lista de fragmentos venga vacía.

**No hay umbral de similitud.** Chroma permite filtrar por score, pero elegir el
corte requiere medir sobre datos reales y el valor no es transferible entre
modelos de embeddings. Queda como mejora para el módulo siguiente, donde entra
búsqueda híbrida y re-ranking.

**Los documentos son cortos.** Cuatro archivos que dan 12 fragmentos en total: el
índice entero entra en el contexto de cualquier modelo actual, así que el RAG acá
no resuelve un problema de escala, muestra el mecanismo.

De paso, esos 12 fragmentos dejan ver por qué el chunking va en tokens. Los cuatro
archivos rondan los 3.000 caracteres, que por la regla de "4 caracteres por token"
darían unos 750 tokens y 2 fragmentos por archivo. Salieron 3. El español con
acentos y con palabras largas tokeniza más pesado que esa estimación, y ahí está la
diferencia entre contar caracteres y contar tokens de verdad.
