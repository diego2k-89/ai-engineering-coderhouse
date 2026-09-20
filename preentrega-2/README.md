# Pre-entrega 2 - Pipeline de procesamiento validado

Pipeline de extracción de entidades técnicas. Recibe un texto libre (un log de
error, una descripción de arquitectura) y devuelve un objeto validado con las
tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

Está armado con LangChain: una cadena LCEL con salida estructurada de Pydantic
y reintentos automáticos.

## Requisitos

Python 3.12 o superior.

## Instalación

Desde la raíz del repo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r preentrega-2/requirements.txt
```

En Linux o Mac la activación es `source .venv/bin/activate`.

## Configuración

```bash
cd preentrega-2
copy .env.example .env
```

Y completar las keys. El `.env` está en el `.gitignore`.

| Variable | Descripción |
|---|---|
| `ANTHROPIC_API_KEY` | Key de Anthropic (console.anthropic.com) |
| `GOOGLE_API_KEY` | Key de Google AI Studio (aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | Key de OpenAI |
| `LLM_PROVIDER` | Proveedor por defecto: `openai`, `anthropic` o `gemini` |
| `ANTHROPIC_MODEL` | Default `claude-sonnet-4-6` |
| `GEMINI_MODEL` | Default `gemini-flash-latest` |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `LLM_TEMPERATURE` | Default 0 |
| `LLM_MAX_TOKENS` | Default 1024 |
| `LLM_MAX_RETRIES` | Intentos antes de rendirse. Default 3 |

Hace falta al menos una key válida del proveedor que se vaya a usar.

## Ejecutar

```bash
cd preentrega-2
python main.py
```

El script corre tres pruebas: el mismo texto contra los tres proveedores, un
texto ambiguo como prueba de estrés, y un truncamiento forzado con
`max_tokens=25` para ver la detección de respuesta incompleta.

## Salida de ejemplo

Para esta entrada:

> Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en
> Redis parece saturarse en picos de tráfico y las conexiones a PostgreSQL se
> agotan porque el pool está mal dimensionado. Esto está afectando a usuarios
> en producción.

```json
{
  "tecnologias": [
    "FastAPI",
    "Redis",
    "PostgreSQL"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API construida con FastAPI presenta timeouts intermitentes en producción causados por saturación del caché en Redis durante picos de tráfico y agotamiento del pool de conexiones a PostgreSQL por un dimensionamiento inadecuado."
}
```

## Archivos

```
schemas.py      modelo Pydantic: EntidadesTecnicas y NivelCriticidad
chain.py        prompt, fábrica de modelos, cadena LCEL y process_text()
main.py         script de prueba
.env.example    plantilla de configuración
```

## Cómo funciona

La cadena tiene tres pasos:

```python
PROMPT | modelo_estructurado | RunnableLambda(verificar)
```

`ChatPromptTemplate` arma el mensaje con la variable `{texto}`.
`with_structured_output(EntidadesTecnicas)` convierte el esquema de Pydantic en
un JSON Schema y se lo pasa al proveedor, así el modelo está obligado a
devolver esa forma. `verificar()` revisa la respuesta antes de darla por buena.

Todo eso va envuelto en `.with_retry()`, que reintenta hasta tres veces con
espera creciente.

### Por qué hay un tercer paso

`with_structured_output()` devuelve el objeto ya parseado y esconde el mensaje
original, donde viene el `finish_reason`. Con `include_raw=True` se recibe un
diccionario con `raw`, `parsed` y `parsing_error`, y ahí sí se puede revisar si
el modelo cortó la respuesta por falta de tokens.

Esto importa porque un JSON truncado a veces sigue siendo válido. Si el modelo
iba a listar cinco tecnologías y alcanzó a escribir tres, el esquema lo acepta
y el objeto parece correcto aunque esté incompleto.

Hay un detalle: con `include_raw=True` un error de validación ya no lanza
excepción, se devuelve callado en `parsing_error`. Y `.with_retry()` solo
reintenta cuando algo lanza. Por eso `verificar()` relanza el error, y por eso
el `.with_retry()` va sobre la cadena entera y no solo sobre el modelo: si
fuera solo sobre el modelo, el paso de verificación quedaría fuera del
reintento y detectar el truncamiento no serviría de nada.

### Cada proveedor nombra distinto el corte

| Proveedor | Clave | Valor cuando se corta |
|---|---|---|
| OpenAI | `finish_reason` | `length` |
| Anthropic | `stop_reason` | `max_tokens` |
| Gemini | `finish_reason` | `MAX_TOKENS` |

`_fue_truncada()` revisa las dos claves y normaliza los valores.

## Qué observé al probarlo

**El texto ambiguo no rompe el pipeline, pero devuelve basura.** Pasándole "El
sistema anda medio raro últimamente, no sé bien qué está pasando", el esquema
exige al menos una tecnología y el prompt pide no inventar. El modelo resolvió
la contradicción devolviendo un placeholder:

```json
{
  "tecnologias": ["<UNKNOWN>"],
  "nivel_de_criticidad": "baja",
  "resumen_tecnico": "El texto no contiene información técnica concreta..."
}
```

No mintió sobre el contenido y lo aclaró en el resumen, pero `<UNKNOWN>` pasa
la validación porque es un string no vacío. Para un sistema que consuma esta
salida, ese valor es peor que un error: no se nota. Se podría rechazar en el
`field_validator` con una lista de placeholders conocidos, o directamente
permitir la lista vacía. Queda anotado como algo a decidir con el caso de uso
real.

**Hay dos capas de reintento y solo escribí una.** El SDK de Google reintenta
por su cuenta ante un 503, con su propio backoff, antes de que
`.with_retry()` se entere de que hubo un problema:

```
Retrying ... in 1.82s ... 2.74s ... 4.89s ... 8.03s ... 16.3s
```

En el peor caso eso se multiplica: 3 reintentos míos por 5 del SDK son 15
llamadas. Conviene saber qué reintenta cada librería antes de sumar otra capa
encima.

**LangChain detecta el truncamiento pero no lanza.** Imprime un error
(`Output parser received a max_tokens stop reason`) y sigue de largo. El paso
`verificar()` es el que lo convierte en excepción para que el reintento se
dispare.

**`.with_retry()` no distingue qué vale la pena reintentar.** En la prueba de
truncamiento, el presupuesto de tokens está fijo, así que los tres intentos
fallan igual. Reintentar ahí es perder tiempo. La versión a mano del módulo 1
sí distinguía entre errores pasajeros y permanentes; acá se gana simplicidad y
se pierde ese control.

## Versiones usadas

`langchain-core` 1.6.3, `langchain-anthropic` 1.7.2, `langchain-openai` 1.6.2,
`langchain-google-genai` 4.4.0, `pydantic` 2.13.5.

No se instala el paquete `langchain` a secas porque arrastra LangGraph, que no
se usa acá.

Sobre el modelo de Anthropic: se usa `claude-sonnet-4-6` y no `claude-sonnet-5`
porque los modelos más nuevos rechazan con un 400 cualquier valor de sampling
que no sea el default, y este pipeline necesita `temperature=0`.
