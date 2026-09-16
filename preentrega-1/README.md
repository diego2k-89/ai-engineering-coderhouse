# Pre-entrega 1 - Cliente de LLM robusto y asíncrono

Cliente asíncrono para hablar con OpenAI, Anthropic o Gemini usando una misma
interfaz. Soporta streaming, valida los datos con Pydantic, reintenta cuando
el error es pasajero y cambia de proveedor si uno no responde.

## Requisitos

Python 3.12 o superior.

## Instalación

Desde la raíz del repo:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r preentrega-1/requirements.txt
```

En Linux o Mac la activación es `source .venv/bin/activate`.

## Configuración

```bash
cd preentrega-1
copy .env.example .env
```

Y completar las keys. El `.env` está en el `.gitignore`, no se sube al repo.

Variables:

| Variable | Descripción |
|---|---|
| `ANTHROPIC_API_KEY` | Key de Anthropic (console.anthropic.com) |
| `GOOGLE_API_KEY` | Key de Google AI Studio (aistudio.google.com/apikey) |
| `OPENAI_API_KEY` | Key de OpenAI |
| `LLM_PROVIDER` | Proveedor principal: `openai`, `anthropic` o `gemini` |
| `LLM_FALLBACK_CHAIN` | Proveedores de respaldo, separados por coma |
| `ANTHROPIC_MODEL` | Default `claude-sonnet-5` |
| `GEMINI_MODEL` | Default `gemini-flash-latest` |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `LLM_TEMPERATURE` | De 0 a 2. Default 0.7 |
| `LLM_MAX_TOKENS` | Default 512 |
| `LLM_MAX_RETRIES` | Intentos por proveedor. Default 3 |
| `LLM_TIMEOUT_SECONDS` | Default 30 |

Hace falta al menos una key válida de algún proveedor de la cadena.

## Ejecutar

```bash
cd preentrega-1
python main.py
```

El script hace tres pruebas: una consulta en modo normal, la misma en
streaming midiendo el tiempo hasta el primer token, y un intento de armar una
config inválida para mostrar la validación de Pydantic.

## Archivos

```
main.py                 script de prueba
.env.example            plantilla de configuración
llm_client/
  schemas.py            modelos de Pydantic
  base.py               clase abstracta BaseLLMClient
  providers.py          los tres clientes
  manager.py            AsyncLLMManager: elige el cliente, reintenta y hace fallback
```

`BaseLLMClient` define dos métodos: `generate()` devuelve la respuesta entera
y `generate_stream()` la va emitiendo por partes. Cada proveedor la implementa
a su manera.

`AsyncLLMManager` es el punto de entrada. Prueba con el primer proveedor de la
cadena y si falla pasa al siguiente. Los errores no se propagan como
excepciones, viajan adentro del `ModelResponse` en el campo `error`.

## Notas

- OpenAI está configurado sin API key a propósito, para que el fallback se
  pueda ver funcionando en cada corrida.
- No todos los errores se reintentan. Un rate limit o un 503 sí, porque pueden
  mejorar esperando. Una key inválida no, así que pasa al siguiente proveedor
  de una.
- El cliente de Anthropic no manda `temperature`: el SDK 1.x la sacó de
  `messages.create()` y los modelos nuevos rechazan valores de sampling
  distintos del default. Igual sigue validada y sigue aplicando a los otros
  dos proveedores.
- Las keys se guardan en `SecretStr`, así no aparecen si se imprime la config.

## Versiones usadas

Probado con `openai` 3.13.0, `anthropic` 1.5.0, `google-genai` 2.23.0 y
`pydantic` 2.13.5. Son versiones bastante más nuevas que las del ejemplo de
clase, y dos cosas cambiaron: Anthropic sacó `temperature` de la firma, y la
respuesta puede venir con un bloque de razonamiento antes del texto, así que
hay que recorrer los bloques en vez de agarrar el primero.
