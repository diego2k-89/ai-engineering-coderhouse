# Pre-entrega 7: API de producción y monitoreo

El orquestador multi-agente de la pre-entrega 6 (supervisor, investigador y
analista sobre la documentación de httpx) expuesto como una API REST con
FastAPI. La API no corre el grafo dentro del request: encola la tarea en Redis,
devuelve un `job_id` al instante y un worker la procesa en segundo plano. El
estado de cada job y los checkpoints de LangGraph viven en Redis, cada ejecución
se traza en LangSmith, y las tareas que superan un presupuesto de costo se
pausan hasta que una persona las aprueba.

```
cliente --POST /tasks--> FastAPI --SET job + RPUSH--> Redis <--BLPOP-- worker --> grafo
   ^                        |                           ^                          |
   +---GET /tasks/{id}------+                           +---- checkpoints ---------+
                                                                                    |
                                                         LangSmith <---- trazas ----+
```

## Estructura

```
app/
├── main.py            FastAPI: POST /tasks, GET /tasks/{id}, GET /health
├── worker.py          saca jobs de la cola y corre el grafo (PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED)
├── jobs.py            registro de cada job en Redis y la cola
├── graph.py           el orquestador de la PE6 + nodo de aprobación + checkpointer de Redis
├── hitl.py            nodo de aprobación humana (interrupt) y POST /tasks/{id}/approve
├── observability.py   verificación de LangSmith y metadatos de cada traza
├── state.py / config.py
└── agents/            supervisor, investigador y analista (de la PE6)
scripts/
├── carga.py           las 5 peticiones concurrentes
└── demo_hitl.py       pausa y aprobación, en dos pasos
tests/                 pytest con Redis y grafo falsos: corren sin API keys
trazas/                salidas de las corridas (carga.json, hitl.json, logs)
screenshots/           capturas de LangSmith
docker-compose.yml     Redis 8
```

## Cómo levantarlo

Requiere Python 3.12 o superior, Docker y el índice de Pinecone de la
pre-entrega 4 ya cargado.

**1. Redis.** Hace falta Redis 8 (o Redis Stack): el checkpointer de LangGraph
usa los módulos RedisJSON y RediSearch, que un Redis común no trae.

```bash
cd preentrega-7
docker compose up -d
```

**2. Dependencias y configuración.** Desde la raíz del repo:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r preentrega-7/requirements.txt
cd preentrega-7
copy .env.example .env
```

En el `.env` van tres keys de modelos (Anthropic, Pinecone, Google) y la de
LangSmith. Sin la de LangSmith la API funciona igual, pero no manda trazas, y
lo avisa en el log al arrancar.

**3. La API**, desde `preentrega-7/`:

```bash
uvicorn app.main:app
```

Al arrancar, el log tiene que mostrar que LangSmith está activo, que Redis
conectó y que el worker escucha la cola.

**4. Probar una tarea:**

```bash
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" -d "{\"pregunta\": \"¿Qué excepción lanza httpx si no consigue conexión del pool?\"}"
curl localhost:8000/tasks/<job_id>
```

También está la documentación interactiva en `http://localhost:8000/docs`.

## Las 5 peticiones concurrentes

Con la API corriendo, en otra terminal:

```bash
python scripts/carga.py
```

Manda 5 preguntas al mismo tiempo, hace polling hasta que terminan y guarda en
`trazas/carga.json` cuánto tardó cada POST en responder y cada job de punta a
punta. Después, en LangSmith (proyecto `pe7-api-agentes`), se toman el costo por
ejecución y la latencia de esas 5 trazas.

## La aprobación humana

```bash
python scripts/demo_hitl.py                     # crea la tarea y espera la pausa
python scripts/demo_hitl.py --aprobar <job_id>  # o --rechazar
```

Entre los dos comandos se puede cortar la API (Ctrl+C) y volver a levantarla: el
grafo pausado está en Redis y se reanuda igual.

## Tests

```bash
pytest
```

Prueban la API y el worker con un Redis en memoria (`fakeredis`) y un grafo
falso, así que no necesitan Docker ni API keys: que el POST responda sin esperar
al grafo, que una excepción deje el job en FAILED, el 404, la pausa y la
aprobación, el 409 al aprobar algo que no está pausado, el ruteo del
presupuesto y la calculadora del analista.

---

## Cómo cubre cada criterio de la rúbrica

### Orquestación Asíncrona y API (30%)

- `POST /tasks` guarda el job en Redis, lo encola y responde **202** con el
  `job_id`. No corre el grafo: son dos operaciones async contra Redis.
- `GET /tasks/{id}` devuelve el estado; 404 si no existe.
- El worker (`worker.py`) es una tarea de asyncio que arranca con la API. Saca
  jobs con `BLPOP` y corre hasta 5 en paralelo (`MAX_CONCURRENCIA`).
- Nada bloquea el event loop: Redis, el modelo y el grafo son async, y la única
  librería sincrónica (el SDK de Pinecone) corre en `asyncio.to_thread`.
- Evidencia: `trazas/carga.json` muestra el tiempo de respuesta de cada POST, y
  el test `test_crear_tarea_responde_sin_esperar_al_grafo` lo verifica con un
  grafo que tarda 2 s.

### Persistencia con Redis (25%)

- **Estado de los jobs:** clave `job:<id>` con un JSON (`jobs.py`). Se escribe
  en PENDING **antes** de encolar, para que una consulta inmediata no dé 404.
- **Paso a FAILED:** `worker.procesar()` captura cualquier excepción del grafo y
  deja el job en FAILED con el error. Lo verifica
  `test_excepcion_en_el_grafo_marca_failed`.
- **Checkpoints de LangGraph:** `AsyncRedisSaver` de
  `langgraph-checkpoint-redis`, con el `job_id` como `thread_id`. Es lo que
  permite pausar en el nodo de aprobación y reanudar después de reiniciar la API.

### Observabilidad (25%)

- LangSmith por variables de entorno: cada ejecución del grafo es una traza con
  un span por nodo, por llamada al modelo y por herramienta.
- La búsqueda en Pinecone no es de LangChain: lleva `@traceable` en
  `agents/research_agent.py` para aparecer como su propio span.
- Cada traza lleva el `job_id` y la versión de los prompts como metadatos
  (`observability.py`).
- Capturas en `screenshots/`.

### Human-in-the-loop (15%)

- **Qué es crítico: el costo.** Una pregunta típica deriva dos veces (investigador
  y analista). Si el supervisor quiere superar `PRESUPUESTO_DERIVACIONES`, el
  grafo pasa por el nodo `aprobacion`, que llama a `interrupt()`.
- El job queda en **WAITING_APPROVAL** con el motivo. `POST /tasks/{id}/approve`
  con `{"aprobado": true}` lo reanuda con `Command(resume=...)`; con `false`, la
  síntesis cierra con lo que ya había.

### Documentación y estructura (5%)

Este README, `requirements.txt`, `.env.example`, `docker-compose.yml` y la
estructura de la consigna, con dos archivos más (`jobs.py` y `agents/`).

---

## Desvíos respecto a la consigna

- **Python 3.14**, no 3.12. Todas las dependencias tienen versiones para 3.14.
- **El modelo es de Anthropic y los embeddings de Google**, como en las
  pre-entregas 4 y 6. Ninguno usa OpenAI.
- **El worker corre en el mismo proceso que la API**, como tarea de asyncio. La
  consigna lo acepta para esta etapa. La cola en Redis permitiría sacarlo a un
  proceso aparte sin cambiar la API.
- **La latencia p95 no la muestra LangSmith.** Sus estadísticas traen P50 y P99
  ([`03_stats_latencia.png`](./screenshots/03_stats_latencia.png)). El p95 se
  calculó con las latencias que LangSmith muestra para cada una de las 5 trazas
  ([`01_trazas_carga.png`](./screenshots/01_trazas_carga.png)), y
  `scripts/carga.py` lo calcula también del lado del cliente.
- **`hitl.py` tiene el endpoint de aprobación y el nodo**, como en la estructura
  de referencia. El registro de jobs está aparte, en `jobs.py`, porque lo usan
  la API, el worker y el endpoint de aprobación.

## Resultados

### Las 5 peticiones concurrentes

Salida en [`trazas/carga.json`](./trazas/carga.json). Latencia y costo según
LangSmith ([`screenshots/01_trazas_carga.png`](./screenshots/01_trazas_carga.png)):

| Pregunta | Agentes | Latencia | Tokens | Costo |
|---|---|---|---|---|
| 250 conexiones vs. límite por defecto | investigador + analista | 40,74 s | 16,5 K | $0,069 |
| Timeout por defecto × 3 intentos | investigador + analista | 34,18 s | 13,5 K | $0,055 |
| Keep-alive por defecto vs. 50 | investigador + analista | 30,88 s | 15,3 K | $0,061 |
| Variable de entorno para certificados SSL | investigador | 25,53 s | 9,0 K | $0,038 |
| Excepción si no hay conexión del pool | investigador | 22,27 s | 9,9 K | $0,040 |

- **Costo por ejecución:** entre $0,038 y $0,069; **$0,053 en promedio** y $0,26
  las cinco. LangSmith lo calcula solo, con los tokens de entrada y salida de
  cada llamada a `claude-sonnet-4-6`.
- **Latencia p95: 40,74 s** (p50: 30,88 s), calculada sobre las latencias que
  muestra LangSmith para esas cinco trazas. Con 5 valores, el p95 por rango más
  cercano es el más lento (interpolando da 39,4 s). Ver "Desvíos".
- **Las cinco corrieron en paralelo.** Pasaron a RUNNING en 13 ms y la corrida
  entera tardó 41,8 s, lo mismo que la más lenta. En fila habrían sido unos
  150 s.
- **Los POST respondieron en ~300 ms**, contra 22 a 41 s de cada grafo.

Las cinco respuestas son correctas contra la documentación, y el supervisor sumó
al analista solo en las tres preguntas que pedían una cuenta.

### Lectura del dashboard

Desglose de la traza más lenta, la de 250 conexiones
([`02`](./screenshots/02_traza_detalle.png),
[`04`](./screenshots/04_traza_investigador.png) y
[`05`](./screenshots/05_traza_analista_sintesis.png)):

| Nodo | Tiempo | % | Tokens | % |
|---|---|---|---|---|
| supervisor (3 decisiones) | 12,5 s | 31% | 5,6 K | 34% |
| investigador | 12,9 s | 32% | 5,8 K | 35% |
| analista | 11,6 s | 28% | 3,8 K | 23% |
| síntesis | 3,7 s | 9% | 1,3 K | 8% |

**Dónde se concentra la latencia: en las llamadas al modelo.** Suman 39,2 de los
40,7 s, el 96%. Las herramientas casi no pesan: cada búsqueda en Pinecone
(`pinecone_busqueda`) tardó 1,1 a 1,5 s, y las del analista menos de 10 ms.

**Qué nodo consume más tokens: el investigador**, seguido muy de cerca por el
supervisor. La llamada más cara de toda la traza es la segunda del investigador,
la que redacta su respuesta: 8,8 s y 4,6 K tokens, porque recibe los fragmentos
de la documentación y devuelve un texto largo con tablas. El analista hace lo
mismo en menor escala.

Dos cosas que se ven en la traza y no se veían en los logs:

- **El investigador hizo dos búsquedas en paralelo.** Pidió las dos en la misma
  respuesta del modelo y `create_react_agent` las corrió a la vez (1,5 s y 1,1 s,
  solapadas). Por eso el nodo tarda 12,9 s y no la suma de sus partes.
- **El supervisor cuesta tanto como un especialista.** Decide tres veces por
  pregunta y cada decisión relee la pregunta y todas las contribuciones, así que
  cada vez manda más tokens (1,5 K, 1,9 K, 2,2 K).

Cada traza lleva como metadatos el `job_id`, la versión de los prompts
(`pe6-v1`) y el commit del repo, así que se puede ir de un job de la API a su
traza y saber con qué código corrió.

### Qué observé

**El worker se moría a los 5 segundos de quedarse sin trabajo.** En la primera
prueba los jobs quedaron en PENDING para siempre, y `/health` mostraba el worker
caído sin ningún error en el log. En la segunda, los jobs llegaron a RUNNING y
se quedaron ahí: la traza de LangSmith mostraba un `CancelledError` en la llamada
del investigador al modelo, a los 4,8 s. Son las dos trazas con error de
[`03_stats_latencia.png`](./screenshots/03_stats_latencia.png).

La causa: en redis-py 8 el cliente tiene un `socket_timeout` de 5 s por defecto
(lo leí en el código del paquete), y el `BLPOP` del worker esperaba también 5 s.
Cuando la cola quedaba vacía, los dos plazos vencían a la vez y la tarea del
worker terminaba cancelada. El mecanismo exacto de esa carrera dentro de
redis-py no lo verifiqué; lo que sí verifiqué es que separando los dos plazos
dejó de pasar. Al terminar, su `finally` cancelaba los jobs en curso, y como
`CancelledError` no hereda de `Exception`, el job no pasaba a FAILED.

Lo arreglé con tres cambios: `BLPOP` espera 2 s, menos que el timeout del
cliente; una cancelación deja el job en FAILED con el motivo; y un error en el
bucle del worker se registra y el bucle sigue. Después de eso, el worker pasó la
prueba de carga con la cola vacía entre jobs sin caerse. El caso que pedía la
consigna, "el cliente se queda esperando en un loop infinito de polling", fue
exactamente lo que pasó.

Además cada job tiene un techo de 180 s (`TIMEOUT_JOB_SEGUNDOS`): si el grafo se
traba esperando una API, el job termina en FAILED con `TimeoutError`.

### Human-in-the-loop

Salida en [`trazas/hitl.json`](./trazas/hitl.json). La pregunta de
`scripts/demo_hitl.py` pide dos datos de archivos distintos y dos cuentas.

| Paso | Qué pasó | Tiempo |
|---|---|---|
| `demo_hitl.py` | El supervisor derivó dos veces al investigador (un dato cada vez) y quiso derivar al analista: tercera derivación, presupuesto 2. El job quedó en **WAITING_APPROVAL** con el motivo y la instrucción que iba a recibir el analista | 40,1 s |
| Ctrl+C y `uvicorn` de nuevo | La API se apagó con el job pausado y volvió a levantar con un proceso nuevo | |
| `demo_hitl.py --aprobar` | El proceso nuevo leyó el grafo pausado desde Redis y siguió desde el analista, **sin repetir las búsquedas**. Terminó en DONE: timeout 5 s → 15 s en 3 intentos; 20 keep-alive → faltan 30 para 50 | 15,5 s |

El reinicio entre la pausa y la aprobación es la prueba de que el estado no vive
en memoria: el grafo pausado quedó en Redis, bajo el `job_id` como `thread_id`.
En LangSmith son dos trazas con el mismo `job_id`, `tarea-iniciar` y
`tarea-reanudar` ([`06_hitl_trazas.png`](./screenshots/06_hitl_trazas.png)). La
entrada de la segunda es el `Command` con `{"resume": {"aprobado": true}}`. Entre
las dos costaron $0,093 (13,7 K + 8,7 K tokens): es la tarea más cara de todas
las corridas, que es justo lo que el presupuesto busca frenar.

La instrucción para el analista ya traía los datos y las cuentas a hacer, así
que quien aprueba sabe exactamente en qué se va a gastar la llamada extra.
