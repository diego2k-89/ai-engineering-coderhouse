# AI Engineering — Coderhouse

Repositorio del proyecto integrador del curso **AI Engineering** (Comisión 2026).
Cada pre-entrega vive en su propia carpeta y construye sobre la anterior.

---

## Recorrido

| Módulo | Tema | Entrega | Estado |
|:---:|---|---|:---:|
| 1 | La interfaz base: conexión y abstracción de LLMs | [`preentrega-1/`](./preentrega-1) — Cliente LLM robusto y asíncrono | ✅ Entregado |
| 2 | Encadenamiento lógico: orquestación con LangChain | [`preentrega-2/`](./preentrega-2) — Pipeline de procesamiento validado | ✅ Entregado |
| 3 | Persistencia de datos y vector DBs | `preentrega-3/` — Sistema de recuperación semántica local (RAG) | 🚧 En curso |


---

## Cómo levantar el proyecto

Requiere Python 3.12 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate                              # Windows
pip install -r preentrega-1/requirements.txt       # o la entrega que vayas a correr
copy preentrega-1\.env.example preentrega-1\.env    # y completar las keys
```

Las instrucciones detalladas de cada entrega están en el README de su carpeta.

---

## Seguridad

Las API keys **nunca** se escriben en el código ni se suben al repositorio.
Se leen de un archivo `.env` local que está excluido por [`.gitignore`](./.gitignore).
El archivo `.env.example` documenta qué variables hacen falta, pero no contiene valores reales.
