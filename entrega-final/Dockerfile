FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Primero las dependencias: si solo cambia el código, Docker reusa esta capa.
COPY requirements.txt .
RUN pip install -r requirements.txt

# El fragmentador mide en tokens con tiktoken, que baja su tabla la primera vez
# que se usa. Se baja acá para que el contenedor arranque sin depender de eso.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY app ./app
COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini .

EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
