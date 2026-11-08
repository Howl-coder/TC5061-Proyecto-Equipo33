# Imagen OCI del servicio de predicción P1 (FastAPI + modelo exportado del Model Registry).
# Antes de construirla: python -m src.model_package --pregunta p1   (genera models/servicio/)
#   podman build -t mibici-api:latest .
#   podman run --rm -p 8000:8000 mibici-api:latest
ARG PYTHON_IMAGE=docker.io/library/python:3.12.13-slim

FROM ${PYTHON_IMAGE} AS dependencias
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-api.txt /tmp/requirements-api.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements-api.txt

FROM ${PYTHON_IMAGE}
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELO_SERVICIO_DIR=/app/models/servicio
RUN useradd --create-home --uid 10001 mibici
WORKDIR /app
COPY --from=dependencias /opt/venv /opt/venv
COPY config.py ./
COPY src/__init__.py src/api.py src/model_package.py src/p1_schema.py ./src/
COPY models/servicio/ ./models/servicio/
USER mibici
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
