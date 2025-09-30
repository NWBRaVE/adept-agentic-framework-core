# Dockerfile for both mcp_server and streamlit_app

# MCP Server stage
FROM python:3.11-slim AS mcp_server

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

ENV HOME=/app
WORKDIR ${HOME}

COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv
RUN uv venv -p 3.11 && uv pip install --no-cache-dir --editable .

RUN mkdir -p /app/data && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

CMD ["uv", "run", "run-mcp-server"]

# Streamlit App stage
FROM python:3.11-slim AS streamlit_app

RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

ENV HOME=/app
WORKDIR ${HOME}

RUN mkdir -p /app/certs

RUN openssl req -x509 -newkey rsa:4096 -nodes \
    -out /app/certs/cert.pem \
    -keyout /app/certs/key.pem \
    -days 3650 \
    -subj "/C=US/ST=Washington/L=Seattle/O=LocalDev/OU=IT/CN=localhost"

COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

RUN pip install --no-cache-dir uv
RUN uv venv -p 3.11 && uv pip install --no-cache-dir --editable .

RUN mkdir -p /app/data/uploaded_files && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8501

CMD ["uv", "run", "run-streamlit-harness"]