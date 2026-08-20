# Dockerfile — Industrial Causal Policy Benchmark
FROM python:3.11-slim

# Instalar dependências de sistema para compiladores e bibliotecas geoespaciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    graphviz \
    libgdal-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar arquivos de dependências
COPY requirements.txt pyproject.toml ./

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar o código fonte do projeto
COPY . /app

# Definir variável de ambiente para PYTHONPATH
ENV PYTHONPATH=/app

# Comando padrão: executa todo o pipeline end-to-end e salva os resultados em /app/results
CMD ["python", "main.py", "--step", "all"]
