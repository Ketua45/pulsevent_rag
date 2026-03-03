FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY API/requirements.txt /app/API/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r /app/API/requirements.txt

COPY API /app/API

RUN useradd -m -u 10001 appuser \
    && mkdir -p /app/vectorstore_normandie \
    && mkdir -p /cache/huggingface \
    && chown -R appuser:appuser /app /cache

USER appuser

ENTRYPOINT ["python", "-m", "API.main"]
CMD ["--help"]
