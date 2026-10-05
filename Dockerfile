FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    OMP_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch first: the default PyPI wheel pulls ~2.5 GB of CUDA libraries we never use.
COPY requirements.txt .
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app/ app/
COPY data/docs/ data/docs/

# Build the index into the image: the embedding model is downloaded once at build time, so
# cold starts never call Hugging Face and every image carries exactly one index version.
RUN python -m app.rag.build_index

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN useradd --system --uid 10001 appuser && chown -R appuser /app /opt/hf
USER appuser

EXPOSE 8080
# Cloud Run injects $PORT. One worker per container: Cloud Run scales by instances, and the
# in-memory rate limiter assumes one process.
CMD ["sh", "-c", "exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-access-log"]
