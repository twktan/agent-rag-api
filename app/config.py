"""Environment-driven configuration. Every tunable lives here, nowhere else."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # utf-8-sig: also reads a .env saved with a BOM (some Windows editors add one), which would
    # otherwise corrupt the first key name.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8-sig", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    service_name: str = "agent-rag-api"

    # --- LLM ---
    openai_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4.1"  # offline evaluation only; deliberately stronger than the generator
    llm_temperature: float = 0.2
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    max_tokens_router: int = 150
    max_tokens_answer: int = 600
    max_tokens_critic: int = 500

    # --- Retrieval (build-time values are frozen into the index manifest) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    # Selected by dev-split MRR in eval/ablate_retrieval.py (see README "Retrieval ablation").
    chunker: Literal["word", "char"] = "char"
    chunk_size: int = 500  # characters for the char chunker, words for the word chunker
    chunk_overlap: int = 100
    chunk_header: bool = True
    embed_threads: int = 2  # match the container's vCPUs
    docs_dir: str = "data/docs"
    index_dir: str = "vectorstore"
    top_k: int = 3
    max_top_k: int = 6
    # Floor, not a router: just below the lowest top-1 score of any in-domain dev question (0.555),
    # so chunks under it are noise. See eval/results/retrieval_dev.json -> similarity_router.
    min_similarity: float = 0.50

    # --- Agent ---
    max_refinements: int = 2

    # --- Guardrails ---
    moderation_enabled: bool = True
    output_moderation_enabled: bool = True
    moderation_model: str = "omni-moderation-latest"

    # --- API security & cost control ---
    auth_enabled: bool = True
    api_keys: SecretStr = SecretStr("")  # comma-separated
    rate_limit_per_minute: int = 10
    daily_quota_per_key: int = 200
    global_daily_quota: int = 1000
    cache_ttl_s: int = 3600
    cache_max_entries: int = 256

    # --- Observability ---
    log_level: str = "INFO"
    log_query_text: bool = False  # off by default: queries may contain personal data
    metrics_enabled: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "agent-rag-system"
    google_cloud_project: str | None = None

    @property
    def api_key_set(self) -> frozenset[str]:
        raw = self.api_keys.get_secret_value()
        return frozenset(k.strip() for k in raw.split(",") if k.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
