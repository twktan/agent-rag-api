import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, env="test", api_keys="test-key,second-key", auth_enabled=True,
                    moderation_enabled=True, output_moderation_enabled=True, metrics_enabled=True,
                    min_similarity=0.5, top_k=3, max_top_k=6, max_refinements=2,
                    rate_limit_per_minute=100, daily_quota_per_key=1000, global_daily_quota=1000,
                    mlflow_tracking_uri=None)
