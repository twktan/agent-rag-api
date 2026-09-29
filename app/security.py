"""API-key authentication and cost-protecting rate limits.

Rate limits are in-memory, so they are exact per instance. The service is deployed with
max-instances=1 (scripts/deploy.sh), which makes them global. Scaling out would move this
state to Redis/Memorystore; the interface would not change.
"""

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.observability.metrics import RATE_LIMITED

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="API key issued by the maintainer")


def key_id(api_key: str) -> str:
    """Non-reversible short id for logs and quotas; the key itself is never logged."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:10]


def make_api_key_dependency(settings: Settings):
    valid_keys = settings.api_key_set

    async def require_api_key(api_key: str | None = Security(api_key_header)) -> str:
        if not settings.auth_enabled:
            return "anonymous"
        # Compare against every key in constant time to avoid timing side channels.
        if api_key and any(hmac.compare_digest(api_key, k) for k in valid_keys):
            return key_id(api_key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key",
                            headers={"WWW-Authenticate": "ApiKey"})

    return require_api_key


class RateLimiter:
    """Sliding 60 s window per key + daily quota per key + global daily quota (UTC days)."""

    def __init__(self, per_minute: int, daily_per_key: int, global_daily: int, clock=time.time):
        self.per_minute, self.daily_per_key, self.global_daily = per_minute, daily_per_key, global_daily
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._daily: dict[str, int] = defaultdict(int)
        self._global = 0

    def _roll_day(self, now: float) -> None:
        day = datetime.fromtimestamp(now, tz=UTC).strftime("%Y-%m-%d")
        if day != self._day:
            self._day, self._daily, self._global = day, defaultdict(int), 0

    @staticmethod
    def _reject(scope: str, detail: str, retry_after: int) -> HTTPException:
        RATE_LIMITED.labels(scope=scope).inc()
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail,
                             headers={"Retry-After": str(max(1, retry_after))})

    def check(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            seconds_to_midnight = int(86400 - now % 86400)
            if self._global >= self.global_daily:
                raise self._reject("global_daily", "Service daily quota reached", seconds_to_midnight)
            if self._daily[key] >= self.daily_per_key:
                raise self._reject("key_daily", "Daily quota for this API key reached", seconds_to_midnight)
            window = self._windows[key]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                raise self._reject("key_minute", "Rate limit exceeded", int(60 - (now - window[0])) + 1)
            window.append(now)
            self._daily[key] += 1
            self._global += 1
