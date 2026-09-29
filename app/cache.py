"""Small TTL + LRU response cache. Demo traffic is dominated by the same example questions,
so exact-match caching removes most of the LLM spend for repeat queries."""

import re
import threading
import time
from collections import OrderedDict


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower()


class TTLCache:
    def __init__(self, max_entries: int, ttl_s: float, clock=time.monotonic):
        self.max_entries, self.ttl_s, self._clock = max_entries, ttl_s, clock
        self._data: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if self._clock() - stored_at > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: dict) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
