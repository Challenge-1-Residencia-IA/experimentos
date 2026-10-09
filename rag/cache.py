"""Cache limitado para pesquisa web, com expiração e acesso concorrente."""

from collections import OrderedDict
from copy import deepcopy
from threading import Lock
from time import monotonic


class ResearchCache:
    def __init__(self, ttl: float = 300, capacity: int = 128):
        if ttl <= 0 or capacity < 1:
            raise ValueError("TTL e capacidade devem ser positivos")
        self.ttl = ttl
        self.capacity = capacity
        self._values = OrderedDict()
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None
            timestamp, value = entry
            if monotonic() - timestamp >= self.ttl:
                del self._values[key]
                return None
            self._values.move_to_end(key)
            return deepcopy(value)

    def put(self, key, value):
        with self._lock:
            self._values[key] = (monotonic(), deepcopy(value))
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
