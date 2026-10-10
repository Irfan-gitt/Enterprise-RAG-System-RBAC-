"""Round-robin LLM pool: same model, several API keys, automatic failover."""
import os
import threading

from dotenv import load_dotenv
from groq import APIConnectionError, InternalServerError, RateLimitError
from langchain_groq import ChatGroq

load_dotenv()

_FAILOVER_ERRORS = (RateLimitError, APIConnectionError, InternalServerError)


def _load_keys() -> list[str]:
    keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in (1, 2, 3)]
    keys = [k for k in keys if k]
    if not keys and os.getenv("GROQ_API_KEY"):
        keys = [os.getenv("GROQ_API_KEY")]
    if not keys:
        raise RuntimeError(
            "No Groq API keys found. Set GROQ_API_KEY_1..3 in .env")
    return keys


class LLMPool:
    def __init__(self, model: str, **kwargs):
        # max_retries=0: switch to the next key immediately instead of waiting on a 429
        kwargs.setdefault("max_retries", 0)
        self._models = [ChatGroq(model=model, api_key=key, **kwargs)
                        for key in _load_keys()]
        self._next = 0
        self._lock = threading.Lock()

    def invoke(self, prompt, **kwargs):
        count = len(self._models)
        with self._lock:  # spread calls evenly across keys
            start = self._next
            self._next = (self._next + 1) % count

        last_error = None
        for offset in range(count):
            index = (start + offset) % count
            try:
                return self._models[index].invoke(prompt, **kwargs)
            except _FAILOVER_ERRORS as error:
                last_error = error
                print(
                    f"Groq key #{index + 1} failed ({type(error).__name__}), trying next key")
        raise last_error  # every key failed
