"""Shared LLM factory. Reads provider from config."""

import os
import time
from functools import lru_cache


class _RateLimitRetry:
    """Wraps any LangChain chat model with automatic 429 retry."""
    def __init__(self, llm, max_retries: int = 6, base_wait: float = 25.0):
        self._llm = llm
        self._max_retries = max_retries
        self._base_wait = base_wait

    def invoke(self, *args, **kwargs):
        for attempt in range(self._max_retries):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    wait = self._base_wait * (attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        return self._llm.invoke(*args, **kwargs)

    # pass-through so isinstance checks / other callers don't break
    def __getattr__(self, name):
        return getattr(self._llm, name)


@lru_cache(maxsize=1)
def get_llm(provider: str = "anthropic", model: str = "claude-sonnet-4-6"):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.1,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=model,
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.1,
            max_tokens=2048,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _RateLimitRetry(llm)
