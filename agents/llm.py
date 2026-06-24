"""Shared LLM factory. Reads provider from config."""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm(provider: str = "anthropic", model: str = "claude-sonnet-4-6"):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=0.1,
            max_tokens=2048,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.1,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0.1,
            max_tokens=2048,
        )
    raise ValueError(f"Unknown provider: {provider}")
