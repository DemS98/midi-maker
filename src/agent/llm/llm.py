from functools import lru_cache

from langchain_openai import ChatOpenAI

from .. import config


@lru_cache
def get_llm(model: str) -> ChatOpenAI:
    # No temperature: GPT-5 reasoning models only accept the default, and langchain drops anything else.
    return ChatOpenAI(model=model, reasoning_effort=config.REASONING_EFFORT)
