"""
LLM Provider Package for CENTAUR.

This package provides a provider-agnostic interface for LLM interactions,
supporting multiple backends (Anthropic Claude, OpenAI GPT-4) with a unified API.

Usage
-----
    from llm import get_provider

    # Get default provider (from config)
    provider = get_provider()

    # Get specific provider
    provider = get_provider('anthropic')
    provider = get_provider('openai')

    # Generate completion
    response = provider.complete(
        prompt="Summarize these results...",
        system="You are an academic writing assistant.",
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LLMProvider


def get_provider(provider_name: str = None) -> 'LLMProvider':
    """
    Get an LLM provider instance.

    Parameters
    ----------
    provider_name : str, optional
        Provider name ('anthropic' or 'openai').
        If not specified, uses LLM_PROVIDER from config.

    Returns
    -------
    LLMProvider
        Configured provider instance.

    Raises
    ------
    ValueError
        If provider_name is not recognized.
    ImportError
        If the required SDK is not installed.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import LLM_PROVIDER, LLM_MODELS, LLM_TEMPERATURE, LLM_MAX_TOKENS

    provider_name = provider_name or LLM_PROVIDER
    model = LLM_MODELS.get(provider_name)

    if provider_name == 'anthropic':
        from .anthropic import AnthropicProvider
        return AnthropicProvider(
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    elif provider_name == 'openai':
        from .openai import OpenAIProvider
        return OpenAIProvider(
            model=model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Supported providers: 'anthropic', 'openai'"
        )


__all__ = ['get_provider']
