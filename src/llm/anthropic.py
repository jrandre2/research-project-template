"""
Anthropic Claude Provider.

Implements the LLMProvider protocol for Anthropic's Claude models.

Usage
-----
    from llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-sonnet-4-20250514')
    response = provider.complete("Summarize these results...")

Environment Variables
--------------------
ANTHROPIC_API_KEY : str
    API key for Anthropic. Required for API calls.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    LLM provider implementation for Anthropic Claude.

    Parameters
    ----------
    model : str
        Model identifier (e.g., 'claude-sonnet-4-20250514', 'claude-3-opus-20240229').
    temperature : float
        Default sampling temperature (0.0-1.0).
    max_tokens : int
        Default maximum tokens for responses.
    api_key : str, optional
        API key. If not provided, uses ANTHROPIC_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = 'claude-sonnet-4-20250514',
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for the Anthropic provider. "
                    "Install it with: pip install anthropic"
                )

            if not self._api_key:
                raise ValueError(
                    "Anthropic API key not found. Set the ANTHROPIC_API_KEY "
                    "environment variable or pass api_key to the constructor."
                )

            self._client = anthropic.Anthropic(api_key=self._api_key)

        return self._client

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a completion using Claude.

        Parameters
        ----------
        prompt : str
            The user prompt/message.
        system : str, optional
            System prompt for context.
        max_tokens : int, optional
            Maximum tokens in response.
        temperature : float, optional
            Sampling temperature.

        Returns
        -------
        str
            The generated completion text.
        """
        client = self._get_client()

        kwargs = {
            'model': self._model,
            'max_tokens': self._get_max_tokens(max_tokens),
            'messages': [{'role': 'user', 'content': prompt}],
        }

        # Only set temperature if specified (Anthropic has good defaults)
        temp = self._get_temperature(temperature)
        if temp is not None:
            kwargs['temperature'] = temp

        # Add system prompt if provided
        if system:
            kwargs['system'] = system

        response = client.messages.create(**kwargs)

        # Extract text from response
        return response.content[0].text

    def validate_connection(self) -> bool:
        """
        Check if the Anthropic API is accessible.

        Returns
        -------
        bool
            True if connection is valid.
        """
        try:
            client = self._get_client()
            # Make a minimal API call to verify connectivity
            response = client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{'role': 'user', 'content': 'Hi'}],
            )
            return bool(response.content)
        except Exception:
            return False
