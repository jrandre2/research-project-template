"""
OpenAI GPT Provider.

Implements the LLMProvider protocol for OpenAI's GPT models.

Usage
-----
    from llm.openai import OpenAIProvider

    provider = OpenAIProvider(model='gpt-4-turbo-preview')
    response = provider.complete("Summarize these results...")

Environment Variables
--------------------
OPENAI_API_KEY : str
    API key for OpenAI. Required for API calls.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    LLM provider implementation for OpenAI GPT.

    Parameters
    ----------
    model : str
        Model identifier (e.g., 'gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo').
    temperature : float
        Default sampling temperature (0.0-2.0).
    max_tokens : int
        Default maximum tokens for responses.
    api_key : str, optional
        API key. If not provided, uses OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str = 'gpt-4-turbo-preview',
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for the OpenAI provider. "
                    "Install it with: pip install openai"
                )

            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not found. Set the OPENAI_API_KEY "
                    "environment variable or pass api_key to the constructor."
                )

            self._client = openai.OpenAI(api_key=self._api_key)

        return self._client

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a completion using GPT.

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

        messages = []

        # Add system message if provided
        if system:
            messages.append({'role': 'system', 'content': system})

        messages.append({'role': 'user', 'content': prompt})

        response = client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._get_max_tokens(max_tokens),
            temperature=self._get_temperature(temperature),
        )

        # Extract text from response
        return response.choices[0].message.content

    def validate_connection(self) -> bool:
        """
        Check if the OpenAI API is accessible.

        Returns
        -------
        bool
            True if connection is valid.
        """
        try:
            client = self._get_client()
            # Make a minimal API call to verify connectivity
            response = client.chat.completions.create(
                model=self._model,
                messages=[{'role': 'user', 'content': 'Hi'}],
                max_tokens=10,
            )
            return bool(response.choices)
        except Exception:
            return False
