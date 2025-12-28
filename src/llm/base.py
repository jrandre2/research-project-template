"""
Base Protocol for LLM Providers.

Defines the interface that all LLM providers must implement.
Uses Python's Protocol for structural subtyping (duck typing with type hints).

Usage
-----
    from llm.base import LLMProvider

    def generate_draft(provider: LLMProvider, prompt: str) -> str:
        return provider.complete(prompt)
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol defining the interface for LLM providers.

    All providers must implement these methods to be compatible
    with the CENTAUR drafting system.

    Attributes
    ----------
    model_name : str
        The model identifier being used.

    Methods
    -------
    complete(prompt, system, max_tokens, temperature)
        Generate a completion for the given prompt.
    validate_connection()
        Check if the provider is properly configured.
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier (e.g., 'claude-sonnet-4-20250514')."""
        ...

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            The user prompt/message to complete.
        system : str, optional
            System prompt to set context/behavior.
        max_tokens : int, optional
            Maximum tokens in the response. Uses provider default if not specified.
        temperature : float, optional
            Sampling temperature (0.0-1.0). Uses provider default if not specified.

        Returns
        -------
        str
            The generated completion text.

        Raises
        ------
        ConnectionError
            If the API is unreachable.
        AuthenticationError
            If the API key is invalid.
        RateLimitError
            If rate limits are exceeded.
        """
        ...

    def validate_connection(self) -> bool:
        """
        Check if the provider is properly configured and reachable.

        Returns
        -------
        bool
            True if connection is valid, False otherwise.
        """
        ...


class BaseLLMProvider:
    """
    Base implementation with common functionality for LLM providers.

    Provides default implementations for shared behavior.
    Concrete providers should inherit from this class.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """
        Initialize provider with default settings.

        Parameters
        ----------
        model : str
            Model identifier to use.
        temperature : float
            Default sampling temperature.
        max_tokens : int
            Default maximum tokens for responses.
        """
        self._model = model
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

    def _get_temperature(self, temperature: Optional[float]) -> float:
        """Get temperature, using default if not specified."""
        return temperature if temperature is not None else self._default_temperature

    def _get_max_tokens(self, max_tokens: Optional[int]) -> int:
        """Get max_tokens, using default if not specified."""
        return max_tokens if max_tokens is not None else self._default_max_tokens
