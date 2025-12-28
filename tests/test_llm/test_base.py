"""Tests for LLM base module and protocol."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from llm.base import LLMProvider, BaseLLMProvider


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider class."""

    def test_init_defaults(self):
        """Test default initialization."""
        provider = BaseLLMProvider(model='test-model')
        assert provider.model_name == 'test-model'
        assert provider._default_temperature == 0.3
        assert provider._default_max_tokens == 4096

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        provider = BaseLLMProvider(
            model='custom-model',
            temperature=0.7,
            max_tokens=2048,
        )
        assert provider.model_name == 'custom-model'
        assert provider._default_temperature == 0.7
        assert provider._default_max_tokens == 2048

    def test_get_temperature_default(self):
        """Test temperature fallback to default."""
        provider = BaseLLMProvider(model='test', temperature=0.5)
        assert provider._get_temperature(None) == 0.5

    def test_get_temperature_override(self):
        """Test temperature override."""
        provider = BaseLLMProvider(model='test', temperature=0.5)
        assert provider._get_temperature(0.8) == 0.8

    def test_get_max_tokens_default(self):
        """Test max_tokens fallback to default."""
        provider = BaseLLMProvider(model='test', max_tokens=1000)
        assert provider._get_max_tokens(None) == 1000

    def test_get_max_tokens_override(self):
        """Test max_tokens override."""
        provider = BaseLLMProvider(model='test', max_tokens=1000)
        assert provider._get_max_tokens(2000) == 2000


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for testing."""

    def __init__(self, response: str = "Mock response", **kwargs):
        super().__init__(**kwargs)
        self.response = response
        self.calls = []

    def complete(self, prompt, system=None, max_tokens=None, temperature=None):
        self.calls.append({
            'prompt': prompt,
            'system': system,
            'max_tokens': max_tokens,
            'temperature': temperature,
        })
        return self.response

    def validate_connection(self):
        return True


class TestMockProvider:
    """Tests for mock provider pattern."""

    def test_mock_complete(self):
        """Test mock provider returns configured response."""
        provider = MockLLMProvider(model='mock', response="Test output")
        result = provider.complete("Test prompt")
        assert result == "Test output"

    def test_mock_tracks_calls(self):
        """Test mock provider tracks calls."""
        provider = MockLLMProvider(model='mock')
        provider.complete("Prompt 1", system="System 1")
        provider.complete("Prompt 2")

        assert len(provider.calls) == 2
        assert provider.calls[0]['prompt'] == "Prompt 1"
        assert provider.calls[0]['system'] == "System 1"
        assert provider.calls[1]['prompt'] == "Prompt 2"

    def test_protocol_compliance(self):
        """Test mock provider satisfies LLMProvider protocol."""
        provider = MockLLMProvider(model='mock')
        assert isinstance(provider, LLMProvider)

    def test_validate_connection(self):
        """Test validate_connection returns True for mock."""
        provider = MockLLMProvider(model='mock')
        assert provider.validate_connection() is True
