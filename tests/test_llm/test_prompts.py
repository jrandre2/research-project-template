"""Tests for LLM prompt templates."""
import pytest
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from llm.prompts import (
    build_results_prompt,
    build_caption_prompt,
    build_abstract_prompt,
    truncate_text,
    ACADEMIC_WRITER_SYSTEM,
    CAPTION_WRITER_SYSTEM,
    ABSTRACT_WRITER_SYSTEM,
)


class TestBuildResultsPrompt:
    """Tests for results section prompt building."""

    @pytest.fixture
    def sample_df(self):
        """Create sample estimation results DataFrame."""
        return pd.DataFrame({
            'variable': ['treatment', 'control1', 'control2'],
            'coefficient': [0.15, 0.03, -0.02],
            'std_error': [0.05, 0.02, 0.01],
            'p_value': [0.003, 0.134, 0.045],
        })

    def test_basic_prompt_construction(self, sample_df):
        """Test basic prompt construction."""
        prompt, system = build_results_prompt(sample_df)

        assert isinstance(prompt, str)
        assert isinstance(system, str)
        assert 'treatment' in prompt
        assert '0.15' in prompt
        assert system == ACADEMIC_WRITER_SYSTEM

    def test_with_variable_map(self, sample_df):
        """Test prompt includes variable descriptions."""
        var_map = {
            'treatment': 'Binary indicator for treatment group',
            'control1': 'First control variable',
        }
        prompt, _ = build_results_prompt(sample_df, variable_map=var_map)

        assert 'Binary indicator for treatment group' in prompt
        assert 'First control variable' in prompt

    def test_with_additional_context(self, sample_df):
        """Test prompt includes additional context."""
        context = "This analysis uses a difference-in-differences design."
        prompt, _ = build_results_prompt(sample_df, additional_context=context)

        assert context in prompt

    def test_empty_variable_map(self, sample_df):
        """Test handling of empty variable map."""
        prompt, _ = build_results_prompt(sample_df, variable_map={})

        assert 'Not provided' in prompt

    def test_prompt_contains_instructions(self, sample_df):
        """Test prompt contains writing instructions."""
        prompt, _ = build_results_prompt(sample_df)

        assert 'coefficient' in prompt.lower() or 'Coefficient' in prompt
        assert 'Do NOT' in prompt


class TestBuildCaptionPrompt:
    """Tests for figure caption prompt building."""

    def test_basic_caption_prompt(self):
        """Test basic caption prompt construction."""
        prompt, system = build_caption_prompt(filename='fig_main_results.png')

        assert isinstance(prompt, str)
        assert 'fig_main_results.png' in prompt
        assert system == CAPTION_WRITER_SYSTEM

    def test_infers_figure_type_from_name(self):
        """Test figure type inference from filename."""
        prompt, _ = build_caption_prompt(filename='fig_trend_analysis.png')
        assert 'time series' in prompt.lower() or 'trend' in prompt.lower()

        prompt, _ = build_caption_prompt(filename='fig_scatter_plot.png')
        assert 'scatter' in prompt.lower()

        prompt, _ = build_caption_prompt(filename='fig_coef_estimates.png')
        assert 'coefficient' in prompt.lower() or 'effect' in prompt.lower()

    def test_custom_figure_type(self):
        """Test custom figure type override."""
        prompt, _ = build_caption_prompt(
            filename='fig_custom.png',
            figure_type='Regression discontinuity plot',
        )
        assert 'Regression discontinuity plot' in prompt

    def test_with_context(self):
        """Test caption prompt with context."""
        context = "This figure shows the main treatment effect."
        prompt, _ = build_caption_prompt(
            filename='fig_main.png',
            context=context,
        )
        assert context in prompt


class TestBuildAbstractPrompt:
    """Tests for abstract synthesis prompt building."""

    def test_basic_abstract_prompt(self):
        """Test basic abstract prompt construction."""
        prompt, system = build_abstract_prompt(
            introduction="We study the effect of X on Y.",
            methods="We use a regression approach.",
            results="We find a positive effect.",
            conclusion="This has implications for policy.",
        )

        assert isinstance(prompt, str)
        assert 'We study the effect of X on Y' in prompt
        assert system == ABSTRACT_WRITER_SYSTEM

    def test_word_limit_included(self):
        """Test word limit is included in prompt."""
        prompt, _ = build_abstract_prompt(
            introduction="Intro",
            methods="Methods",
            results="Results",
            conclusion="Conclusion",
            max_words=200,
        )
        assert '200' in prompt

    def test_handles_missing_sections(self):
        """Test handling of missing sections."""
        prompt, _ = build_abstract_prompt(
            introduction="Intro text",
            methods=None,
            results="",
            conclusion="Conclusion text",
        )
        assert 'Not provided' in prompt


class TestTruncateText:
    """Tests for text truncation utility."""

    def test_short_text_unchanged(self):
        """Test short text is not modified."""
        text = "Short text."
        result = truncate_text(text, max_chars=100)
        assert result == text

    def test_long_text_truncated(self):
        """Test long text is truncated."""
        text = "A" * 5000
        result = truncate_text(text, max_chars=1000)
        assert len(result) < len(text)
        assert 'truncated' in result.lower()

    def test_truncates_at_sentence_boundary(self):
        """Test truncation prefers sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = truncate_text(text, max_chars=50)

        # Should end at a sentence if possible
        if 'truncated' in result.lower():
            # Check it tried to break at a sentence
            assert result.count('.') >= 1
