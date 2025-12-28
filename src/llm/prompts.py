"""
Prompt Templates for AI-Assisted Manuscript Writing.

This module contains carefully crafted prompts for generating manuscript content.
Each prompt is designed to produce academic-quality output while avoiding
common pitfalls (metacommentary, file references, etc.).

Usage
-----
    from llm.prompts import build_results_prompt, build_caption_prompt

    prompt, system = build_results_prompt(df, variable_map)
    response = provider.complete(prompt, system=system)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

ACADEMIC_WRITER_SYSTEM = """You are an expert academic writer specializing in quantitative research for peer-reviewed journals.

Your writing must:
- Be precise, objective, and professional
- Follow standard academic conventions
- Report findings accurately without overstating conclusions
- Use appropriate hedging language where uncertainty exists
- Never include metacommentary, file paths, or internal references
- Never reference scripts, code, or implementation details

Write as if producing final manuscript text ready for journal submission."""


CAPTION_WRITER_SYSTEM = """You are an expert at writing figure captions for academic publications.

Your captions must:
- Begin with a brief, descriptive title
- Clearly describe what the figure shows
- Note key patterns or findings visible in the figure
- Include relevant details (units, sample sizes, time periods)
- Be self-contained (reader should understand without reading main text)
- Follow standard academic figure caption conventions

Never include file paths, internal references, or metacommentary."""


ABSTRACT_WRITER_SYSTEM = """You are an expert at writing concise abstracts for academic publications.

Your abstracts must:
- State the research question clearly
- Briefly describe the methodology
- Report key findings with specific results
- State the contribution and implications
- Be self-contained and compelling
- Stay within the specified word limit

Never include citations, file paths, or metacommentary."""


# =============================================================================
# RESULTS SECTION PROMPTS
# =============================================================================

RESULTS_SECTION_TEMPLATE = """Given the following estimation results, write a results section for an academic manuscript.

## Estimation Results Table

{table_content}

## Variable Descriptions

{variable_descriptions}

## Additional Context

{additional_context}

## Instructions

Write 2-3 paragraphs that:
1. State the main finding clearly and directly
2. Report key coefficient estimates with confidence intervals or standard errors
3. Note statistical significance using appropriate language (e.g., "statistically significant at the 5% level")
4. Interpret the magnitude in practical terms relevant to the research context
5. Acknowledge any notable patterns in secondary results

Do NOT:
- Use phrases like "As shown in Table X" or "The table shows"
- Include hedging language unless truly warranted by the results
- Reference file names, scripts, or data sources
- Add metacommentary about the writing process

Write the results section:"""


def build_results_prompt(
    table_df: pd.DataFrame,
    variable_map: Optional[dict] = None,
    additional_context: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build a prompt for results section drafting.

    Parameters
    ----------
    table_df : pd.DataFrame
        DataFrame containing estimation results.
    variable_map : dict, optional
        Mapping of variable names to descriptions.
    additional_context : str, optional
        Additional context about the analysis.

    Returns
    -------
    tuple[str, str]
        (user_prompt, system_prompt) ready for LLM completion.
    """
    # Convert DataFrame to markdown table
    table_content = table_df.to_markdown(index=False)

    # Format variable descriptions
    if variable_map:
        var_descriptions = "\n".join(
            f"- **{var}**: {desc}" for var, desc in variable_map.items()
        )
    else:
        var_descriptions = "Not provided. Infer from variable names where possible."

    # Handle additional context
    context = additional_context or "No additional context provided."

    prompt = RESULTS_SECTION_TEMPLATE.format(
        table_content=table_content,
        variable_descriptions=var_descriptions,
        additional_context=context,
    )

    return prompt, ACADEMIC_WRITER_SYSTEM


# =============================================================================
# FIGURE CAPTION PROMPTS
# =============================================================================

CAPTION_TEMPLATE = """Generate a complete figure caption for an academic manuscript.

## Figure Information

Filename: {filename}
Figure type: {figure_type}

## Context from Manuscript

{context}

## Instructions

Write a figure caption that:
1. Starts with a brief title (one descriptive sentence)
2. Describes what the figure displays
3. Notes any key patterns, trends, or findings visible
4. Includes relevant technical details (units, time periods, sample information)

Format: "Figure X. [Title]. [Description]. [Key observation]. [Technical details]."

The caption should be 2-4 sentences and self-contained.

Generate the caption (without "Figure X." prefix - that will be added later):"""


def build_caption_prompt(
    filename: str,
    context: Optional[str] = None,
    figure_type: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build a prompt for figure caption generation.

    Parameters
    ----------
    filename : str
        Name of the figure file.
    context : str, optional
        Context about the figure from the manuscript.
    figure_type : str, optional
        Type of figure (e.g., 'scatter plot', 'bar chart', 'time series').

    Returns
    -------
    tuple[str, str]
        (user_prompt, system_prompt) ready for LLM completion.
    """
    # Infer figure type from filename if not provided
    if figure_type is None:
        filename_lower = filename.lower()
        if 'trend' in filename_lower or 'time' in filename_lower:
            figure_type = "Time series or trend plot"
        elif 'scatter' in filename_lower:
            figure_type = "Scatter plot"
        elif 'bar' in filename_lower:
            figure_type = "Bar chart"
        elif 'coef' in filename_lower or 'effect' in filename_lower:
            figure_type = "Coefficient or effect plot"
        elif 'dist' in filename_lower or 'hist' in filename_lower:
            figure_type = "Distribution or histogram"
        elif 'map' in filename_lower:
            figure_type = "Map or spatial visualization"
        else:
            figure_type = "Unknown - infer from context"

    context_text = context or "No context provided. Generate a general caption based on the filename."

    prompt = CAPTION_TEMPLATE.format(
        filename=filename,
        figure_type=figure_type,
        context=context_text,
    )

    return prompt, CAPTION_WRITER_SYSTEM


# =============================================================================
# ABSTRACT PROMPTS
# =============================================================================

ABSTRACT_TEMPLATE = """Synthesize an abstract for an academic manuscript.

## Target Length

{max_words} words (approximately {sentence_count} sentences)

## Manuscript Sections

### Introduction / Research Question
{introduction}

### Methods
{methods}

### Key Results
{results}

### Conclusion / Implications
{conclusion}

## Instructions

Write an abstract that:
1. Opens with 1 sentence stating the research question or motivation
2. Briefly describes the methodology (1-2 sentences)
3. Reports key findings with specific numbers where available (2-3 sentences)
4. Closes with the main contribution or implication (1 sentence)

The abstract must be:
- Self-contained (no citations or references to sections)
- Compelling and clear to a general academic audience
- Within the {max_words} word limit

Generate the abstract:"""


def build_abstract_prompt(
    introduction: str,
    methods: str,
    results: str,
    conclusion: str,
    max_words: int = 250,
) -> tuple[str, str]:
    """
    Build a prompt for abstract synthesis.

    Parameters
    ----------
    introduction : str
        Text from the introduction section.
    methods : str
        Text from the methods section.
    results : str
        Text from the results section.
    conclusion : str
        Text from the conclusion section.
    max_words : int
        Target word limit for the abstract.

    Returns
    -------
    tuple[str, str]
        (user_prompt, system_prompt) ready for LLM completion.
    """
    # Estimate sentence count (average academic sentence ~20 words)
    sentence_count = max(3, max_words // 25)

    prompt = ABSTRACT_TEMPLATE.format(
        max_words=max_words,
        sentence_count=sentence_count,
        introduction=introduction or "Not provided.",
        methods=methods or "Not provided.",
        results=results or "Not provided.",
        conclusion=conclusion or "Not provided.",
    )

    return prompt, ABSTRACT_WRITER_SYSTEM


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def truncate_text(text: str, max_chars: int = 3000) -> str:
    """
    Truncate text to fit within token limits.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_chars : int
        Maximum characters to keep.

    Returns
    -------
    str
        Truncated text with indicator if truncated.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to break at a sentence boundary
    last_period = truncated.rfind('. ')
    if last_period > max_chars * 0.7:
        truncated = truncated[:last_period + 1]

    return truncated + "\n\n[... text truncated for length ...]"
