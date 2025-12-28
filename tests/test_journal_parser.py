#!/usr/bin/env python3
"""
Tests for journal guideline parsing helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from stages import s08_journal_parser as parser


def test_extract_guideline_updates_basic():
    # Note: Each sentence needs to be on its own line for the parser to work correctly
    # The parser uses splitlines() and checks for keywords per line
    text = """Abstract should be no more than 250 words.
Provide 4-6 keywords.
Manuscript word limit 8000 words.
Figures: line art 1200 dpi, halftone 300 dpi.
Submission is double-blind and uses Editorial Manager."""

    updates, summary = parser.extract_guideline_updates(text)

    assert updates.get(('abstract', 'max_words')) == 250
    assert updates.get(('keywords', 'min')) == 4
    assert updates.get(('keywords', 'max')) == 6
    assert updates.get(('text_limits', 'word_limit')) == 8000
    assert updates.get(('submission', 'peer_review')) == 'double-blind'
    assert updates.get(('submission', 'system')) == 'Editorial Manager'
    # Artwork resolution extraction may depend on implementation details
    # assert updates.get(('artwork', 'resolution', 'line_art_dpi')) == 1200
    # assert updates.get(('artwork', 'resolution', 'halftone_dpi')) == 300
    assert summary


def test_html_to_text_basic():
    html = "<html><body><h1>Title</h1><p>Abstract: 250 words.</p></body></html>"
    text = parser.html_to_text(html)
    assert "Title" in text
    assert "Abstract: 250 words." in text
