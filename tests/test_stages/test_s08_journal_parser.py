#!/usr/bin/env python3
"""
Tests for src/stages/s08_journal_parser.py

Tests cover:
- HTML to text conversion
- Text normalization
- Content type detection
- URL fetching helpers
"""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s08_journal_parser import (
    HTMLTextExtractor,
    normalize_text,
    html_to_text,
    decode_bytes,
    is_html_content,
    is_pdf_content,
    slugify,
    guess_extension,
    extract_text_from_bytes,
)


# ============================================================
# HTML TEXT EXTRACTOR TESTS
# ============================================================

class TestHTMLTextExtractor:
    """Tests for the HTMLTextExtractor class."""

    def test_extracts_simple_text(self):
        """Test extracting text from simple HTML."""
        parser = HTMLTextExtractor()
        parser.feed('<p>Hello World</p>')
        text = parser.get_text()

        assert 'Hello World' in text

    def test_handles_nested_tags(self):
        """Test handling nested tags."""
        parser = HTMLTextExtractor()
        parser.feed('<div><p>Nested <strong>text</strong></p></div>')
        text = parser.get_text()

        assert 'Nested' in text
        assert 'text' in text

    def test_adds_newlines_for_block_tags(self):
        """Test that block tags add newlines."""
        parser = HTMLTextExtractor()
        parser.feed('<p>First</p><p>Second</p>')
        text = parser.get_text()

        assert '\n' in text


# ============================================================
# NORMALIZE TEXT TESTS
# ============================================================

class TestNormalizeText:
    """Tests for the normalize_text function."""

    def test_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello    World"
        result = normalize_text(text)
        assert result == "Hello World"

    def test_converts_unicode_dashes(self):
        """Test Unicode dash conversion."""
        text = "Pages 1\u20132"  # en-dash
        result = normalize_text(text)
        assert '-' in result

    def test_removes_excess_newlines(self):
        """Test removal of excess newlines."""
        text = "First\n\n\n\nSecond"
        result = normalize_text(text)
        assert '\n\n\n' not in result

    def test_strips_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        text = "  Hello World  "
        result = normalize_text(text)
        assert result == "Hello World"


# ============================================================
# HTML TO TEXT TESTS
# ============================================================

class TestHtmlToText:
    """Tests for the html_to_text function."""

    def test_converts_simple_html(self):
        """Test converting simple HTML to text."""
        html = "<html><body><p>Hello World</p></body></html>"
        result = html_to_text(html)

        assert 'Hello World' in result

    def test_removes_tags(self):
        """Test that HTML tags are removed."""
        html = "<p><strong>Bold</strong> and <em>italic</em></p>"
        result = html_to_text(html)

        assert '<' not in result
        assert '>' not in result
        assert 'Bold' in result
        assert 'italic' in result

    def test_handles_lists(self):
        """Test handling HTML lists."""
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = html_to_text(html)

        assert 'Item 1' in result
        assert 'Item 2' in result


# ============================================================
# DECODE BYTES TESTS
# ============================================================

class TestDecodeBytes:
    """Tests for the decode_bytes function."""

    def test_decodes_utf8(self):
        """Test decoding UTF-8 bytes."""
        data = "Hello World".encode('utf-8')
        result = decode_bytes(data, 'text/plain; charset=utf-8')

        assert result == "Hello World"

    def test_decodes_latin1(self):
        """Test decoding Latin-1 bytes."""
        data = "Hello World".encode('latin-1')
        result = decode_bytes(data, 'text/plain; charset=latin-1')

        assert result == "Hello World"

    def test_defaults_to_utf8(self):
        """Test defaulting to UTF-8."""
        data = "Hello World".encode('utf-8')
        result = decode_bytes(data, 'text/plain')

        assert result == "Hello World"


# ============================================================
# CONTENT TYPE DETECTION TESTS
# ============================================================

class TestIsHtmlContent:
    """Tests for the is_html_content function."""

    def test_detects_html_content_type(self):
        """Test detecting HTML from content type."""
        assert is_html_content('text/html', None) is True
        assert is_html_content('text/html; charset=utf-8', None) is True

    def test_detects_html_extension(self):
        """Test detecting HTML from file extension."""
        assert is_html_content('', Path('file.html')) is True
        assert is_html_content('', Path('file.htm')) is True

    def test_returns_false_for_other(self):
        """Test returns False for non-HTML."""
        assert is_html_content('text/plain', None) is False
        assert is_html_content('', Path('file.txt')) is False


class TestIsPdfContent:
    """Tests for the is_pdf_content function."""

    def test_detects_pdf_content_type(self):
        """Test detecting PDF from content type."""
        assert is_pdf_content('application/pdf', None) is True

    def test_detects_pdf_extension(self):
        """Test detecting PDF from file extension."""
        assert is_pdf_content('', Path('file.pdf')) is True

    def test_returns_false_for_other(self):
        """Test returns False for non-PDF."""
        assert is_pdf_content('text/plain', None) is False
        assert is_pdf_content('', Path('file.txt')) is False


# ============================================================
# SLUGIFY TESTS
# ============================================================

class TestSlugify:
    """Tests for the slugify function."""

    def test_creates_slug(self):
        """Test creating URL-friendly slug."""
        result = slugify("Journal of Testing")
        assert result == "journal_of_testing"

    def test_removes_special_chars(self):
        """Test removing special characters."""
        result = slugify("Journal: Testing & More!")
        assert ':' not in result
        assert '&' not in result
        assert '!' not in result

    def test_handles_empty_string(self):
        """Test handling empty string."""
        result = slugify("")
        assert result == "journal"

    def test_handles_only_special_chars(self):
        """Test handling only special characters."""
        result = slugify("!@#$%")
        assert result == "journal"


# ============================================================
# GUESS EXTENSION TESTS
# ============================================================

class TestGuessExtension:
    """Tests for the guess_extension function."""

    def test_guesses_pdf(self):
        """Test guessing PDF extension."""
        assert guess_extension('application/pdf', None) == '.pdf'

    def test_guesses_html(self):
        """Test guessing HTML extension."""
        assert guess_extension('text/html', None) == '.html'

    def test_guesses_text(self):
        """Test guessing text extension."""
        assert guess_extension('text/plain', None) == '.txt'

    def test_uses_url_fallback(self):
        """Test using URL for fallback."""
        assert guess_extension('', 'http://example.com/file.pdf') == '.pdf'

    def test_defaults_to_txt(self):
        """Test defaulting to .txt."""
        assert guess_extension('application/unknown', None) == '.txt'


# ============================================================
# EXTRACT TEXT FROM BYTES TESTS
# ============================================================

class TestExtractTextFromBytes:
    """Tests for the extract_text_from_bytes function."""

    def test_extracts_plain_text(self):
        """Test extracting plain text."""
        data = b"Hello World"
        result = extract_text_from_bytes(data, 'text/plain')

        assert result == "Hello World"

    def test_extracts_html(self):
        """Test extracting text from HTML."""
        data = b"<p>Hello World</p>"
        result = extract_text_from_bytes(data, 'text/html')

        assert 'Hello World' in result

    def test_raises_for_pdf(self):
        """Test raises error for PDF content."""
        data = b"%PDF-1.4"

        with pytest.raises(ValueError, match="PDF guidelines detected"):
            extract_text_from_bytes(data, 'application/pdf')


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestJournalParserIntegration:
    """Integration tests for journal parser."""

    def test_full_html_extraction(self):
        """Test complete HTML extraction workflow."""
        html = """
        <html>
        <head><title>Guidelines</title></head>
        <body>
            <h1>Submission Guidelines</h1>
            <p>Maximum word count: 8000 words</p>
            <p>Abstract limit: 250 words</p>
        </body>
        </html>
        """

        text = html_to_text(html)

        assert 'Submission Guidelines' in text
        assert '8000' in text
        assert '250' in text
