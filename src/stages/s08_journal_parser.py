#!/usr/bin/env python3
"""
Stage 08: Journal Guidelines Parser

Purpose: Parse and manage journal configuration files for manuscript submission.

Commands
--------
parse     : Parse raw journal guidelines into structured YAML config
fetch     : Download journal guidelines from a URL
validate  : Validate a journal config against the comprehensive template
compare   : Compare manuscript against journal requirements
list      : List available journal configurations

Usage
-----
    python src/pipeline.py journal_parse --input guidelines.txt --output journal.yml
    python src/pipeline.py journal_parse --url https://example.com/guidelines --output journal.yml --save-raw
    python src/pipeline.py journal_fetch --url https://example.com/guidelines --text
    python src/pipeline.py journal_validate --config natural_hazards.yml
    python src/pipeline.py journal_compare --journal natural_hazards
    python src/pipeline.py journal_list
"""
from __future__ import annotations

import re
import yaml
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Iterable, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stages._qa_utils import generate_qa_report, QAMetrics

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MANUSCRIPT_DIR = PROJECT_ROOT / 'manuscript_quarto'
JOURNAL_CONFIGS = MANUSCRIPT_DIR / 'journal_configs'
TEMPLATE_FILE = JOURNAL_CONFIGS / 'template_comprehensive.yml'
RAW_GUIDELINES_DIR = PROJECT_ROOT / 'doc' / 'journal_guidelines'

# Extraction prompt for LLM-based parsing
EXTRACTION_PROMPT = '''
Extract journal submission requirements from the following author guidelines.
Output as YAML matching this template structure:

{template}

Guidelines to parse:
{raw_guidelines}

Be comprehensive. Include all numeric limits, format specifications, and policies mentioned.
For fields not specified in the guidelines, use null.
'''

DEFAULT_USER_AGENT = 'CENTAUR/0.1 (journal-guidelines-parser)'


class HTMLTextExtractor(HTMLParser):
    """Convert HTML into plain text with basic structure preservation."""

    BLOCK_TAGS = {
        'p', 'br', 'li', 'div', 'tr', 'td', 'th',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'section', 'article', 'header', 'footer', 'nav',
    }

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]) -> None:
        if tag in self.BLOCK_TAGS:
            self._chunks.append('\n')

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self._chunks.append('\n')

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return ''.join(self._chunks)


def normalize_text(text: str) -> str:
    """Normalize whitespace for parsing."""
    text = text.replace('\r', '\n')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2264', '<=')
    text = text.replace('\u20ac', ' EUR ').replace('\u00a3', ' GBP ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def decode_bytes(data: bytes, content_type: str) -> str:
    """Decode bytes using charset from content type when available."""
    charset = None
    if 'charset=' in content_type:
        charset = content_type.split('charset=')[-1].split(';')[0].strip()

    for encoding in (charset, 'utf-8', 'latin-1'):
        if not encoding:
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode('utf-8', errors='replace')


def html_to_text(html: str) -> str:
    """Strip HTML markup into readable text."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return normalize_text(parser.get_text())


def guess_extension(content_type: str, url: Optional[str]) -> str:
    """Guess file extension from content type or URL."""
    if 'application/pdf' in content_type:
        return '.pdf'
    if 'text/html' in content_type:
        return '.html'
    if 'text/plain' in content_type:
        return '.txt'

    if url:
        path = urlparse(url).path
        if path.endswith('.pdf'):
            return '.pdf'
        if path.endswith('.html') or path.endswith('.htm'):
            return '.html'
        if path.endswith('.txt'):
            return '.txt'

    return '.txt'


def is_html_content(content_type: str, path: Optional[Path]) -> bool:
    if 'text/html' in content_type:
        return True
    if path and path.suffix.lower() in ('.html', '.htm'):
        return True
    return False


def is_pdf_content(content_type: str, path: Optional[Path]) -> bool:
    if 'application/pdf' in content_type:
        return True
    if path and path.suffix.lower() == '.pdf':
        return True
    return False


def extract_text_from_bytes(
    data: bytes,
    content_type: str = '',
    path_hint: Optional[Path] = None
) -> str:
    """Convert raw bytes to parseable text."""
    if is_pdf_content(content_type, path_hint):
        raise ValueError(
            "PDF guidelines detected. Convert to text or HTML before parsing."
        )

    decoded = decode_bytes(data, content_type)
    if is_html_content(content_type, path_hint):
        return html_to_text(decoded)

    return normalize_text(decoded)


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug."""
    value = re.sub(r'[^A-Za-z0-9]+', '_', value.strip())
    value = value.strip('_')
    return value.lower() or 'journal'


def fetch_guidelines(
    url: str,
    timeout: int = 20,
    user_agent: Optional[str] = None
) -> tuple[bytes, str, str]:
    """Download guidelines from a URL."""
    req = Request(url, headers={'User-Agent': user_agent or DEFAULT_USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        data = response.read()
        content_type = response.headers.get('Content-Type', '')
        final_url = response.geturl()
    return data, content_type, final_url


def save_raw_guidelines(
    data: bytes,
    filename: str,
    output_dir: Path,
    overwrite: bool = False
) -> Path:
    """Save raw guideline bytes to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")

    output_path.write_bytes(data)
    return output_path


def find_context_windows(
    text: str,
    keyword: str,
    window: int = 220,
    max_windows: int = 6
) -> Iterable[str]:
    """Yield context windows around keyword occurrences."""
    count = 0
    for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        yield text[start:end]
        count += 1
        if count >= max_windows:
            break


def extract_range(text: str, unit_pattern: str) -> Optional[Tuple[int, int]]:
    pattern = rf'(\d{{1,5}})\s*(?:-|to)\s*(\d{{1,5}})\s*{unit_pattern}'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def extract_max(text: str, unit_pattern: str) -> Optional[int]:
    pattern = rf'(?:no more than|max(?:imum)?|up to|<=)\s*(\d{{1,5}})\s*{unit_pattern}'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_min(text: str, unit_pattern: str) -> Optional[int]:
    pattern = rf'(?:at least|min(?:imum)?(?: of)?)\s*(\d{{1,5}})\s*{unit_pattern}'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_limits_for_keyword(
    text: str,
    keyword: str,
    unit_pattern: str
) -> tuple[Optional[int], Optional[int]]:
    for window in find_context_windows(text, keyword):
        range_match = extract_range(window, unit_pattern)
        if range_match:
            return range_match
        max_val = extract_max(window, unit_pattern)
        min_val = extract_min(window, unit_pattern)
        if max_val or min_val:
            return min_val, max_val
    return None, None


def extract_manuscript_word_limit(text: str) -> Optional[int]:
    keywords = ['word limit', 'manuscript', 'article', 'paper', 'main text']
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        line_lower = line.lower()
        if 'word' not in line_lower:
            continue
        if not any(k in line_lower for k in keywords):
            continue
        if 'abstract' in line_lower or 'keyword' in line_lower:
            continue
        range_match = extract_range(line, 'words?')
        if range_match:
            return range_match[1]
        max_val = extract_max(line, 'words?')
        if max_val:
            return max_val
        word_limit_match = re.search(r'word limit\s*[:\-]?\s*(\d{3,6})', line, re.IGNORECASE)
        if word_limit_match:
            return int(word_limit_match.group(1))
    return None


def extract_peer_review(text: str) -> Optional[str]:
    lower = text.lower()
    if re.search(r'double[-\s]blind|double[-\s]anonym', lower):
        return 'double-blind'
    if re.search(r'single[-\s]blind|single[-\s]anonym', lower):
        return 'single-blind'
    if 'open peer review' in lower:
        return 'open'
    return None


def extract_submission_system(text: str) -> Optional[str]:
    systems = {
        'Editorial Manager': ['editorial manager', 'elsevier editorial system'],
        'ScholarOne': ['scholarone', 'manuscript central'],
        'OJS': ['open journal systems', 'ojs'],
        'EVISE': ['evise'],
    }
    lower = text.lower()
    for name, patterns in systems.items():
        if any(pat in lower for pat in patterns):
            return name
    return None


def extract_submission_fee(text: str) -> Optional[int]:
    match = re.search(
        r'(submission fee|submission charge|handling fee).*?\$?\s*([0-9][0-9,]{1,6})',
        text,
        re.IGNORECASE
    )
    if match:
        return int(match.group(2).replace(',', ''))
    return None


def extract_line_spacing(text: str) -> Optional[str]:
    lower = text.lower()
    if 'double-spaced' in lower or 'double spaced' in lower:
        return 'double'
    if '1.5-spaced' in lower or '1.5 spaced' in lower or 'one-and-a-half' in lower:
        return '1.5'
    if 'single-spaced' in lower or 'single spaced' in lower:
        return 'single'
    return None


def extract_font(text: str) -> tuple[Optional[str], Optional[int]]:
    fonts = ['Times New Roman', 'Arial', 'Helvetica', 'Calibri', 'Cambria', 'Garamond', 'Georgia']
    for font in fonts:
        match = re.search(rf'{re.escape(font)}.*?(\d{{1,2}})\s*(?:pt|point)', text, re.IGNORECASE)
        if match:
            return font, int(match.group(1))
    return None, None


def extract_margins(text: str) -> Optional[dict]:
    match = re.search(
        r'(\d+(?:\.\d+)?)\s*(inch|inches|cm|mm)\s*margins?',
        text,
        re.IGNORECASE
    )
    if not match:
        return None

    size = float(match.group(1))
    unit = match.group(2).lower()
    unit = 'in' if unit in ('inch', 'inches') else unit
    return {'unit': unit, 'top': size, 'bottom': size, 'left': size, 'right': size}


def extract_reference_style(text: str) -> Optional[str]:
    lower = text.lower()
    if 'author-year' in lower or 'author date' in lower or 'harvard' in lower or 'apa' in lower:
        return 'author-year'
    if 'vancouver' in lower or 'numeric' in lower or 'numbered' in lower:
        return 'numeric'
    return None


def extract_formats(text: str, formats: list[str]) -> list[str]:
    found = []
    for fmt in formats:
        if fmt.startswith('.'):
            pattern = re.escape(fmt)
        else:
            pattern = rf'\b{re.escape(fmt)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            found.append(fmt)
    return sorted(set(found))


def extract_dpi(text: str, keyword: str) -> Optional[int]:
    for window in find_context_windows(text, keyword):
        match = re.search(r'(\d{3,4})\s*dpi', window, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_data_availability_required(text: str) -> Optional[bool]:
    for window in find_context_windows(text, 'data availability'):
        lower = window.lower()
        if any(term in lower for term in ('required', 'must', 'mandatory')):
            return True
    return None


def extract_apc(text: str) -> tuple[Optional[int], Optional[str]]:
    match = re.search(
        r'(article processing charge|apc).*?([0-9][0-9,]{1,6})\s*(usd|eur|gbp|\$)?',
        text,
        re.IGNORECASE
    )
    if not match:
        return None, None

    amount = int(match.group(2).replace(',', ''))
    currency = match.group(3)
    if currency == '$':
        currency = 'USD'
    elif currency:
        currency = currency.upper()
    return amount, currency


def set_config_value(config: dict, path: tuple[str, ...], value) -> None:
    """Set a nested config value if provided."""
    if value is None:
        return
    if isinstance(value, list) and not value:
        return

    cursor = config
    for key in path[:-1]:
        if key not in cursor or cursor[key] is None:
            cursor[key] = {}
        cursor = cursor[key]

    cursor[path[-1]] = value


def extract_guideline_updates(text: str) -> tuple[dict, list[str]]:
    """Extract guideline values and return update map plus summary lines."""
    updates: dict[tuple[str, ...], object] = {}
    summary: list[str] = []

    abstract_min, abstract_max = extract_limits_for_keyword(text, 'abstract', 'words?')
    if abstract_min is not None:
        updates[('abstract', 'min_words')] = abstract_min
        summary.append(f"abstract.min_words = {abstract_min}")
    if abstract_max is not None:
        updates[('abstract', 'max_words')] = abstract_max
        summary.append(f"abstract.max_words = {abstract_max}")

    kw_min, kw_max = extract_limits_for_keyword(text, 'keyword', r'key\s*words?')
    if kw_min is not None:
        updates[('keywords', 'min')] = kw_min
        summary.append(f"keywords.min = {kw_min}")
    if kw_max is not None:
        updates[('keywords', 'max')] = kw_max
        summary.append(f"keywords.max = {kw_max}")

    word_limit = extract_manuscript_word_limit(text)
    if word_limit is not None:
        updates[('text_limits', 'word_limit')] = word_limit
        updates[('paper_types', 'research_article', 'word_limit')] = word_limit
        summary.append(f"text_limits.word_limit = {word_limit}")

    peer_review = extract_peer_review(text)
    if peer_review:
        updates[('submission', 'peer_review')] = peer_review
        summary.append(f"submission.peer_review = {peer_review}")

    system = extract_submission_system(text)
    if system:
        updates[('submission', 'system')] = system
        summary.append(f"submission.system = {system}")

    fee = extract_submission_fee(text)
    if fee is not None:
        updates[('submission', 'fee')] = fee
        summary.append(f"submission.fee = {fee}")

    line_spacing = extract_line_spacing(text)
    if line_spacing:
        updates[('text_formatting', 'line_spacing')] = line_spacing
        summary.append(f"text_formatting.line_spacing = {line_spacing}")

    font_family, font_size = extract_font(text)
    if font_family:
        updates[('text_formatting', 'font', 'family')] = font_family
        summary.append(f"text_formatting.font.family = {font_family}")
    if font_size:
        updates[('text_formatting', 'font', 'size_pt')] = font_size
        summary.append(f"text_formatting.font.size_pt = {font_size}")

    margins = extract_margins(text)
    if margins:
        for key, value in margins.items():
            updates[('text_formatting', 'margins', key)] = value
        summary.append("text_formatting.margins = detected")

    ref_style = extract_reference_style(text)
    if ref_style:
        updates[('references', 'style')] = ref_style
        summary.append(f"references.style = {ref_style}")

    vector_formats = extract_formats(text, ['EPS', 'PDF', 'SVG'])
    if vector_formats:
        updates[('artwork', 'formats', 'vector_preferred')] = vector_formats
        summary.append(f"artwork.formats.vector_preferred = {', '.join(vector_formats)}")

    raster_formats = extract_formats(text, ['TIFF', 'TIF', 'JPEG', 'JPG', 'PNG'])
    if raster_formats:
        updates[('artwork', 'formats', 'raster_acceptable')] = raster_formats
        summary.append(f"artwork.formats.raster_acceptable = {', '.join(raster_formats)}")

    line_art_dpi = extract_dpi(text, 'line art')
    if line_art_dpi:
        updates[('artwork', 'resolution', 'line_art_dpi')] = line_art_dpi
        summary.append(f"artwork.resolution.line_art_dpi = {line_art_dpi}")

    halftone_dpi = extract_dpi(text, 'halftone')
    if halftone_dpi:
        updates[('artwork', 'resolution', 'halftone_dpi')] = halftone_dpi
        summary.append(f"artwork.resolution.halftone_dpi = {halftone_dpi}")

    combination_dpi = extract_dpi(text, 'combination')
    if combination_dpi:
        updates[('artwork', 'resolution', 'combination_dpi')] = combination_dpi
        summary.append(f"artwork.resolution.combination_dpi = {combination_dpi}")

    file_formats = extract_formats(text, ['.docx', '.doc', '.tex'])
    if 'latex' in text.lower():
        file_formats.append('.tex')
        updates[('text_formatting', 'latex_accepted')] = True
    if file_formats:
        file_formats = sorted(set(file_formats))
        updates[('text_formatting', 'file_formats')] = file_formats
        updates[('submission', 'source_files', 'formats')] = file_formats
        summary.append(f"text_formatting.file_formats = {', '.join(file_formats)}")

    data_availability = extract_data_availability_required(text)
    if data_availability is True:
        updates[('declarations', 'data_availability', 'required')] = True
        updates[('submission', 'required_documents', 'data_availability_statement', 'required')] = True
        summary.append("declarations.data_availability.required = true")

    apc_amount, apc_currency = extract_apc(text)
    if apc_amount is not None:
        updates[('open_access', 'apc', 'amount')] = apc_amount
        summary.append(f"open_access.apc.amount = {apc_amount}")
    if apc_currency:
        updates[('open_access', 'apc', 'currency')] = apc_currency
        summary.append(f"open_access.apc.currency = {apc_currency}")

    return updates, summary


def apply_guideline_updates(config: dict, updates: dict) -> None:
    """Apply extracted updates to a config dict."""
    for path, value in updates.items():
        set_config_value(config, path, value)


def list_configs() -> None:
    """List available journal configurations."""
    print("Available Journal Configurations")
    print("=" * 50)

    if not JOURNAL_CONFIGS.exists():
        print("\nNo journal configs directory found.")
        return

    configs = sorted(JOURNAL_CONFIGS.glob('*.yml'))
    template_files = ['template.yml', 'template_comprehensive.yml']

    print("\nTemplates:")
    for f in configs:
        if f.name in template_files:
            print(f"  - {f.name}")

    print("\nJournal Configs:")
    for f in configs:
        if f.name not in template_files:
            # Load and show basic info
            try:
                with open(f) as fp:
                    config = yaml.safe_load(fp)
                name = config.get('journal', {}).get('name', 'Unknown')
                abbrev = config.get('journal', {}).get('abbreviation', '')
                print(f"  - {f.stem}: {name} ({abbrev})")
            except Exception:
                print(f"  - {f.stem}: (error reading config)")

    print("\n" + "=" * 50)


def validate_config(config_name: str) -> None:
    """Validate a journal config against the comprehensive template."""
    print(f"Validating: {config_name}")
    print("=" * 50)

    config_file = JOURNAL_CONFIGS / f'{config_name}.yml'
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return

    if not TEMPLATE_FILE.exists():
        print(f"ERROR: Template file not found: {TEMPLATE_FILE}")
        return

    # Load files
    with open(config_file) as f:
        config = yaml.safe_load(f)
    with open(TEMPLATE_FILE) as f:
        template = yaml.safe_load(f)

    # Check required sections
    required_sections = [
        'journal', 'submission', 'abstract', 'keywords',
        'text_formatting', 'references', 'declarations',
        'tables', 'artwork', 'ethical_responsibilities',
        'authorship', 'data_policy'
    ]

    missing = []
    present = []

    for section in required_sections:
        if section in config and config[section]:
            present.append(section)
        else:
            missing.append(section)

    print(f"\nRequired Sections: {len(present)}/{len(required_sections)} present")
    print("\nPresent:")
    for s in present:
        print(f"  [x] {s}")

    if missing:
        print("\nMissing:")
        for s in missing:
            print(f"  [ ] {s}")

    # Check key fields
    print("\nKey Field Validation:")

    checks = [
        ('journal.name', config.get('journal', {}).get('name')),
        ('journal.publisher', config.get('journal', {}).get('publisher')),
        ('abstract.max_words', config.get('abstract', {}).get('max_words')),
        ('keywords.max', config.get('keywords', {}).get('max')),
        ('references.style', config.get('references', {}).get('style')),
    ]

    for field, value in checks:
        status = "[x]" if value else "[ ]"
        print(f"  {status} {field}: {value}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('config_name', config_name)
    metrics.add('sections_present', len(present))
    metrics.add('sections_missing', len(missing))
    metrics.add_pct('completeness', (len(present) / len(required_sections)) * 100)
    generate_qa_report('s08_journal_validate', metrics)

    print("\n" + "=" * 50)


def compare_manuscript(journal_name: str, manuscript_path: Optional[str] = None) -> None:
    """Compare manuscript against journal requirements."""
    print(f"Comparing manuscript to: {journal_name}")
    print("=" * 50)

    config_file = JOURNAL_CONFIGS / f'{journal_name}.yml'
    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_file}")
        return

    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Default manuscript location
    if manuscript_path is None:
        manuscript_path = MANUSCRIPT_DIR

    manuscript_dir = Path(manuscript_path)
    if not manuscript_dir.exists():
        print(f"ERROR: Manuscript directory not found: {manuscript_dir}")
        return

    print(f"\nManuscript: {manuscript_dir}")
    print(f"Journal: {config.get('journal', {}).get('name', 'Unknown')}")

    # Check for required files
    print("\nRequired Files:")
    required_files = [
        ('index.qmd', 'Main manuscript'),
        ('references.bib', 'Bibliography'),
    ]

    for filename, desc in required_files:
        filepath = manuscript_dir / filename
        status = "[x]" if filepath.exists() else "[ ]"
        print(f"  {status} {filename} - {desc}")

    # Check abstract requirements
    abstract_config = config.get('abstract', {})
    print(f"\nAbstract Requirements:")
    print(f"  Min words: {abstract_config.get('min_words', 'Not specified')}")
    print(f"  Max words: {abstract_config.get('max_words', 'Not specified')}")
    print(f"  No abbreviations: {abstract_config.get('no_abbreviations', False)}")
    print(f"  No citations: {abstract_config.get('no_citations', False)}")

    # Check keyword requirements
    keywords_config = config.get('keywords', {})
    print(f"\nKeyword Requirements:")
    print(f"  Min: {keywords_config.get('min', 'Not specified')}")
    print(f"  Max: {keywords_config.get('max', 'Not specified')}")

    # Check figure requirements
    artwork_config = config.get('artwork', {})
    resolution = artwork_config.get('resolution', {})
    print(f"\nFigure Resolution Requirements:")
    print(f"  Line art: {resolution.get('line_art_dpi', 'Not specified')} dpi")
    print(f"  Halftone: {resolution.get('halftone_dpi', 'Not specified')} dpi")
    print(f"  Combination: {resolution.get('combination_dpi', 'Not specified')} dpi")

    # Check reference style
    refs_config = config.get('references', {})
    print(f"\nReference Style:")
    print(f"  Style: {refs_config.get('style', 'Not specified')}")
    print(f"  In-text format: {refs_config.get('in_text_format', 'Not specified')}")

    # Submission checklist
    submission = config.get('submission', {})
    checklist = submission.get('checklist', [])
    if checklist:
        print(f"\nSubmission Checklist:")
        for item in checklist:
            print(f"  [ ] {item}")

    print("\n" + "=" * 50)


def parse_guidelines(
    input_file: Optional[str],
    output_file: str,
    journal_name: Optional[str] = None,
    url: Optional[str] = None,
    template_name: str = 'template_comprehensive',
    save_raw: bool = False,
    raw_dir: Optional[str] = None,
    overwrite: bool = False
) -> None:
    """Parse raw guidelines into structured config."""
    print("Journal Guidelines Parser")
    print("=" * 50)

    if not input_file and not url:
        print("ERROR: Provide --input or --url for parsing.")
        return

    output_path = JOURNAL_CONFIGS / output_file
    if not output_file.endswith('.yml'):
        output_path = JOURNAL_CONFIGS / f'{output_file}.yml'

    raw_text = ''
    source_label = ''
    guidelines_url = None
    raw_path = None

    if url:
        try:
            raw_bytes, content_type, final_url = fetch_guidelines(url)
        except Exception as exc:
            print(f"ERROR: Failed to fetch guidelines: {exc}")
            return

        guidelines_url = final_url
        source_label = final_url

        if save_raw:
            slug = slugify(journal_name or Path(output_path).stem)
            ext = guess_extension(content_type, final_url)
            filename = f"{slug}{ext}"
            output_dir = Path(raw_dir) if raw_dir else RAW_GUIDELINES_DIR
            try:
                raw_path = save_raw_guidelines(raw_bytes, filename, output_dir, overwrite=overwrite)
            except FileExistsError as exc:
                print(f"ERROR: {exc}")
                return

        try:
            raw_text = extract_text_from_bytes(raw_bytes, content_type, raw_path)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return
    else:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERROR: Input file not found: {input_path}")
            return

        source_label = str(input_path)
        raw_bytes = input_path.read_bytes()
        try:
            raw_text = extract_text_from_bytes(raw_bytes, '', input_path)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return

    print(f"\nSource: {source_label}")
    print(f"Output: {output_path}")
    print(f"\nGuidelines length: {len(raw_text)} characters")

    # Load template
    template_file = template_name if template_name.endswith('.yml') else f'{template_name}.yml'
    template_path = JOURNAL_CONFIGS / template_file
    if not template_path.exists():
        print(f"ERROR: Template file not found: {template_path}")
        return

    with open(template_path) as f:
        config = yaml.safe_load(f)

    # Set journal metadata if provided
    if journal_name:
        config.setdefault('journal', {})
        config['journal']['name'] = journal_name
    if guidelines_url:
        config.setdefault('journal', {})
        config['journal']['guidelines_url'] = guidelines_url

    updates, summary = extract_guideline_updates(raw_text)
    apply_guideline_updates(config, updates)

    # Write output
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nCreated: {output_path}")
    if raw_path:
        print(f"Saved guidelines: {raw_path}")
    if summary:
        print("\nExtracted fields:")
        for line in summary:
            print(f"  - {line}")
    else:
        print("\nNo structured fields extracted; manual review required.")

    print("\nNext steps:")
    print("1. Review and edit the generated config")
    print("2. Fill in missing fields from the guidelines")
    print("3. Create a Quarto profile if needed (_quarto-{abbrev}.yml)")
    print("4. Test with: python src/pipeline.py journal_validate --config {output_file}")


def fetch_guidelines_cli(
    url: str,
    output: Optional[str] = None,
    journal_name: Optional[str] = None,
    raw_dir: Optional[str] = None,
    overwrite: bool = False,
    save_text: bool = False
) -> None:
    """Fetch guidelines and save them locally."""
    print("Journal Guidelines Fetcher")
    print("=" * 50)

    try:
        raw_bytes, content_type, final_url = fetch_guidelines(url)
    except Exception as exc:
        print(f"ERROR: Failed to fetch guidelines: {exc}")
        return

    ext = guess_extension(content_type, final_url)
    if output:
        output_path = Path(output)
        if output_path.parent == Path('.'):
            output_dir = Path(raw_dir) if raw_dir else RAW_GUIDELINES_DIR
            output_path = output_dir / output_path.name
        elif not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        slug = slugify(journal_name or Path(urlparse(final_url).path).stem or 'journal')
        output_dir = Path(raw_dir) if raw_dir else RAW_GUIDELINES_DIR
        output_path = output_dir / f"{slug}{ext}"

    try:
        output_path = save_raw_guidelines(raw_bytes, output_path.name, output_path.parent, overwrite=overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"\nSource: {final_url}")
    print(f"Saved: {output_path}")

    if save_text:
        try:
            text = extract_text_from_bytes(raw_bytes, content_type, output_path)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return

        text_path = output_path.with_suffix('.txt')
        if text_path.exists() and not overwrite:
            print(f"ERROR: Text file already exists: {text_path}")
            return
        text_path.write_text(text)
        print(f"Text version: {text_path}")


def main(action: str = 'list', **kwargs) -> None:
    """Main entry point for journal parser."""
    if action == 'list':
        list_configs()
    elif action == 'validate':
        config = kwargs.get('config', 'template_comprehensive')
        validate_config(config)
    elif action == 'compare':
        journal = kwargs.get('journal', 'natural_hazards')
        manuscript = kwargs.get('manuscript')
        compare_manuscript(journal, manuscript)
    elif action == 'fetch':
        url = kwargs.get('url')
        output = kwargs.get('output')
        journal = kwargs.get('journal')
        raw_dir = kwargs.get('raw_dir')
        overwrite = kwargs.get('overwrite', False)
        save_text = kwargs.get('text', False)
        if not url:
            print("ERROR: --url required for fetching")
            return
        fetch_guidelines_cli(
            url=url,
            output=output,
            journal_name=journal,
            raw_dir=raw_dir,
            overwrite=overwrite,
            save_text=save_text
        )
    elif action == 'parse':
        input_file = kwargs.get('input')
        url = kwargs.get('url')
        output = kwargs.get('output', 'new_journal.yml')
        journal = kwargs.get('journal')
        template_name = kwargs.get('template', 'template_comprehensive')
        save_raw = kwargs.get('save_raw', False)
        raw_dir = kwargs.get('raw_dir')
        overwrite = kwargs.get('overwrite', False)
        if not input_file and not url:
            print("ERROR: --input or --url required for parsing")
            return
        parse_guidelines(
            input_file=input_file,
            output_file=output,
            journal_name=journal,
            url=url,
            template_name=template_name,
            save_raw=save_raw,
            raw_dir=raw_dir,
            overwrite=overwrite
        )
    else:
        print(f"Unknown action: {action}")
        print("Available: list, validate, compare, fetch, parse")


if __name__ == '__main__':
    main()
