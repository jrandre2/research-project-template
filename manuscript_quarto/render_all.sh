#!/bin/bash
# Render all Quarto output formats (HTML, PDF, DOCX)
#
# IMPORTANT: Quarto book projects overwrite _output on each render.
# This script renders all formats sequentially and preserves outputs.
#
# Usage: ./render_all.sh [--profile <name>]
# Examples:
#   ./render_all.sh                    # Default profile
#   ./render_all.sh --profile jeem     # JEEM submission format
#   ./render_all.sh --profile aer      # AER submission format

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
PROFILE_ARG=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --profile) PROFILE_ARG="--profile $2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Ensure we have the quarto binary
QUARTO="../tools/bin/quarto"
if [ ! -f "$QUARTO" ]; then
    QUARTO="quarto"  # Fall back to system quarto
fi

echo "=== Rendering Quarto Manuscript (All Formats) ==="
[ -n "$PROFILE_ARG" ] && echo "    Using profile: $PROFILE_ARG"

# Create temp directory to preserve outputs
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Render HTML first
echo ""
echo ">>> Rendering HTML..."
$QUARTO render --to html $PROFILE_ARG
cp -r _output/* "$TEMP_DIR/"
echo "    HTML saved."

# Render PDF
echo ""
echo ">>> Rendering PDF..."
$QUARTO render --to pdf $PROFILE_ARG
cp _output/*.pdf "$TEMP_DIR/" 2>/dev/null || true
cp _output/*.tex "$TEMP_DIR/" 2>/dev/null || true
echo "    PDF saved."

# Render DOCX
echo ""
echo ">>> Rendering DOCX..."
$QUARTO render --to docx $PROFILE_ARG
cp _output/*.docx "$TEMP_DIR/" 2>/dev/null || true
echo "    DOCX saved."

# Restore all outputs to _output
echo ""
echo ">>> Combining all outputs..."
rm -rf _output/*
cp -r "$TEMP_DIR"/* _output/

echo ""
echo "=== Done! All formats in _output/ ==="
ls -la _output/*.html _output/*.pdf _output/*.docx 2>/dev/null || ls -la _output/
