#!/usr/bin/env python3
"""
pdf_to_markdown.py — Convert a PDF file to Markdown using markitdown.

Usage:
    python pdf_to_markdown.py <input.pdf> [output.md]

If output path is omitted, the .md file is saved alongside the PDF
with the same base name.

Install dependency:
    pip install markitdown
"""

import sys
import os
import argparse
from markitdown import MarkItDown


def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """
    Convert *pdf_path* to Markdown and save it.
    Returns the output file path.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        base, _ = os.path.splitext(pdf_path)
        output_path = base + ".md"

    md = MarkItDown()
    result = md.convert(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to a Markdown (.md) file using markitdown.",
    )
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path for the output .md file (default: same directory as input)",
    )
    args = parser.parse_args()

    print(f"📄 Input : {args.input}")
    try:
        out = pdf_to_markdown(args.input, args.output)
        print(f"✅ Saved : {out}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
