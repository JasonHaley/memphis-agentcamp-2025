#!/usr/bin/env python3
"""
pdf_to_markdown.py — Convert a PDF file to a Markdown file.

Usage:
    python pdf_to_markdown.py <input.pdf> [output.md]

If output path is omitted, the .md file is saved alongside the PDF
with the same base name.

Install dependencies:
    pip install pdfplumber pypdf
"""

import sys
import re
import os
import argparse
import pdfplumber
from pypdf import PdfReader


# ── Heuristics ────────────────────────────────────────────────────────────────

def looks_like_heading(line: str) -> tuple[bool, int]:
    """
    Return (is_heading, level) based on simple heuristics:
      - All-caps short line  → H2
      - Title-cased short line → H3
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False, 0

    word_count = len(stripped.split())
    if word_count > 12:
        return False, 0

    if stripped.isupper() and word_count <= 8:
        return True, 2
    if stripped.istitle() and word_count <= 10:
        return True, 3

    return False, 0


def clean_text(text: str) -> str:
    """Remove hyphenation at line breaks and tidy whitespace."""
    # Re-join hyphenated words split across lines
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines → single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ── Table helpers ──────────────────────────────────────────────────────────────

def table_to_markdown(table: list[list]) -> str:
    """Convert a pdfplumber table (list-of-lists) to a GFM Markdown table."""
    if not table:
        return ""

    # Normalise cells: replace None / newlines
    def cell(v):
        if v is None:
            return ""
        return str(v).replace("\n", " ").strip()

    rows = [[cell(c) for c in row] for row in table]

    # Determine column widths
    col_count = max(len(r) for r in rows)
    # Pad shorter rows
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    widths = [max(len(rows[i][j]) for i in range(len(rows))) for j in range(col_count)]
    widths = [max(w, 3) for w in widths]  # minimum width of 3 for the separator

    def fmt_row(row):
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[i] for i in range(col_count)) + " |"

    lines = [fmt_row(rows[0]), separator]
    for row in rows[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


# ── Page processor ─────────────────────────────────────────────────────────────

def process_page(page, page_num: int) -> str:
    """
    Extract content from a single pdfplumber page and return Markdown text.
    Tables are extracted first and their bounding boxes excluded from the text pass.
    """
    md_parts = []

    # ── Tables ──
    tables = page.extract_tables()
    table_bboxes = []

    if tables:
        # pdfplumber can give us table objects with bbox info
        found_tables = page.find_tables()
        for t_obj, t_data in zip(found_tables, tables):
            table_bboxes.append(t_obj.bbox)
            md_parts.append("\n\n" + table_to_markdown(t_data) + "\n\n")

    # ── Text (cropped to exclude table regions) ──
    text_page = page
    # Crop out table bounding boxes so their text isn't duplicated
    for bbox in table_bboxes:
        try:
            text_page = text_page.filter(
                lambda obj, b=bbox: not (
                    obj.get("x0", 0) >= b[0]
                    and obj.get("top", 0) >= b[1]
                    and obj.get("x1", 0) <= b[2]
                    and obj.get("bottom", 0) <= b[3]
                )
            )
        except Exception:
            pass  # If filtering fails, proceed without masking

    raw_text = text_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    raw_text = clean_text(raw_text)

    text_lines = raw_text.splitlines()
    text_md_parts = []
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            text_md_parts.append(" ".join(paragraph_buffer))
            paragraph_buffer.clear()

    for line in text_lines:
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            text_md_parts.append("")
            continue

        is_heading, level = looks_like_heading(stripped)
        if is_heading:
            flush_paragraph()
            text_md_parts.append(f"\n{'#' * level} {stripped}\n")
        else:
            paragraph_buffer.append(stripped)

    flush_paragraph()

    # Combine text output
    text_block = "\n".join(text_md_parts).strip()

    # Interleave: if tables were found, keep simple ordering (tables first per page)
    if table_bboxes:
        combined = text_block + "\n\n" + "\n\n".join(
            table_to_markdown(t) for t in tables
        )
    else:
        combined = text_block

    return combined.strip()


# ── Metadata helper ────────────────────────────────────────────────────────────

def extract_metadata(pdf_path: str) -> dict:
    """Extract PDF metadata using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        meta = reader.metadata or {}
        return {
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
            "subject": meta.get("/Subject", ""),
            "pages": len(reader.pages),
        }
    except Exception:
        return {}


# ── Main converter ─────────────────────────────────────────────────────────────

def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """
    Convert *pdf_path* to Markdown.

    Returns the output file path.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        base, _ = os.path.splitext(pdf_path)
        output_path = base + ".md"

    meta = extract_metadata(pdf_path)

    md_sections = []

    # Front-matter block (only if we have useful metadata)
    if meta.get("title") or meta.get("author"):
        front = []
        if meta.get("title"):
            front.append(f"# {meta['title']}")
        if meta.get("author"):
            front.append(f"*Author: {meta['author']}*")
        if meta.get("subject"):
            front.append(f"*Subject: {meta['subject']}*")
        md_sections.append("\n".join(front))

    # Page content
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"  Converting {total} page(s)…")

        for i, page in enumerate(pdf.pages, start=1):
            print(f"  Page {i}/{total}", end="\r", flush=True)
            page_md = process_page(page, i)
            if page_md:
                if total > 1:
                    md_sections.append(f"\n\n---\n<!-- page {i} -->\n\n{page_md}")
                else:
                    md_sections.append(page_md)

    print()  # newline after progress

    final_md = "\n\n".join(md_sections).strip() + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_md)

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to a Markdown (.md) file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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