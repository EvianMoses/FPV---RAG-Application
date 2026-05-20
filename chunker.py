"""
Semantic-aware recursive character text splitter for FPViber.

Why a custom splitter?
----------------------
Technical drone documents contain:
  - Numbered build steps that must stay together to be useful.
  - Tables (pipe-separated rows).
  - LaTeX-style math (`$...$`, `\\(...\\)`, `\\[...\\]`).
Naive sentence-tokenization (NLTK) frequently slices these blocks mid-formula
or mid-table, destroying the semantics.

This module:
  1. First isolates protected blocks (tables + math) as atomic units.
  2. Recursively splits the surrounding prose on a hierarchy of separators
     (paragraphs > line breaks > sentences > clauses > words).
  3. Greedily packs the pieces back into chunks of ~CHUNK_SIZE chars with
     CHUNK_OVERLAP characters of overlap, so neighbouring context is shared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CHUNK_SIZE = 900            # characters
CHUNK_OVERLAP = 150         # characters
HARD_MAX = 1600             # an atomic protected block can be larger than
                            # CHUNK_SIZE but we still cap it for safety

# Separators ordered from "biggest semantic boundary" to "smallest".
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


# ----------------------------------------------------------------------
# Protected block detection
# ----------------------------------------------------------------------

_MATH_PATTERNS = [
    re.compile(r"\$\$.+?\$\$", re.DOTALL),
    re.compile(r"\\\[.+?\\\]", re.DOTALL),
    re.compile(r"\\\(.+?\\\)", re.DOTALL),
    re.compile(r"\$[^$\n]+\$"),
]


@dataclass
class _Block:
    text: str
    protected: bool   # if True, do not split further


def _extract_math_blocks(text: str) -> list[_Block]:
    """Slice out math expressions and mark them protected."""
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    for pat in _MATH_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()

    # Merge overlapping spans
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    blocks: list[_Block] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            blocks.append(_Block(text[cursor:s], protected=False))
        blocks.append(_Block(text[s:e], protected=True))
        cursor = e
    if cursor < len(text):
        blocks.append(_Block(text[cursor:], protected=False))
    return blocks


def _extract_table_blocks(blocks: list[_Block]) -> list[_Block]:
    """
    Inside the non-protected blocks, glue consecutive pipe-separated lines
    together and mark them as protected (tables).
    """
    out: list[_Block] = []
    for block in blocks:
        if block.protected:
            out.append(block)
            continue

        lines = block.text.split("\n")
        buf: list[str] = []
        table_buf: list[str] = []
        for line in lines:
            if "|" in line and line.count("|") >= 1 and len(line.strip()) > 0:
                if buf:
                    out.append(_Block("\n".join(buf), protected=False))
                    buf = []
                table_buf.append(line)
            else:
                if table_buf:
                    out.append(_Block("\n".join(table_buf), protected=True))
                    table_buf = []
                buf.append(line)
        if table_buf:
            out.append(_Block("\n".join(table_buf), protected=True))
        if buf:
            out.append(_Block("\n".join(buf), protected=False))
    return out


# ----------------------------------------------------------------------
# Recursive splitter for free text
# ----------------------------------------------------------------------

def _recursive_split(text: str, separators: list[str] = SEPARATORS) -> list[str]:
    """Split `text` recursively using the separator hierarchy."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            pieces: list[str] = []
            for i, piece in enumerate(parts):
                if i < len(parts) - 1:
                    piece = piece + sep
                if len(piece) > CHUNK_SIZE:
                    pieces.extend(
                        _recursive_split(piece, separators[separators.index(sep) + 1:])
                    )
                else:
                    pieces.append(piece)
            return pieces

    # Fall back: hard slice
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


# ----------------------------------------------------------------------
# Packing
# ----------------------------------------------------------------------

def _pack(pieces: list[str]) -> list[str]:
    """Greedy pack pieces into chunks of ~CHUNK_SIZE with CHUNK_OVERLAP."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip("\n")
        if not piece.strip():
            continue

        if len(piece) > HARD_MAX:
            piece = piece[:HARD_MAX]

        if not current:
            current = piece
            continue

        if len(current) + 1 + len(piece) <= CHUNK_SIZE:
            sep = "\n" if "\n" in current[-2:] or "\n" in piece[:2] else " "
            current = f"{current}{sep}{piece}"
        else:
            chunks.append(current.strip())
            overlap_tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            current = f"{overlap_tail} {piece}".strip() if overlap_tail else piece

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def split_text(text: str) -> list[str]:
    """
    Split `text` into semantically coherent chunks, protecting tables
    and math from being broken in the middle.
    """
    if not text or not text.strip():
        return []

    blocks = _extract_math_blocks(text)
    blocks = _extract_table_blocks(blocks)

    pieces: list[str] = []
    for block in blocks:
        body = block.text.strip()
        if not body:
            continue
        if block.protected:
            pieces.append(body)
        else:
            pieces.extend(_recursive_split(body))

    return _pack(pieces)
