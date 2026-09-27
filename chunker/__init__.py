"""PDF chunker for the SIP RFC corpus.

Extracts per-page text from the RFC PDFs (running headers/footers removed,
printed page numbers preserved), produces one cleaned body per RFC, and splits
it two ways: a naive fixed-window baseline and RFC-section-aware structural
chunking. Entry point: ``python -m chunker run``.
"""

from .schemas import Chunk, Page

__all__ = ["Chunk", "Page"]
