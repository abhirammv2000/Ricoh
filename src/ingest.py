"""
src/ingest.py - PDF parsing and metadata-preserving chunking pipeline.

This is the **heart of Phase 1**.  It:

1. Discovers all PDFs in ``data/``.
2. Extracts text **page-by-page** via PyMuPDF, keeping each page's
   ``source_document`` (filename) and ``page_number``.
3. Splits page text into sliding-window chunks (~500 words, 50-word
   overlap) that **never lose** their page-number provenance.
4. Returns a flat list of chunk dicts ready for ChromaDB + BM25
   insertion in Phase 2.

Design decisions
────────────────
• We use PyMuPDF (``import fitz``) because it gives us precise
  page-number control and handles scanned-text PDFs well.
• "Token" is approximated by whitespace-split words - simple,
  deterministic, zero extra dependencies.
• Each chunk records the page(s) it originated from so the
  hackathon rubric's *strict citation* requirement is met.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF - page-level PDF text extraction

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
)

# ── Logging (configured centrally in config.py) ──
logger = logging.getLogger(__name__)


# ====================================================================
# 1. PDF TEXT EXTRACTION
# ====================================================================

def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from every page of a single PDF.

    Args:
        pdf_path: Absolute or relative path to a ``.pdf`` file.

    Returns:
        A list of dicts, one per page::

            {
                "text":            <str>,   # raw page text
                "page_number":     <int>,   # 1-indexed
                "source_document": <str>,   # filename (stem + ext)
            }

    Why page-level extraction?
        The hackathon rubric demands **exact page citations**.  By
        capturing the page number at extraction time we guarantee
        downstream chunks never lose this provenance.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    source_name = pdf_path.name  # e.g. "Ricoh_IM_C3500_Manual.pdf"

    pages: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")  # plain-text extraction

        # Skip empty / image-only pages that yield no useful text
        if not text or not text.strip():
            logger.debug(
                "Skipping empty page %d in %s", page_idx + 1, source_name
            )
            continue

        pages.append(
            {
                "text": text.strip(),
                "page_number": page_idx + 1,       # 1-indexed for humans
                "source_document": source_name,
            }
        )

    doc.close()
    logger.info(
        "Extracted %d non-empty pages from '%s'.", len(pages), source_name
    )
    return pages


# ====================================================================
# 2. SLIDING-WINDOW CHUNKER  (metadata-safe)
# ====================================================================

def _generate_chunk_id(source: str, page: int | str, index: int) -> str:
    """Deterministic chunk ID = sha256(source|page|index)[:16].

    A short hash avoids collisions while staying human-readable
    in ChromaDB logs.
    """
    raw = f"{source}|{page}|{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split page texts into overlapping word-level chunks.

    Each chunk inherits the ``source_document`` and ``page_number``
    from its parent page.

    Args:
        pages:         Output of :func:`extract_pages`.
        chunk_size:    Max words per chunk.
        chunk_overlap: Words of overlap between consecutive chunks.

    Returns:
        Flat list of chunk dicts::

            {
                "id":              <str>,   # deterministic hash
                "text":            <str>,   # chunk content
                "page_number":     <int>,   # originating page
                "source_document": <str>,   # originating PDF filename
                "chunk_index":     <int>,   # position within that page
            }

    Why NOT cross page boundaries?
        Mixing text from different pages makes page-number citation
        ambiguous.  We chunk **within** each page so every chunk maps
        to exactly ONE page number - clean citations, zero ambiguity.
    """
    chunks: list[dict[str, Any]] = []
    global_idx = 0  # running counter for unique IDs across all pages

    for page in pages:
        words = page["text"].split()
        source = page["source_document"]
        page_num = page["page_number"]

        # If the page has fewer words than one chunk, emit as-is
        if len(words) <= chunk_size:
            chunks.append(
                {
                    "id": _generate_chunk_id(source, page_num, global_idx),
                    "text": " ".join(words),
                    "page_number": page_num,
                    "source_document": source,
                    "chunk_index": global_idx,
                }
            )
            global_idx += 1
            continue

        # Sliding window with `chunk_overlap` word overlap
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]

            chunks.append(
                {
                    "id": _generate_chunk_id(source, page_num, global_idx),
                    "text": " ".join(chunk_words),
                    "page_number": page_num,
                    "source_document": source,
                    "chunk_index": global_idx,
                }
            )
            global_idx += 1

            # Advance by (chunk_size - overlap) words
            start += chunk_size - chunk_overlap

            # Avoid creating a tiny trailing chunk (< overlap size)
            remaining = len(words) - start
            if 0 < remaining <= chunk_overlap:
                break

    logger.info(
        "Chunked %d pages into %d chunks (size=%d, overlap=%d).",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


# ====================================================================
# 3. ORCHESTRATOR - ingest all PDFs in data/
# ====================================================================

def ingest_all(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    """Discover PDFs in *data_dir*, extract + chunk them all.

    Args:
        data_dir: Directory containing raw PDF files.

    Returns:
        Flat list of chunk dicts (same schema as :func:`chunk_pages`).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning("Data directory '%s' does not exist.", data_dir)
        return []

    pdf_files = sorted(
        f for f in data_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", data_dir)
        return []

    logger.info("Found %d PDF(s) in '%s'.", len(pdf_files), data_dir)

    all_chunks: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    logger.info(
        "Ingestion complete: %d total chunks from %d PDF(s).",
        len(all_chunks),
        len(pdf_files),
    )
    return all_chunks


# ====================================================================
# 4. __main__ - quick smoke test with a generated sample PDF
# ====================================================================

def _create_sample_pdf(path: Path) -> None:
    """Generate a tiny multi-page PDF for testing the pipeline.

    We create the PDF programmatically via PyMuPDF so there is
    zero dependency on external files.
    """
    doc = fitz.open()  # new empty PDF

    sample_texts = [
        (
            "Page 1: Ricoh IM C3500 Overview. "
            "The Ricoh IM C3500 is a versatile multifunction printer "
            "designed for modern office environments. It supports high-speed "
            "color printing at up to 35 pages per minute and offers advanced "
            "scanning capabilities with optical character recognition. "
            "This device integrates seamlessly with cloud-based document "
            "management systems, allowing teams to collaborate efficiently. "
            "Key features include a 10.1-inch Smart Operation Panel, "
            "RICOH Always Current Technology for firmware updates, and "
            "robust security protocols including user authentication "
            "and data encryption. The IM C3500 is ideal for workgroups "
            "of 5 to 20 users who require reliable, high-quality output. "
            "Maintenance is simplified through Ricoh's proactive service "
            "platform which monitors toner levels and device health remotely."
        ),
        (
            "Page 2: Paper Handling and Tray Configuration. "
            "The IM C3500 supports a wide range of media types and sizes. "
            "The standard paper capacity is 1,200 sheets across two trays, "
            "expandable to 4,700 sheets with optional additional trays. "
            "Supported paper sizes include A3, A4, A5, B4, B5, and custom "
            "sizes ranging from 90 x 148 mm to 305 x 457 mm. The bypass "
            "tray handles envelopes, labels, and heavyweight stock up to "
            "300 gsm. Automatic duplex printing is standard. For high-volume "
            "environments, the optional large-capacity tray holds 2,000 "
            "sheets of A4 paper. Tray settings can be configured via the "
            "Smart Operation Panel or Web Image Monitor for remote management. "
            "Always ensure paper guides are adjusted correctly to prevent "
            "misfeeds and paper jams during operation."
        ),
        (
            "Page 3: Troubleshooting Common Errors. "
            "Error SC542 indicates a fusing unit temperature abnormality. "
            "Power off the device, wait 30 seconds, and restart. If the "
            "error persists, contact Ricoh service. Error SC400 relates to "
            "the transfer belt unit and may require replacement. For paper "
            "jam codes J001 through J009, open the front cover and gently "
            "remove jammed paper in the direction of paper travel. Never "
            "use sharp objects to remove paper as this may damage the drum. "
            "Network connectivity issues (error N001) can be resolved by "
            "verifying Ethernet cable connections and running the built-in "
            "network diagnostic from System Settings > Network > Diagnostics. "
            "For persistent issues, generate a diagnostic report via the "
            "Service Program mode (SP mode) and share it with your Ricoh "
            "service technician for expedited resolution."
        ),
    ]

    for text in sample_texts:
        page = doc.new_page(width=595, height=842)  # A4 dimensions
        # Insert text block with automatic line wrapping
        text_rect = fitz.Rect(50, 50, 545, 792)
        page.insert_textbox(
            text_rect,
            text,
            fontsize=11,
            fontname="helv",
        )

    doc.save(str(path))
    doc.close()
    logger.info("Created sample PDF at '%s' (%d pages).", path, len(sample_texts))


if __name__ == "__main__":
    import json
    import sys

    print("=" * 70)
    print("  RicohLibrary - Phase 1 Ingestion Pipeline Smoke Test")
    print("=" * 70)

    # ── Create a sample PDF in data/ for testing ──
    sample_path = DATA_DIR / "_sample_ricoh_manual.pdf"
    _create_sample_pdf(sample_path)

    # ── Run the full ingestion pipeline ──
    chunks = ingest_all(DATA_DIR)

    if not chunks:
        print("\n❌  No chunks produced - something went wrong.")
        sys.exit(1)

    # ── Summary statistics ──
    print(f"\n✅  Ingestion successful!")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Source files  : {set(c['source_document'] for c in chunks)}")

    page_counts: dict[str, int] = {}
    for c in chunks:
        key = f"{c['source_document']} p.{c['page_number']}"
        page_counts[key] = page_counts.get(key, 0) + 1
    print(f"   Chunks/page   : {page_counts}")

    # ── Print first chunk as a JSON sample ──
    print("\n── Sample chunk (first) ──")
    sample = {k: v for k, v in chunks[0].items()}
    sample["text"] = sample["text"][:200] + "..."  # truncate for readability
    print(json.dumps(sample, indent=2))

    # ── Clean up sample PDF (optional - comment out to keep it) ──
    sample_path.unlink(missing_ok=True)
    print(f"\n🧹  Cleaned up sample PDF.")
    print("=" * 70)
    print("Phase 1 smoke test complete. Ready for Phase 2!")
    print("=" * 70)
