"""Turn the PDFs in data/ into chunks the retriever can index.

Text is pulled page by page with PyMuPDF and split into overlapping word windows (about 500 words
with 50 overlap). A chunk never crosses a page, so every chunk has one page number to cite. Size is
counted in words rather than tokens, so no tokenizer is needed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
)
from src.vision_ingest import get_cached_description

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Text of every page of one PDF, as dicts with text, page_number (starting at 1) and source_document."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    source_name = pdf_path.name

    pages: list[dict[str, Any]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1
        text = page.get_text("text")
        text = text.strip() if text else ""

        # images and diagrams don't show up in the text layer (see vision_ingest.py). If a
        # description was generated for this page, add it as its own labelled section.
        vision_description = get_cached_description(source_name, page_number)
        if vision_description:
            visual_section = f"\n\n[Embedded image/diagram content on this page]\n{vision_description}"
            text = (text + visual_section).strip() if text else visual_section.strip()

        # nothing to index on this page
        if not text:
            logger.debug(
                "Skipping empty page %d in %s", page_number, source_name
            )
            continue

        pages.append(
            {
                "text": text,
                "page_number": page_number,
                "source_document": source_name,
            }
        )

    doc.close()
    logger.info(
        "Extracted %d non-empty pages from '%s'.", len(pages), source_name
    )
    return pages


def _generate_chunk_id(source: str, page: int | str, index: int) -> str:
    """Chunk id: the first 16 hex characters of sha256 of source, page and index."""
    raw = f"{source}|{page}|{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split pages into overlapping word chunks that keep their page's source_document and page_number.

    Chunks never cross a page boundary, so each one maps to a single page for citations.
    """
    chunks: list[dict[str, Any]] = []
    global_idx = 0  # keeps ids unique across pages

    for page in pages:
        words = page["text"].split()
        source = page["source_document"]
        page_num = page["page_number"]

        # a short page is one chunk
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

        # otherwise slide a window with chunk_overlap words shared
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

            start += chunk_size - chunk_overlap

            # don't leave a tiny last chunk
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


def ingest_all(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    """Extract and chunk every PDF in data_dir."""
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


# quick manual check with a generated sample pdf

def _create_sample_pdf(path: Path) -> None:
    """Make a small three page PDF for trying the pipeline."""
    doc = fitz.open()

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
        page = doc.new_page(width=595, height=842)  # A4
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
    print("  Citera ingestion smoke test")
    print("=" * 70)

    sample_path = DATA_DIR / "_sample_ricoh_manual.pdf"
    _create_sample_pdf(sample_path)

    chunks = ingest_all(DATA_DIR)

    if not chunks:
        print("\nNo chunks produced - something went wrong.")
        sys.exit(1)

    print(f"\nIngestion successful!")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Source files  : {set(c['source_document'] for c in chunks)}")

    page_counts: dict[str, int] = {}
    for c in chunks:
        key = f"{c['source_document']} p.{c['page_number']}"
        page_counts[key] = page_counts.get(key, 0) + 1
    print(f"   Chunks/page   : {page_counts}")

    print("\nSample chunk (first)")
    sample = {k: v for k, v in chunks[0].items()}
    sample["text"] = sample["text"][:200] + "..."
    print(json.dumps(sample, indent=2))

    sample_path.unlink(missing_ok=True)
    print(f"\nCleaned up sample PDF.")
    print("=" * 70)
    print("Smoke test complete.")
    print("=" * 70)
