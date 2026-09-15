"""
src/vision_ingest.py - Vision-based description of image/diagram pages.

extract_pages() in ingest.py does plain text extraction, so a screenshot, UI
dialog, or table embedded in a page is invisible to retrieval: whatever
PyMuPDF's text layer happened to capture around it, nothing of what the image
itself shows. A local scan of the 733-document corpus first flagged 116 pages
with *any* embedded image object, but 77% of those images (146/190) turned
out to be decorative icons under 50px, tiny bullet/toggle/note glyphs a
5-page pilot confirmed produce useless descriptions ("a green toggle switch
icon"). Filtering to images with a long edge >= IMAGE_SIZE_THRESHOLD_PX
narrows this to 31 pages across 28 documents, all screenshots, diagrams, or
tables large enough to plausibly carry information the text layer missed.

This module renders each such page to an image and asks a vision-capable
model to describe what it shows, in the terms a technician reading the
screenshot would use (dialog titles, field/button labels, table values, menu
paths, diagram steps), then that description gets appended to the page's
extracted text before chunking. Results are cached to disk by
(document, page) so re-running ingestion never re-pays for a page already
described, the same discipline finetune/scripts/generate_training_data.py
uses for its own cache/manifest.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.config import DATA_DIR, PROJECT_ROOT
from src.llm_factory import get_llm, response_text

logger = logging.getLogger(__name__)

# Not under data/: that whole directory is gitignored (the 733 source PDFs
# are ~223 MB and are RICOH's documentation, not ours to republish - see
# src/build_demo_index.py), which would silently drop this small, valuable
# cache too. eval/ is where this project already commits small generated
# artifacts needed to reproduce a run (ground_truth.json, generated
# questions), so the vision cache lives there rather than being lost.
CACHE_PATH: Path = PROJECT_ROOT / "eval" / "vision_cache.json"
VISION_MODEL: str = "claude-sonnet-5"

# Long-edge pixel threshold below which an embedded image is treated as
# decorative (bullet points, toggle icons, note-box glyphs) rather than
# content-bearing. Measured, not guessed: every image under 50px in the
# corpus was decorative in a manual check; 150px gives headroom above that
# without pulling the small-but-real diagrams (the smallest genuine
# screenshot found was ~350px) back out.
IMAGE_SIZE_THRESHOLD_PX: int = 150

# Render at 2x (~144 DPI). Sharp enough to read small UI labels; a typical
# letter/A4 page at this zoom stays well under the high-resolution tier's
# 2576px long-edge cap, so it costs one predictable token count rather than
# being downscaled unpredictably.
RENDER_ZOOM: float = 2.0

NO_VISUAL_CONTENT_MARKER = "NO_VISUAL_CONTENT"

_DESCRIBE_PROMPT = (
    "This is a page from a Ricoh product/workflow manual. Plain-text "
    "extraction from this PDF already pulled out the prose; what it cannot "
    "see is what any embedded screenshot, dialog, diagram, or table on this "
    "page actually shows.\n\n"
    "Describe ONLY the visual content: window/dialog titles, field and "
    "button labels, menu paths, table contents (row/column values), and "
    "diagram steps or connections, in the order a technician reading the "
    f"page would encounter them. Do not repeat prose paragraphs the text "
    f"layer already captures, and do not editorialize. If the page has no "
    f"meaningful embedded image content beyond decorative elements (a logo, "
    f"a rule line), respond with exactly: {NO_VISUAL_CONTENT_MARKER}"
)


# 1. FIND CANDIDATE PAGES  (free, local, no API calls)

def find_image_pages(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Every page containing at least one embedded image whose long edge is
    >= IMAGE_SIZE_THRESHOLD_PX, i.e. plausibly a screenshot, diagram, or
    table rather than a decorative icon. 31 pages across 28 documents on the
    current corpus.
    """
    pages: list[dict[str, Any]] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        for page_idx, page in enumerate(doc):
            has_real_image = False
            for img in page.get_images(full=True):
                try:
                    base = doc.extract_image(img[0])
                except Exception as e:  # noqa: BLE001 - a handful of malformed
                    # embedded image streams should not abort the whole scan.
                    logger.debug(
                        "Could not extract image xref %s on %s p.%d: %s",
                        img[0], pdf_path.name, page_idx + 1, e,
                    )
                    continue
                if max(base["width"], base["height"]) >= IMAGE_SIZE_THRESHOLD_PX:
                    has_real_image = True
                    break
            if has_real_image:
                pages.append(
                    {"source_document": pdf_path.name, "page_number": page_idx + 1}
                )
        doc.close()
    return pages


# 2. CACHE  (avoid re-paying for a page already described)

def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_key(source_document: str, page_number: int) -> str:
    return f"{source_document}|{page_number}"


# 3. RENDER + DESCRIBE

def _render_page_png(pdf_path: Path, page_number: int, zoom: float = RENDER_ZOOM) -> bytes:
    """Render one 1-indexed page to PNG bytes."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def describe_page_image(
    pdf_path: Path, page_number: int, llm=None
) -> str | None:
    """Ask the vision model what the embedded image content on this page
    shows. Returns None for a page with no meaningful visual content (the
    model said so explicitly), never an empty string masquerading as one.
    """
    llm = llm or get_llm(provider="anthropic", model=VISION_MODEL, max_tokens=1024)

    png_bytes = _render_page_png(pdf_path, page_number)
    b64_data = base64.standard_b64encode(png_bytes).decode("utf-8")

    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": _DESCRIBE_PROMPT},
            {
                "type": "image",
                "source_type": "base64",
                "data": b64_data,
                "mime_type": "image/png",
            },
        ]
    )
    description = response_text(llm.invoke([message]))

    if NO_VISUAL_CONTENT_MARKER in description:
        return None
    return description


# 4. ORCHESTRATOR

def describe_all(
    data_dir: Path = DATA_DIR, limit: int | None = None
) -> dict[str, Any]:
    """Describe every candidate page not already in the cache.

    Args:
        limit: process at most this many NEW (uncached) pages. Used for the
               pilot run before committing to the full 116-page batch.

    Returns:
        {"described": int, "no_visual_content": int, "skipped_cached": int,
         "failed": list[str]}
    """
    pages = find_image_pages(data_dir)
    cache = _load_cache()
    llm = get_llm(provider="anthropic", model=VISION_MODEL, max_tokens=1024)

    stats = {"described": 0, "no_visual_content": 0, "skipped_cached": 0, "failed": []}
    processed = 0

    for p in pages:
        key = _cache_key(p["source_document"], p["page_number"])
        if key in cache:
            stats["skipped_cached"] += 1
            continue
        if limit is not None and processed >= limit:
            break

        try:
            description = describe_page_image(
                data_dir / p["source_document"], p["page_number"], llm=llm
            )
        except Exception as e:  # noqa: BLE001 - log and continue, one bad page
            # should not abort a 116-page batch.
            logger.warning(
                "Vision description failed for %s p.%d: %s",
                p["source_document"], p["page_number"], e,
            )
            stats["failed"].append(key)
            processed += 1
            continue

        cache[key] = description if description is not None else NO_VISUAL_CONTENT_MARKER
        if description is None:
            stats["no_visual_content"] += 1
        else:
            stats["described"] += 1
        processed += 1

        # Save incrementally so a crash mid-batch (as happened during the
        # finetune data-gen run) does not lose already-paid-for work.
        _save_cache(cache)

    return stats


# 5. INGEST-TIME LOOKUP  (consumed by ingest.py)
#
# A full ingest visits ~1000 pages; re-reading and re-parsing the cache file
# on every single page for what is at most a few dozen hits is wasteful, so
# it is loaded once per process and reused. The cache is written offline by
# describe_all(), never concurrently with an ingest run, so there is no
# staleness a process restart doesn't already resolve.
_in_memory_cache: dict[str, str] | None = None


def get_cached_description(source_document: str, page_number: int) -> str | None:
    """Return the cached vision description for a page, or None if there is
    none (not a candidate page, not yet processed, or genuinely no visual
    content)."""
    global _in_memory_cache
    if _in_memory_cache is None:
        _in_memory_cache = _load_cache()
    value = _in_memory_cache.get(_cache_key(source_document, page_number))
    if value is None or value == NO_VISUAL_CONTENT_MARKER:
        return None
    return value


if __name__ == "__main__":
    import sys

    pilot_limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Running vision description pilot: up to {pilot_limit} new page(s)...")
    result = describe_all(limit=pilot_limit)
    print(json.dumps(result, indent=2))
