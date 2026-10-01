"""Describe the screenshots and diagrams in the PDFs, so retrieval can find them.

Plain text extraction misses what an image shows. A scan of the 733 documents found 116 pages with
any image, but 77% of those images were tiny icons under 50px, which gave useless descriptions. Only
images with a long edge of at least IMAGE_SIZE_THRESHOLD_PX are used, which leaves 31 pages in 28
documents. Each page is rendered and a vision model describes it the way a technician would
(dialog titles, labels, table values, menu paths). The description is added to the page text before
chunking, and cached by (document, page) so a rerun doesn't pay for it again.
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

# kept in eval/, not data/, because data/ is gitignored (the PDFs are Ricoh's) and this small
# cache should be committed
CACHE_PATH: Path = PROJECT_ROOT / "eval" / "vision_cache.json"
VISION_MODEL: str = "claude-sonnet-5"

# images with a longer edge under this are treated as decorative. Every image under 50px was an
# icon when checked by hand, and the smallest real screenshot was about 350px
IMAGE_SIZE_THRESHOLD_PX: int = 150

# render at 2x (about 144 DPI), sharp enough for small labels and under the long-edge cap
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


# finding pages with real images (local, no API calls)

def find_image_pages(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Pages with at least one image over IMAGE_SIZE_THRESHOLD_PX (31 pages in 28 documents right now)."""
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


# cache

def _load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_key(source_document: str, page_number: int) -> str:
    return f"{source_document}|{page_number}"


# render and describe

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
    """Ask the vision model what the images on this page show, or None if it says there's nothing there."""
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


# running it over everything

def describe_all(
    data_dir: Path = DATA_DIR, limit: int | None = None
) -> dict[str, Any]:
    """Describe every candidate page that isn't cached yet.

    limit stops after that many new pages, for a pilot run. Returns counts of described,
    no_visual_content and skipped_cached pages, plus the list of failures.
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

        # save as we go so a crash doesn't lose pages already paid for
        _save_cache(cache)

    return stats


# lookup used by ingest.py. The cache is loaded once per process, since a full ingest visits
# about 1000 pages and only a few dozen have a hit
_in_memory_cache: dict[str, str] | None = None


def get_cached_description(source_document: str, page_number: int) -> str | None:
    """The cached description for a page, or None if it has none."""
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
