"""Load the input files as the pages the graph reads: an image is one page, a PDF one per page."""
import logging
from base64 import b64encode
from mimetypes import guess_type
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from .imaging import MAX_WIDTH, TARGET_STAFF_SPACING, encode_png, find_staves
from .models import Page

logger = logging.getLogger(__name__)

PDF_MIME_TYPE = "application/pdf"
# PDF sizes are in points, 72 to the inch.
POINTS_PER_INCH = 72
# Vector pages are first rendered at this resolution to measure their staff spacing...
PROBE_DPI = 150
# ...and at this one if no staves are found (the graph then reads the page whole).
FALLBACK_DPI = 200
# A raster image covering at least this much of the page makes it a scan. Smaller ones are logos.
SCAN_MIN_COVERAGE = 0.25


def read_pages(paths: list[str]) -> list[Page]:
    """
    Load sheet-music files as pages, in the order given.

    An image file is one page. Each page of a PDF is rendered to a PNG.

    Raises:
        ValueError: if a file is missing, is neither an image nor a PDF, or is a PDF that can't be read.
    """
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise ValueError(f"No such file: {', '.join(missing)}")
    mime_types = [guess_type(path)[0] or "" for path in paths]
    unsupported = [
        path for path, mime in zip(paths, mime_types)
        if mime != PDF_MIME_TYPE and not mime.startswith("image/")
    ]
    if unsupported:
        raise ValueError(f"Not an image or a PDF: {', '.join(unsupported)}")

    pages: list[Page] = []
    for path, mime in zip(paths, mime_types):
        if mime == PDF_MIME_TYPE:
            pages += _read_pdf(path)
        else:
            image = b64encode(Path(path).read_bytes()).decode("utf-8")
            pages.append({"image": image, "mime_type": mime})
    return pages


def _read_pdf(path: str) -> list[Page]:
    rendered: list[tuple[float, Image.Image]] = []
    try:
        with pdfium.PdfDocument(path) as pdf:
            for page in pdf:
                dpi = _render_dpi(page)
                rendered.append((dpi, _render(page, dpi)))
    except pdfium.PdfiumError as error:
        raise ValueError(f"Couldn't read {path}: {error}") from error
    if not rendered:
        raise ValueError(f"{path} has no pages.")

    dpis = sorted({round(dpi) for dpi, _ in rendered})
    logger.info(
        "Read %d page(s) from %s, rendered at %s dpi.",
        len(rendered), path, dpis[0] if len(dpis) == 1 else f"{dpis[0]}-{dpis[-1]}",
    )
    return [{"image": encode_png(image), "mime_type": "image/png"} for _, image in rendered]


def _render_dpi(page: pdfium.PdfPage) -> float:
    """
    The resolution to render a page at, capped so it is at most MAX_WIDTH px wide.

    A scan keeps its own resolution, as if it were given as an image file: rendering it larger
    would only make up pixels, and the line splitter enlarges lines itself. Vector notation is
    rendered so staff lines land TARGET_STAFF_SPACING px apart, the size the splitter enlarges
    lines to. Much larger would make the lines too thick for find_staves, which looks for thin
    ridges.
    """
    width_limit = MAX_WIDTH * POINTS_PER_INCH / page.get_width()
    scan_dpi = _scan_dpi(page)
    if scan_dpi:
        return min(scan_dpi, width_limit)

    probe_dpi = min(PROBE_DPI, width_limit)
    _, spacing = find_staves(_render(page, probe_dpi))
    wanted = probe_dpi * TARGET_STAFF_SPACING / spacing if spacing else FALLBACK_DPI
    return min(wanted, width_limit)


def _scan_dpi(page: pdfium.PdfPage) -> float | None:
    """The resolution of the scanned image that makes up the page, or None if it isn't a scan."""
    page_area = page.get_width() * page.get_height()
    dpis = []
    for image in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)):
        left, bottom, right, top = image.get_bounds()
        if (right - left) * (top - bottom) >= SCAN_MIN_COVERAGE * page_area:
            metadata = image.get_metadata()
            dpis.append(max(metadata.horizontal_dpi, metadata.vertical_dpi))
    return max((dpi for dpi in dpis if dpi > 0), default=None)


def _render(page: pdfium.PdfPage, dpi: float) -> Image.Image:
    """Render a page in grayscale on white."""
    return page.render(scale=dpi / POINTS_PER_INCH, grayscale=True).to_pil()
