"""Split a sheet-music page into systems (lines) so the model can read each one up close."""
import logging
from base64 import b64decode, b64encode
from io import BytesIO
from typing import NamedTuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# After upscaling, aim for this many pixels between two staff lines.
TARGET_STAFF_SPACING = 16
MAX_SCALE = 4
MAX_WIDTH = 3000
# Narrower than this (in staff spaces), the gap before the first barline only holds a clef and key signature.
MIN_FIRST_BAR_WIDTH = 8


class LineImage(NamedTuple):
    image: str
    """The line, base64 PNG."""
    bars: int | None
    """Bars counted from the barlines, or None if they can't be counted reliably."""


def load_page(sheet: str) -> Image.Image:
    """Decode a base64 image into a grayscale page."""
    return Image.open(BytesIO(b64decode(sheet))).convert("L")


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("utf-8")


def find_staves(page: Image.Image) -> tuple[list[tuple[float, float]], float]:
    """
    Locate every five-line staff on the page.

    Staff lines are thin horizontal ridges: darker than the rows two pixels above and below,
    across most of the page width. Noteheads, text and beams are thicker or shorter, so they
    don't qualify.

    Returns:
        The (top, bottom) rows of each staff, top to bottom, and the spacing between staff lines.
    """
    darkness = 1 - np.asarray(page, dtype=float) / 255
    padded = np.pad(darkness, ((2, 2), (0, 0)))
    ridge = darkness - np.maximum(padded[:-4], padded[4:])
    line_rows = np.where((ridge > 0.12).mean(axis=1) > 0.35)[0]

    lines: list[list[int]] = []
    for row in line_rows:
        if lines and row - lines[-1][-1] <= 1:
            lines[-1].append(row)
        else:
            lines.append([row])
    centers = [sum(line) / len(line) for line in lines]
    if len(centers) < 5:
        return [], 0.0

    gaps = np.diff(centers)
    spacing = float(np.median(gaps[gaps < np.median(gaps) * 1.5]))

    clusters: list[list[float]] = [[centers[0]]]
    for center in centers[1:]:
        if center - clusters[-1][0] <= 4 * spacing * 1.25:
            clusters[-1].append(center)
        else:
            clusters.append([center])

    # A real staff shows at least four of its five lines; shorter clusters are stray ledger lines or beams.
    staves = [(cluster[0], cluster[-1]) for cluster in clusters if len(cluster) >= 4]
    return staves, spacing


def _joined_by_barline(page: np.ndarray, upper: tuple[float, float], lower: tuple[float, float]) -> bool:
    """True if a vertical line crosses the whole gap between two staves (they belong to one system)."""
    gap = page[int(upper[1]) + 2 : int(lower[0]) - 1] > 0.3
    if gap.shape[0] == 0:
        return True
    # A thin line can be smeared over two pixel columns in low-resolution scans.
    columns = gap[:, :-1] | gap[:, 1:]
    return bool(columns.mean(axis=0).max() > 0.9)


def count_bars(page: np.ndarray, staves: list[tuple[float, float]], spacing: float) -> int | None:
    """
    Count the bars of a system from its barlines.

    In a multi-staff system, barlines run unbroken from the top staff to the bottom one; stems and
    beams never do. A single staff gives no such signal (a long stem can span it), so it returns None.
    """
    if len(staves) < 2:
        return None
    rows = page[int(staves[0][0]) : int(staves[-1][1]) + 1] > 0.3
    columns = rows[:, :-1] | rows[:, 1:]
    barline_columns = np.where(columns.mean(axis=0) > 0.95)[0]

    # Thick-thin pairs (repeat signs, final barlines) count as one barline.
    barlines: list[list[int]] = []
    for x in barline_columns:
        if barlines and x - barlines[-1][-1] <= 1.5 * spacing:
            barlines[-1].append(x)
        else:
            barlines.append([x])

    # The system's opening line is a barline too. A start-repeat barline right after the clef and
    # key signature leaves a "bar" too narrow to hold notes: skip it.
    if len(barlines) > 2 and barlines[1][0] - barlines[0][-1] < MIN_FIRST_BAR_WIDTH * spacing:
        barlines.pop(0)
    return len(barlines) - 1 if len(barlines) > 1 else None


def split_systems(sheet: str, staves_per_system: int) -> list[LineImage] | None:
    """
    Crop each system out of the page, upscale it, and count its bars.

    Staves belong to the same system when barlines join them; systems are separated by
    a gap that no barline crosses.

    Args:
        sheet: The page, base64-encoded.
        staves_per_system: Staves the layout pass expects in one system (2 for a piano grand staff).

    Returns:
        One LineImage per system, top to bottom, or None if the page doesn't split into
        systems of that size.
    """
    page = load_page(sheet)
    staves, spacing = find_staves(page)
    if not staves:
        logger.info("Found no staves on the %dx%d px page.", page.width, page.height)
        return None
    logger.info("Found %d staves on the %dx%d px page, %.1f px between staff lines.",
                len(staves), page.width, page.height, spacing)

    darkness = 1 - np.asarray(page, dtype=float) / 255
    groups = [[staves[0]]]
    for staff in staves[1:]:
        if _joined_by_barline(darkness, groups[-1][-1], staff):
            groups[-1].append(staff)
        else:
            groups.append([staff])
    if any(len(group) != staves_per_system for group in groups):
        logger.info(
            "Barlines join the staves into systems of %s staves, but the layout expects %d per system.",
            [len(group) for group in groups], staves_per_system,
        )
        return None

    systems = [(group[0][0], group[-1][1]) for group in groups]
    between = [nxt[0] - cur[1] for cur, nxt in zip(systems, systems[1:])]
    margin = max(float(np.median(between)) / 2, 4 * spacing) if between else 6 * spacing
    scale = min(MAX_SCALE, max(1, round(TARGET_STAFF_SPACING / spacing)), MAX_WIDTH // page.width or 1)
    if scale > 1:
        logger.info("Enlarging each line %dx.", scale)

    crops = []
    for index, (top, bottom) in enumerate(systems):
        upper = (systems[index - 1][1] + top) / 2 if index > 0 else top - margin
        lower = (bottom + systems[index + 1][0]) / 2 if index + 1 < len(systems) else bottom + margin
        crop = page.crop((0, max(0, int(upper)), page.width, min(page.height, int(lower) + 1)))
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        crops.append(LineImage(encode_png(crop), count_bars(darkness, groups[index], spacing)))

    return crops
