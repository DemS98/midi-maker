from typing import Annotated, NamedTuple, NotRequired, TypedDict
import operator

from .transcription import PageLayout, SystemTranscription


class Page(TypedDict):
    image: str
    """The page image, base64."""
    mime_type: str


class Line(NamedTuple):
    """One line (system) cut out of a page."""

    page: int
    """Index of the page it is on."""
    index: int
    """Index of the line on its page, top to bottom."""
    count: int
    """Lines on its page."""
    image: str
    """The line, base64 PNG."""
    bars: int | None
    """Bars counted from the barlines, or None if they can't be counted reliably."""
    whole_page: bool
    """True when the page couldn't be split into lines and this "line" is the whole page."""


class AgentState(TypedDict):
    # The pages of the piece, in order. The layout is read from the first one only.
    pages: list[Page]
    layout: NotRequired[PageLayout]
    # Every line of every page, in reading order.
    lines: NotRequired[list[Line]]
    # (position in `lines`, transcription); lines are read in parallel and finish in any order.
    transcriptions: Annotated[list[tuple[int, SystemTranscription]], operator.add]
    result: NotRequired[str]


class LineTask(TypedDict):
    """What one parallel transcribe_line run receives."""

    order: int
    """Position of the line in reading order across all pages."""
    line: Line
    page_count: int
    layout: PageLayout
