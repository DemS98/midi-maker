from typing import Annotated, NotRequired, TypedDict
import operator

from ..imaging import LineImage
from .transcription import PageLayout, SystemTranscription


class AgentState(TypedDict):
    sheet: str
    mime_type: str
    layout: NotRequired[PageLayout]
    # Each line (system) of the page, top to bottom.
    lines: NotRequired[list[LineImage]]
    # True when the page couldn't be split into lines and is read in one go.
    whole_page: NotRequired[bool]
    # (line index, transcription); lines are read in parallel and finish in any order.
    transcriptions: Annotated[list[tuple[int, SystemTranscription]], operator.add]
    result: NotRequired[str]


class LineTask(TypedDict):
    """What one parallel transcribe_line run receives."""

    index: int
    image: str
    # Bars counted from the barlines, if they could be counted.
    bars: int | None
    layout: PageLayout
    line_count: int
    whole_page: bool
