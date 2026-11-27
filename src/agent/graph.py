import logging
import time
from base64 import b64decode
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import config
from .imaging import encode_png, load_page, split_systems
from .llm import get_llm
from .models import AgentState, Line, LineTask
from .models.transcription import PageLayout, SystemTranscription
from .notation import check_system, describe_system, to_score
from .prompts import LAYOUT_PROMPT, RETRY_PROMPT, TRANSCRIBE_PROMPT, VERIFY_PROMPT
from .tools import write_midi

logger = logging.getLogger(__name__)


def elapsed(start: float) -> str:
    """Time since a time.monotonic() reading, e.g. "42s" or "3m05s"."""
    seconds = round(time.monotonic() - start)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def _image(mime_type: str, data: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}}


def _save_debug(name: str, content: str | bytes) -> None:
    if not config.DEBUG_DIR:
        return
    path = Path(config.DEBUG_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


async def read_layout(state: AgentState):
    if config.DEBUG_DIR:
        logger.info("Saving debug files to %s.", config.DEBUG_DIR)
    first = state["pages"][0]
    logger.info(
        "Reading the page layout%s with %s...",
        " from the first page" if len(state["pages"]) > 1 else "", config.COST_SAVING_MODEL,
    )
    start = time.monotonic()
    reader = get_llm(config.COST_SAVING_MODEL).with_structured_output(PageLayout, strict=True)
    layout = await reader.ainvoke([
        SystemMessage(content=LAYOUT_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Report the layout of this page."},
            _image(first["mime_type"], first["image"]),
        ]),
    ])
    _save_debug("layout.json", layout.model_dump_json(indent=2))

    if not layout.is_music_sheet:
        logger.warning("Read the layout in %s: not a music sheet, stopping.", elapsed(start))
        return {"layout": layout, "result": "Not a music sheet."}

    staves = ", ".join(f"{s.name} ({s.clef})" for s in layout.staves)
    meter = layout.time_signature
    logger.info(
        "Read the layout in %s: %s; %s/%s; %g BPM.",
        elapsed(start), staves, meter.numerator, meter.denominator, layout.tempo_bpm,
    )
    return {"layout": layout}


def is_music_sheet(state: AgentState) -> str:
    return "split_pages" if state["layout"].is_music_sheet else END


def split_pages(state: AgentState):
    staves_per_system = len(state["layout"].staves)
    page_count = len(state["pages"])
    lines: list[Line] = []
    for number, page in enumerate(state["pages"]):
        name = "the page" if page_count == 1 else f"page {number + 1}/{page_count}"
        found = split_systems(page["image"], staves_per_system)
        if found is None:
            logger.warning("Couldn't split %s into lines; reading it whole.", name)
            whole = encode_png(load_page(page["image"]))
            lines.append(Line(page=number, index=0, count=1, image=whole, bars=None, whole_page=True))
            continue
        bars = [line.bars if line.bars is not None else "?" for line in found]
        logger.info("Split %s into %d lines; bars counted from barlines: %s.", name, len(found), bars)
        lines += [
            Line(page=number, index=index, count=len(found), image=line.image, bars=line.bars, whole_page=False)
            for index, line in enumerate(found)
        ]
    return {"lines": lines}


def read_each_line(state: AgentState) -> list[Send]:
    lines, page_count = state["lines"], len(state["pages"])
    count = len(lines)
    if all(line.whole_page for line in lines):
        what = "the whole page" if page_count == 1 else f"{page_count} whole pages in parallel"
    else:
        what = "1 line" if count == 1 else f"{count} lines"
        if page_count > 1:
            what += f" from {page_count} pages"
        if count > 1:
            what += " in parallel"
    logger.info(
        "Transcribing %s with %s (%s reasoning, up to %d attempts per line)...",
        what, config.BEST_MODEL, config.REASONING_EFFORT, config.MAX_RETRIES + 1,
    )
    return [
        Send("transcribe_line", LineTask(order=order, line=line, page_count=page_count, layout=state["layout"]))
        for order, line in enumerate(lines)
    ]


def _line_label(task: LineTask) -> str:
    """How log messages name the line: "Line 2/4" on a single page, "Page 3/5, line 2/4" otherwise."""
    line = task["line"]
    page = f"Page {line.page + 1}/{task['page_count']}" if task["page_count"] > 1 else "Page"
    if line.whole_page:
        return page
    where = f"line {line.index + 1}/{line.count}"
    return f"{page}, {where}" if task["page_count"] > 1 else where.capitalize()


def _debug_name(task: LineTask) -> str:
    """Prefix of the line's debug files: "line2" on a single page, "page3_line2" otherwise."""
    line = task["line"]
    name = f"line{line.index + 1}"
    return f"page{line.page + 1}_{name}" if task["page_count"] > 1 else name


def _line_context(task: LineTask) -> str:
    layout, line = task["layout"], task["line"]
    staves = "; ".join(f"{i}. {s.name} ({s.clef} clef)" for i, s in enumerate(layout.staves, start=1))
    meter = layout.time_signature
    page = f"page {line.page + 1} of {task['page_count']}" if task["page_count"] > 1 else "the page"
    if line.whole_page:
        where = (
            f"This image is {'the whole page' if task['page_count'] == 1 else f'all of {page}'}: it could not "
            "be split into lines. Transcribe every bar of every line in reading order as one sequence, "
            "numbering the bars from 1."
        )
    else:
        where = f"This image is line {line.index + 1} of {line.count} on {page}, enlarged."
        if task["order"] == 0:
            where += (
                " It is the first line of the piece: its first bar may be a pickup, and if the time signature "
                f"printed at its start isn't {meter.numerator}/{meter.denominator}, set time_change on bar 1."
            )
    return "\n".join([
        where,
        f"Staves in each line, top to bottom, with the clef each starts the piece in: {staves}.",
        f"Time signature at the start of the piece: {meter.numerator}/{meter.denominator}.",
    ])


def _changed_bars(before: SystemTranscription, after: SystemTranscription) -> list[int] | None:
    """Numbers of the bars that differ, or None if the bar count changed."""
    if len(before.measures) != len(after.measures):
        return None
    return [
        b.number for b, a in zip(before.measures, after.measures)
        if b.model_dump(exclude={"uncertain"}) != a.model_dump(exclude={"uncertain"})
    ]


async def transcribe_line(task: LineTask):
    layout, line_image = task["layout"], task["line"].image
    staff_names = [s.name for s in layout.staves]
    label, debug_name = _line_label(task), _debug_name(task)
    context = _line_context(task)
    image = _image("image/png", line_image)
    reader = get_llm(config.BEST_MODEL).with_structured_output(SystemTranscription, strict=True)
    _save_debug(f"{debug_name}.png", b64decode(line_image))
    started = time.monotonic()

    # Read, check, and re-read with the problems spelled out until the line passes the checks.
    messages = [
        SystemMessage(content=TRANSCRIBE_PROMPT),
        HumanMessage(content=[{"type": "text", "text": f"{context}\n\nTranscribe this line."}, image]),
    ]
    max_attempts = config.MAX_RETRIES + 1
    best, best_errors, best_attempt = None, [], 0
    for attempt in range(1, max_attempts + 1):
        attempt_start = time.monotonic()
        line = await reader.ainvoke(messages)
        # The barline count is never shown up front (the model would anchor on it). A mismatch only
        # asks for a recount: barline detection can be wrong too, so a model that keeps its count wins.
        expected = task["line"].bars if attempt == 1 else None
        errors = check_system(line, layout.time_signature, staff_names, expected)
        _save_debug(f"{debug_name}_attempt{attempt}.json", line.model_dump_json(indent=2))
        if best is None or len(errors) < len(best_errors):
            best, best_errors, best_attempt = line, errors, attempt

        outcome = f"{label}: attempt {attempt}/{max_attempts} took {elapsed(attempt_start)}, {len(line.measures)} bars"
        if not errors:
            logger.info("%s, passed the checks%s.", outcome, "; proofreading" if config.VERIFY else "")
            break
        logger.info(
            "%s, %d problem(s); %s:%s",
            outcome, len(errors), "re-reading" if attempt < max_attempts else "no attempts left",
            "".join(f"\n    - {e}" for e in errors),
        )
        messages += [
            AIMessage(content=line.model_dump_json()),
            HumanMessage(content=RETRY_PROMPT.format(errors="\n".join(f"- {e}" for e in errors))),
        ]

    if best_errors:
        logger.warning(
            "%s: no attempt passed the checks; keeping attempt %d, which has %d problem(s).",
            label, best_attempt, len(best_errors),
        )

    # Rhythmically valid bars can still be made up, so a fresh pass proofreads against the image.
    if config.VERIFY and not best_errors:
        proofread_start = time.monotonic()
        proofread = await reader.ainvoke([
            SystemMessage(content=TRANSCRIBE_PROMPT),
            HumanMessage(content=[
                {"type": "text", "text": f"{context}\n\n" + VERIFY_PROMPT.format(
                    transcription=describe_system(best, staff_names),
                )},
                image,
            ]),
        ])
        _save_debug(f"{debug_name}_proofread.json", proofread.model_dump_json(indent=2))
        changed = _changed_bars(best, proofread)
        took = elapsed(proofread_start)
        if changed == []:
            logger.info("%s: proofreading took %s and found nothing to change.", label, took)
        elif check_system(proofread, layout.time_signature, staff_names):
            logger.info("%s: proofreading took %s; ignored its changes, which failed the checks.", label, took)
        else:
            best = proofread
            logger.info(
                "%s: proofreading took %s and changed %s.",
                label, took, f"bars {changed}" if changed else "the bar count",
            )

    uncertain = [m.number for m in best.measures if m.uncertain]
    logger.info(
        "%s: done in %s, %d bars%s.", label, elapsed(started), len(best.measures),
        f"; bars {uncertain} were hard to read" if uncertain else "",
    )
    return {"transcriptions": [(task["order"], best)]}


def render_midi(state: AgentState):
    # Lines of all pages in reading order, so the pages play one after another as one piece.
    lines = [line for _, line in sorted(state["transcriptions"], key=lambda item: item[0])]
    score, warnings = to_score(state["layout"], lines)
    for warning in warnings:
        logger.warning(warning)
    result = write_midi(score)
    logger.info(result)
    return {"result": result}


def get_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("read_layout", read_layout)
    graph.add_node("split_pages", split_pages)
    graph.add_node("transcribe_line", transcribe_line)
    graph.add_node("render_midi", render_midi)

    graph.add_edge(START, "read_layout")
    graph.add_conditional_edges("read_layout", is_music_sheet, ["split_pages", END])
    graph.add_conditional_edges("split_pages", read_each_line, ["transcribe_line"])
    graph.add_edge("transcribe_line", "render_midi")
    graph.add_edge("render_midi", END)

    return graph
