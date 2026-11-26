import logging
import time
from base64 import b64decode
from pathlib import Path

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import config
from .imaging import LineImage, encode_png, load_page, split_systems
from .llm import get_llm
from .models import AgentState, LineTask
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
    logger.info("Reading the page layout with %s...", config.COST_SAVING_MODEL)
    start = time.monotonic()
    reader = get_llm(config.COST_SAVING_MODEL).with_structured_output(PageLayout, strict=True)
    layout = await reader.ainvoke([
        SystemMessage(content=LAYOUT_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": "Report the layout of this page."},
            _image(state["mime_type"], state["sheet"]),
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
    return "split_page" if state["layout"].is_music_sheet else END


def split_page(state: AgentState):
    lines = split_systems(state["sheet"], len(state["layout"].staves))
    if lines is None:
        logger.warning("Couldn't split the page into lines; reading it whole.")
        page = encode_png(load_page(state["sheet"]))
        return {"lines": [LineImage(page, None)], "whole_page": True}
    bars = [line.bars if line.bars is not None else "?" for line in lines]
    logger.info("Split the page into %d lines; bars counted from barlines: %s.", len(lines), bars)
    return {"lines": lines, "whole_page": False}


def read_each_line(state: AgentState) -> list[Send]:
    count = len(state["lines"])
    logger.info(
        "Transcribing %s with %s (%s reasoning, up to %d attempts per line)...",
        "the whole page" if state["whole_page"] else "1 line" if count == 1 else f"{count} lines in parallel",
        config.BEST_MODEL, config.REASONING_EFFORT, config.MAX_RETRIES + 1,
    )
    return [
        Send("transcribe_line", LineTask(
            index=index,
            image=line.image,
            bars=line.bars,
            layout=state["layout"],
            line_count=count,
            whole_page=state["whole_page"],
        ))
        for index, line in enumerate(state["lines"])
    ]


def _line_context(task: LineTask) -> str:
    layout = task["layout"]
    staves = "; ".join(f"{i}. {s.name} ({s.clef} clef)" for i, s in enumerate(layout.staves, start=1))
    meter = layout.time_signature
    if task["whole_page"]:
        where = (
            "This image is the whole page: it could not be split into lines. Transcribe every bar of "
            "every line in reading order as one sequence, numbering the bars from 1."
        )
    else:
        where = f"This image is line {task['index'] + 1} of {task['line_count']} on the page, enlarged."
        if task["index"] == 0:
            where += (
                " It is the first line: its first bar may be a pickup, and if the time signature printed "
                f"at its start isn't {meter.numerator}/{meter.denominator}, set time_change on bar 1."
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
    layout, index = task["layout"], task["index"]
    staff_names = [s.name for s in layout.staves]
    label = "Page" if task["whole_page"] else f"Line {index + 1}/{task['line_count']}"
    context = _line_context(task)
    image = _image("image/png", task["image"])
    reader = get_llm(config.BEST_MODEL).with_structured_output(SystemTranscription, strict=True)
    _save_debug(f"line{index + 1}.png", b64decode(task["image"]))
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
        expected = task["bars"] if attempt == 1 else None
        errors = check_system(line, layout.time_signature, staff_names, expected)
        _save_debug(f"line{index + 1}_attempt{attempt}.json", line.model_dump_json(indent=2))
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
        _save_debug(f"line{index + 1}_proofread.json", proofread.model_dump_json(indent=2))
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
    return {"transcriptions": [(index, best)]}


def render_midi(state: AgentState):
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
    graph.add_node("split_page", split_page)
    graph.add_node("transcribe_line", transcribe_line)
    graph.add_node("render_midi", render_midi)

    graph.add_edge(START, "read_layout")
    graph.add_conditional_edges("read_layout", is_music_sheet, ["split_page", END])
    graph.add_conditional_edges("split_page", read_each_line, ["transcribe_line"])
    graph.add_edge("transcribe_line", "render_midi")
    graph.add_edge("render_midi", END)

    return graph
