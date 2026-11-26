# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agent that turns a photo or scan of printed sheet music into a MIDI file. It is a LangGraph pipeline (`src/agent/`) that uses OpenAI vision models through `langchain_openai`. It writes MIDI with `python-midi/`, a copy of the `python-midi` library (imported as `midi`) that is kept in this repo.

## Commands

Run everything from the repo root. The code imports `src.agent...`, and output paths are relative to the current directory.

```sh
./generate-midi.sh path/to/sheet.png   # full run; creates .venv and installs requirements.txt the first time
```

- A run writes `output.mid` (`tools.OUTPUT_PATH`) and redraws `graph.png` (a Mermaid render of the graph, gitignored) in the current directory. A full run takes several minutes at `high` reasoning effort.
- It needs `OPENAI_API_KEY` in `.env`; `.env.example` is the template. The Claude settings block reads of `.env`, so leave that file alone.
- The repo contains no sample sheet images. Bring your own image to test with.
- The settings in `src/agent/config.py` can be overridden with environment variables or `.env`:
  - `MIDI_AGENT_MODEL` / `MIDI_AGENT_MODEL_CHEAP`: model names
  - `MIDI_AGENT_REASONING_EFFORT`: reasoning effort
  - `MIDI_AGENT_MAX_RETRIES`: re-reads per line
  - `MIDI_AGENT_VERIFY=1`: turns on the proofreading pass
  - `MIDI_AGENT_DEBUG_DIR=<dir>`: saves `layout.json`, each line's PNG and every attempt's JSON there. Use this for debugging accuracy.
- The image steps run without any model calls, so you can check them quickly and for free:
  ```sh
  .venv/bin/python -c "from base64 import b64encode; from src.agent.imaging import split_systems; \
  print([l.bars for l in split_systems(b64encode(open('sheet.png','rb').read()).decode(), 2)])"
  ```
  The `2` is the number of staves per system (2 for a piano grand staff). The output is the bar count of each line, with `None` where the bars couldn't be counted.
  `notation.check_system` / `notation.to_score` are pure functions over the pydantic models, so you can test them the same way.
- The main app has no test suite and no linter configured.

### python-midi

- `requirements.txt` installs `./python-midi` as a regular (non-editable) package. After editing `python-midi/src/midi/`, reinstall it with `.venv/bin/pip install ./python-midi`. Until then, the agent keeps using the old installed copy.
- Ignore `python-midi/build/`. It holds stale build output.
- Tests use pytest, which is not in `.venv`. Install it, then:
  - run all tests: `cd python-midi && ../.venv/bin/python -m pytest`
  - run one test: `../.venv/bin/python -m pytest tests/test_events.py::<test_name>`

## Architecture

### Graph (`src/agent/graph.py`)

1. **`read_layout`**: one call to the cheap model with the whole page. It returns `PageLayout`: whether the image is music, tempo, starting meter, and the staves of one system with their clefs and GM programs. If the image is not music, the graph ends here.
2. **`split_page`**: deterministic, in `imaging.split_systems`, with no model involved.
   - It finds staff lines as thin horizontal ridges in the pixels.
   - It groups staves into systems by checking for barlines that span the gap between them. The group size must match `len(layout.staves)`.
   - It upscales each system to about 16 px staff spacing.
   - It counts bars from barlines that run the full height of the system. This returns `None` for single-staff systems, where a long stem can't be told apart from a barline.
   - If splitting fails, the whole page is sent as one "line" (`whole_page=True`).
3. **`transcribe_line`**: one parallel run per line, started with `Send`, using the best model.
   - The model returns a `SystemTranscription`.
   - `notation.check_system` checks it: bar numbering, staves per bar, and voice durations against the meter.
   - Any errors are sent back with `RETRY_PROMPT`, up to `MAX_RETRIES` times, and the attempt with the fewest errors is kept.
   - The bar count from the barlines is only checked on attempt 1 and is never put in the prompt, because the model would anchor on it. A mismatch only asks the model to recount, since barline detection can also be wrong.
4. **`render_midi`**: sorts the transcriptions by line index, runs `notation.to_score`, then `tools.write_midi`.

`AgentState.transcriptions` uses an `operator.add` reducer because parallel lines finish in any order.

### Core design rule: the model reads, code computes

`models/transcription.py` describes notation **exactly as printed**: letter and octave of the staff position, only the accidentals printed in front of a note, written note values, ties, tuplets, repeat and volta marks. `notation.py` does all the music arithmetic:

- key signatures, and accidentals carrying through the bar (per staff, across voices)
- ties
- tuplet ratios
- pickup bars
- repeat and ending expansion (`play_order`)
- dynamics → velocity
- absolute ticks at `music.PPQ = 480`

Don't ask the model for MIDI numbers or ticks. An earlier single-shot design did that and was much less accurate.

### Constraints on the transcription schemas

- They are used with `with_structured_output(..., strict=True)`, so every field must be required and fields can't use range constraints like `ge`/`le`. Express ranges as `Literal`s, or check them in code.
- Field `description`s are effectively part of the prompt, alongside `prompts.py`.

### Output models

- `models/music.py` (`Score`/`Track`/`Note`, absolute ticks) is the resolved form that `write_midi` consumes. Code builds it; the model never fills it in.
- `to_score` keeps at most 15 staves and skips MIDI channel 9, which is reserved for percussion.
- `write_midi` merges overlapping notes of the same pitch on one channel, so one note's note-off doesn't cut the others short.

### LLM and inputs

- `llm.get_llm` is cached per model name. Don't pass `temperature`, because the GPT-5-family reasoning models reject anything other than the default.
- Accuracy has mostly been tested on low-resolution previews, about 536 px wide with about 4.5 px staff spacing. On those, reading pitches, not the bar structure, is what limits accuracy.
