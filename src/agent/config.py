"""Agent settings. Each can be overridden with an environment variable or in .env."""
import os

import dotenv


dotenv.load_dotenv()

COST_SAVING_MODEL = os.getenv("MIDI_AGENT_MODEL_CHEAP", "gpt-5-nano")
BEST_MODEL = os.getenv("MIDI_AGENT_MODEL", "gpt-6.1-terra")
# Reading notation is perception-heavy: more reasoning means fewer misread notes, at more latency and cost.
REASONING_EFFORT = os.getenv("MIDI_AGENT_REASONING_EFFORT", "high")
# How many times a line is re-read when the checks in notation.check_system find problems.
MAX_RETRIES = int(os.getenv("MIDI_AGENT_MAX_RETRIES", "2"))
# Whether a second pass proofreads each line against its image, at one extra call per line. Off by
# default: on bach.png it changed nothing on any of the 8 lines it checked across two runs.
VERIFY = os.getenv("MIDI_AGENT_VERIFY", "0") == "1"
# If set, every run saves the line images and each transcription attempt in this directory.
DEBUG_DIR = os.getenv("MIDI_AGENT_DEBUG_DIR")
