#!/bin/sh

set -e

help() {
    echo "./generate-midi.sh [sheet-image]"
    echo "Example: ./generate-midi.sh bach.png"
}

init() {
    # Fall back to virtualenv only if the stdlib venv module fails (e.g. python3-venv not installed).
    if ! python3 -m venv .venv; then
        rm -rf .venv
        virtualenv .venv
    fi
    . .venv/bin/activate
    pip install -r requirements.txt
}

run_agent() {
    if [ -d ".venv" ]; then
        . .venv/bin/activate
    else
        # If .venv doesn't exist, create one
        init
    fi

    # The path goes in as an argument and the heredoc is quoted, so the shell never rewrites the Python code.
    python - "$1" <<'EOF'
import asyncio
import logging
import sys
import time
from base64 import b64encode
from mimetypes import MimeTypes

from src.agent.graph import elapsed, get_graph

logger = logging.getLogger(__name__)

# How often to report that the run is still going, so a model call that takes minutes doesn't look like a hang.
HEARTBEAT_SECONDS = 60


async def main(file: str):
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # Only this app's messages at INFO: libraries like httpx log every request at that level.
    for name in (__name__, "src.agent"):
        logging.getLogger(name).setLevel(logging.INFO)

    graph = get_graph().compile()

    graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

    mime = MimeTypes().guess_type(file)[0]

    if not mime or not mime.startswith("image/"):
        logger.error("%s is not an image!", file)
        return

    with open(file, "rb") as image_file:
        image = b64encode(image_file.read()).decode("utf-8")

    start = time.monotonic()
    lines_total, lines_done = None, 0

    async def heartbeat():
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            progress = f", {lines_done}/{lines_total} lines transcribed" if lines_total else ""
            logger.info("Still working: %s elapsed%s.", elapsed(start), progress)

    ticker = asyncio.create_task(heartbeat())
    try:
        async for update in graph.astream({"sheet": image, "mime_type": mime}):
            for node, values in update.items():
                if node == "split_page":
                    lines_total = len(values["lines"])
                elif node == "transcribe_line":
                    lines_done += 1
                    logger.info("%d/%d lines transcribed, %s elapsed.", lines_done, lines_total, elapsed(start))
    finally:
        ticker.cancel()

    logger.info("Finished in %s.", elapsed(start))

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
EOF
}

case "$1" in
    ""|help|-h|--help) help ;;
    *) run_agent "$1" ;;
esac