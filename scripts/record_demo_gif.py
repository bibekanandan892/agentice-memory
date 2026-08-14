"""Render docs/media/demo.gif — a simulated interactive `memento` session,
drawn frame-by-frame with Pillow rather than captured from a real terminal
(no terminal-recording tool is assumed to be installed).

Drives the REAL Assistant / CommandRegistry / TranscriptStore / Memory
components directly (the same ones tests/integration/test_cli_session.py
exercises), so the mechanics shown (search -> prompt injection -> reply,
background add(), "Memory updated (check)", user-scoped /memories, /forget)
are all genuine, tested code paths — only the LLM's *text* differs between
modes:

    uv run --extra media python scripts/record_demo_gif.py --live
        Real Gemini + real local embeddings. Needs GEMINI_API_KEY in .env
        and `uv sync --extra local`. Slower, real replies.

    uv run --extra media python scripts/record_demo_gif.py --fake   (default)
        FakeLLM with pre-written, realistic replies + a deterministic
        embedder. No API key or model download needed — this is what
        generated the gif committed to the repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "media" / "demo.gif"

FONT_PATH = Path(r"C:\Windows\Fonts\consola.ttf")
FONT_SIZE = 16
LINE_HEIGHT = 24
CANVAS_WIDTH = 920
TITLE_BAR_HEIGHT = 36
PADDING = 18

BG_COLOR = (24, 24, 24)
TITLE_BAR_COLOR = (45, 45, 45)
DOT_COLORS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
COLOR_USER_PROMPT = (79, 214, 255)
COLOR_ASSISTANT_PROMPT = (137, 209, 133)
COLOR_TEXT = (212, 212, 212)
COLOR_DIM = (110, 110, 110)
COLOR_SYSTEM = (200, 200, 140)

BLOCK_HOLD_MS = 900
USER_INPUT_HOLD_MS = 1400
FINAL_HOLD_MS = 3000


CHECKMARK = "__CHECKMARK__"  # sentinel: render_frame draws a hand-drawn check, not text
MAX_LINE_WIDTH_PX = CANVAS_WIDTH - 2 * PADDING


class Line:
    __slots__ = ("segments",)

    def __init__(self, segments: list[tuple[str, tuple[int, int, int]]]) -> None:
        self.segments = segments


def _oneline(text: str) -> str:
    """Collapse newlines/runs of whitespace — real LLM replies are multiline,
    and PIL's textlength() refuses multiline strings (wrap_line handles the
    re-wrapping to the canvas width itself)."""
    return " ".join(text.split())


def plain(text: str, color=COLOR_TEXT) -> Line:
    return Line([("", COLOR_TEXT), (_oneline(text), color)])


def user_line(text: str) -> Line:
    return Line([("you> ", COLOR_USER_PROMPT), (_oneline(text), COLOR_TEXT)])


def assistant_line(text: str) -> Line:
    return Line([("memento> ", COLOR_ASSISTANT_PROMPT), (_oneline(text), COLOR_TEXT)])


def write_confirmation_line() -> Line:
    return Line([("", COLOR_TEXT), ("Memory updated ", COLOR_DIM), (CHECKMARK, COLOR_DIM)])


def wrap_line(
    line: Line, font: ImageFont.FreeTypeFont, measurer: ImageDraw.ImageDraw
) -> list[Line]:
    """Word-wrap a (prefix, text) Line to MAX_LINE_WIDTH_PX, indenting continuation
    lines under where the text (not the prefix) starts.
    """
    prefix_text, prefix_color = line.segments[0]
    body_text, body_color = line.segments[1]
    indent = " " * len(prefix_text)

    words = body_text.split(" ")
    rows: list[tuple[str, str]] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        lead = prefix_text if not rows and not current else indent
        if measurer.textlength(lead + candidate, font=font) > MAX_LINE_WIDTH_PX and current:
            rows.append((prefix_text if not rows else indent, current))
            current = word
        else:
            current = candidate
    rows.append((prefix_text if not rows else indent, current))

    wrapped = [Line([(lead, prefix_color), (text, body_color)]) for lead, text in rows]
    if len(line.segments) > 2:  # trailing sentinel (e.g. the checkmark) rides the last row
        wrapped[-1].segments.append(line.segments[2])
    return wrapped


def table_lines(title: str, rows: list[tuple[str, str, str]]) -> list[Line]:
    headers = ("id", "memory", "category")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(3)
    ]
    border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header_row = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(3)) + " |"
    lines = [plain(title, COLOR_SYSTEM), plain(border), plain(header_row), plain(border)]
    for row in rows:
        lines.append(plain("| " + " | ".join(row[i].ljust(widths[i]) for i in range(3)) + " |"))
    lines.append(plain(border))
    return lines


def build_script(fake: bool):
    """Run the scripted session and return a list of (lines_snapshot, hold_ms)
    frames, using the real memento components.
    """
    import sys
    import tempfile

    sys.path.insert(0, str(REPO_ROOT))
    from memento.assistant import Assistant
    from memento.commands import CommandContext, build_default_registry
    from memento.transcript import TranscriptStore
    from memlayer.memory import Memory
    from memlayer.storage.history import SQLiteHistoryStore
    from memlayer.vector_stores.local import LocalVectorStore

    data_dir = Path(tempfile.mkdtemp(prefix="memento_gif_"))

    if fake:
        from tests.conftest import FakeEmbedder, FakeLLM

        chat_llm = FakeLLM()
        memory_llm = FakeLLM()
        embedder = FakeEmbedder()
    else:
        import os

        from memlayer.embeddings.sentence_transformer import SentenceTransformerEmbedder
        from memlayer.llms.gemini import DEFAULT_MODEL, GeminiLLM

        # Honor GEMINI_MODEL like the rest of the app (memento/config.py does) —
        # useful when the default alias's model is under heavy demand (503s).
        model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        chat_llm = GeminiLLM(model=model)
        memory_llm = GeminiLLM(model=model)
        embedder = SentenceTransformerEmbedder()

    memory = Memory(
        llm=memory_llm,
        embedder=embedder,
        vector_store=LocalVectorStore(db_path=data_dir / "v.db"),
        history_store=SQLiteHistoryStore(db_path=data_dir / "h.db"),
    )
    transcript = TranscriptStore(db_path=data_dir / "t.db")
    assistant = Assistant(memory=memory, transcript=transcript, chat_llm=chat_llm)
    context = CommandContext(memory=memory, active_user_id="bibek")
    registry = build_default_registry()

    frames: list[tuple[list[Line], int]] = []
    accumulated: list[Line] = []

    def push(line: Line, hold_ms: int = BLOCK_HOLD_MS) -> None:
        accumulated.append(line)
        frames.append((list(accumulated), hold_ms))

    if fake:
        chat_llm.queue(
            "Nice to meet you, Bibek! Good luck with the AIML course "
            "\u2014 and filter coffee is a great choice."
        )
        memory_llm.queue(
            json.dumps(
                {
                    "facts": [
                        {"text": "Name is Bibek", "category": "semantic"},
                        {"text": "Is doing an AIML course", "category": "semantic"},
                        {"text": "Likes filter coffee", "category": "semantic"},
                    ]
                }
            )
        )
        memory_llm.queue(
            json.dumps(
                {
                    "memory": [
                        {"text": "Name is Bibek", "event": "ADD"},
                        {"text": "Is doing an AIML course", "event": "ADD"},
                        {"text": "Likes filter coffee", "event": "ADD"},
                    ]
                }
            )
        )

    turn1_text = "Hi, I'm Bibek. I'm taking an AIML course and I love filter coffee."
    push(user_line(turn1_text), USER_INPUT_HOLD_MS)
    reply = assistant.chat(turn1_text, "bibek")
    push(assistant_line(reply))
    # Deterministic for the recording; real usage is async. Generous timeout:
    # in --live mode the background add() makes two real LLM calls, and a
    # free-tier rate-limit retry can add tens of seconds on top.
    assistant.shutdown(timeout=90.0)
    outcome = assistant.poll_write_result()
    if outcome and outcome[0] == "ok":
        push(write_confirmation_line())
    else:
        # A failed/missing write means the rest of the recording (memories
        # table, /forget) would be empty or wrong — fail loudly instead of
        # producing a broken gif.
        raise RuntimeError(f"Background memory write did not succeed: {outcome!r}")

    push(user_line("/memories"), USER_INPUT_HOLD_MS)
    rows = [
        (row["id"][:8], row["memory"], row.get("memory_category", ""))
        for row in memory.get_all(user_id="bibek")["results"]
    ]
    for line in table_lines(f"Memories for {context.active_user_id}", rows):
        push(line)
    # Pick the coffee memory to /forget when we can find it; fall back to the
    # last row — in --live mode the real extractor decides how many facts to
    # store and in what order, so a fixed index is not safe.
    results = memory.get_all(user_id="bibek")["results"]
    forget_target = next(
        (row for row in results if "coffee" in row["memory"].lower()), results[-1]
    )
    forget_id = forget_target["id"]

    push(user_line("/user alice"), USER_INPUT_HOLD_MS)
    result = registry.dispatch("/user alice", context)
    push(plain(str(result.render), COLOR_SYSTEM))

    push(user_line("/memories"), USER_INPUT_HOLD_MS)
    result = registry.dispatch("/memories", context)
    push(plain(str(result.render), COLOR_SYSTEM))

    push(user_line("/user bibek"), USER_INPUT_HOLD_MS)
    result = registry.dispatch("/user bibek", context)
    push(plain(str(result.render), COLOR_SYSTEM))

    push(user_line(f"/forget {forget_id[:8]}"), USER_INPUT_HOLD_MS)
    result = registry.dispatch(f"/forget {forget_id[:8]}", context)
    push(plain(str(result.render), COLOR_SYSTEM))

    push(user_line("/exit"), USER_INPUT_HOLD_MS)
    result = registry.dispatch("/exit", context)
    push(plain(str(result.render), COLOR_SYSTEM), FINAL_HOLD_MS)

    return frames


def draw_checkmark(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Hand-drawn check mark — Consolas has no glyph for U+2713, so this
    avoids relying on font glyph coverage entirely."""
    baseline = y + LINE_HEIGHT * 0.6
    draw.line([(x, baseline), (x + 5, baseline + 5), (x + 13, baseline - 8)], fill=color, width=2)


def render_frame(
    lines: list[Line], font: ImageFont.FreeTypeFont, measurer: ImageDraw.ImageDraw
) -> Image.Image:
    wrapped_lines = [wrapped for line in lines for wrapped in wrap_line(line, font, measurer)]
    height = TITLE_BAR_HEIGHT + 2 * PADDING + len(wrapped_lines) * LINE_HEIGHT
    image = Image.new("RGB", (CANVAS_WIDTH, max(height, TITLE_BAR_HEIGHT + 80)), BG_COLOR)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, CANVAS_WIDTH, TITLE_BAR_HEIGHT], fill=TITLE_BAR_COLOR)
    for i, color in enumerate(DOT_COLORS):
        cx = 20 + i * 20
        draw.ellipse([cx - 6, 12, cx + 6, 24], fill=color)
    label = "memento"
    label_width = draw.textlength(label, font=font)
    draw.text(((CANVAS_WIDTH - label_width) / 2, 9), label, font=font, fill=(180, 180, 180))

    y = TITLE_BAR_HEIGHT + PADDING
    for line in wrapped_lines:
        x = PADDING
        for text, color in line.segments:
            if text == CHECKMARK:
                draw_checkmark(draw, x + 5, y, color)
                x += 24
                continue
            draw.text((x, y), text, font=font, fill=color)
            x += draw.textlength(text, font=font)
        y += LINE_HEIGHT

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Use real Gemini + real embeddings."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to write the gif (default: docs/media/demo.gif).",
    )
    args = parser.parse_args()

    if args.live:
        from dotenv import load_dotenv

        load_dotenv()

    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    measurer = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    frames = build_script(fake=not args.live)

    images = [render_frame(lines, font, measurer) for lines, _hold in frames]
    max_height = max(img.height for img in images)
    padded = []
    for img in images:
        if img.height < max_height:
            canvas = Image.new("RGB", (CANVAS_WIDTH, max_height), BG_COLOR)
            canvas.paste(img, (0, 0))
            padded.append(canvas)
        else:
            padded.append(img)

    durations = [hold for _lines, hold in frames]
    durations[-1] = FINAL_HOLD_MS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    padded[0].save(
        args.output,
        save_all=True,
        append_images=padded[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {args.output} ({len(padded)} frames, {args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
