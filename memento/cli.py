"""The REPL entry point.

See docs/design/03-lld-memento.md §4 (REPL state diagram). Wires AppConfig ->
memlayer.Memory + TranscriptStore + a chat GeminiLLM -> Assistant -> the
default CommandRegistry, then runs the prompt loop.
"""

from __future__ import annotations

import sys

from rich.console import Console

from memento.assistant import Assistant
from memento.commands import CommandContext, build_default_registry
from memento.config import AppConfig, MissingApiKeyError
from memento.transcript import TranscriptStore

WELCOME_MESSAGE = (
    "[bold]Memento[/bold] — your personal AI assistant with memory. Type /help for commands."
)
PROMPT = "[bold cyan]you>[/bold cyan] "
REPLY_PREFIX = "[bold green]memento>[/bold green] "
WRITE_OK_INDICATOR = "[dim]Memory updated ✓[/dim]"
SHUTDOWN_TIMEOUT_SECONDS = 5.0


def main() -> None:
    console = Console()
    try:
        config = AppConfig.load_from_env()
    except MissingApiKeyError as exc:
        console.print(str(exc))
        sys.exit(1)

    from memlayer.llms.gemini import GeminiLLM
    from memlayer.memory import Memory

    memory = Memory.from_config(config.to_memlayer_config())
    transcript = TranscriptStore(db_path=config.transcript_db_path)
    chat_llm = GeminiLLM(model=config.gemini_model, api_key=config.gemini_api_key)
    assistant = Assistant(memory=memory, transcript=transcript, chat_llm=chat_llm)

    context = CommandContext(memory=memory, active_user_id=config.default_user_id)
    registry = build_default_registry()

    console.print(WELCOME_MESSAGE)
    console.print(f"Active user: {context.active_user_id}")

    try:
        run_prompt_loop(console, assistant, registry, context)
    finally:
        assistant.shutdown(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        _drain_write_results(console, assistant)


def run_prompt_loop(
    console: Console, assistant: Assistant, registry, context: CommandContext
) -> None:
    while True:
        try:
            line = console.input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return

        if not line.strip():
            continue

        if line.startswith("/"):
            result = registry.dispatch(line, context)
            if result.render is not None:
                console.print(result.render)
            if result.should_exit:
                return
            continue

        with console.status("Thinking..."):
            reply = assistant.chat(line, context.active_user_id)
        console.print(f"{REPLY_PREFIX}{reply}")
        _drain_write_results(console, assistant)


def _drain_write_results(console: Console, assistant: Assistant) -> None:
    """Print an indicator for every background write that has completed since
    the last check — may lag a turn behind by design (HLD N3), and on exit
    this drains anything still pending after shutdown() has joined the
    writer threads, so the user always sees a final confirmation.
    """
    while (outcome := assistant.poll_write_result()) is not None:
        status, _payload = outcome
        if status == "ok":
            console.print(WRITE_OK_INDICATOR)


if __name__ == "__main__":
    main()
