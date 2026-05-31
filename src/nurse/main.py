"""Entry point — starts the nurse voice assistant."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()
app = typer.Typer(add_completion=False)

# Libraries whose DEBUG output is pure noise (HTTP wire logs, model fetch chatter).
# Pinned to WARNING on every handler so even the file log stays readable.
_NOISY_LIBS = (
    "faster_whisper", "urllib3", "httpx", "httpcore", "sentence_transformers",
    "huggingface_hub", "asyncio", "filelock", "matplotlib",
)


def _setup_logging(verbose: bool) -> Path:
    """Configure console + file logging.

    Console honors --verbose (INFO/DEBUG). A timestamped file under logs/ always
    captures full DEBUG (minus the noisy libraries) so a run can be inspected after
    the fact without re-running with shell redirection. Returns the log file path.
    """
    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Single fixed log file, truncated on every start, so the latest run is always
    # at the same path and easy to inspect.
    log_path = log_dir / "nurse.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s — %(message)s", datefmt="%H:%M:%S"
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter; root must pass everything
    # Clear any handlers basicConfig/libraries may have installed so our two win.
    for h in list(root.handlers):
        root.removeHandler(h)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    for lib in _NOISY_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return log_path


@app.command()
def run(
    patient: str = typer.Option("default", "--patient", "-p", help="Patient ID to load"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG retrieval"),
    no_proactive: bool = typer.Option(False, "--no-proactive", help="Disable proactive engagement"),
) -> None:
    """Start the Nurse Brain voice assistant."""
    log_path = _setup_logging(verbose)

    console.print(Panel.fit(
        "[bold cyan]Nurse Brain[/bold cyan]\n"
        f"Patient: [green]{patient}[/green]  |  RAG: {'off' if no_rag else 'on'}"
        f"  |  Proactive: {'off' if no_proactive else 'on'}\n"
        f"[dim]Log: {log_path}[/dim]\n"
        "[dim]Ctrl+C to end session[/dim]",
        border_style="cyan",
    ))

    # Lazy imports so startup is fast
    from nurse.pipeline import NursePipeline
    from nurse.audio.input import MicrophoneStream
    from nurse.rag.index import build_index
    from nurse.proactive.scheduler import ProactiveScheduler

    if not no_rag:
        try:
            build_index()
        except Exception as e:
            console.print(f"[yellow]RAG index build skipped: {e}[/yellow]")

    pipeline = NursePipeline(patient_id=patient)

    async def _main():
        loop = asyncio.get_running_loop()
        mic = MicrophoneStream()
        pipeline.set_mic(mic)
        # Barge-in: let the patient talk over Aria (active only if barge_in.enabled and an
        # AEC backend is present; otherwise a harmless no-op — mic stays muted as before).
        mic.set_barge_in_callback(pipeline.request_barge_in)
        # Prime the LLM (load weights + pre-fill the cached prompt prefix) before
        # greeting, so the patient's first utterance doesn't eat the cold-start.
        pipeline.llm.warmup()
        pipeline.greet()
        console.print("[dim]Listening…[/dim]")

        # Proactive scheduler runs concurrently. The pipeline's busy-lock keeps it
        # mutually exclusive with reactive turns; engagements run in the executor so
        # the event loop (and the scheduler's sleep) is never blocked.
        scheduler_task = None
        if not no_proactive:
            scheduler = ProactiveScheduler(pipeline)
            scheduler_task = asyncio.create_task(scheduler.run())

        try:
            async for audio in mic:
                # Run the blocking turn off the event loop so the scheduler keeps ticking.
                await loop.run_in_executor(None, pipeline.process_audio, audio)
                console.print("[dim]Listening…[/dim]")
        except asyncio.CancelledError:
            pass
        finally:
            if scheduler_task:
                scheduler_task.cancel()

    loop = asyncio.new_event_loop()

    def _shutdown(sig, frame):
        console.print("\n[yellow]Ending session…[/yellow]")
        pipeline.end_session()
        loop.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(_main())
    finally:
        pipeline.end_session()
        loop.close()


@app.command()
def devices() -> None:
    """List available audio input/output devices."""
    import sounddevice as sd
    console.print(sd.query_devices())


@app.command()
def bench(
    patient: str = typer.Option("default", "--patient", "-p"),
    text: str = typer.Option("I have a headache and my temperature is 99.2 degrees.", "--text", "-t"),
) -> None:
    """Benchmark end-to-end latency without microphone input."""
    import time
    import numpy as np
    from nurse.pipeline import NursePipeline

    _setup_logging(False)
    console.print(f"[bold]Benchmarking with:[/bold] {text}")

    pipeline = NursePipeline(patient_id=patient)

    # Synthesize fake audio to warm up ASR (skip actual transcription for bench)
    console.print("Warming up models…")
    from nurse.asr.whisper_stream import _load_model as _warmup_asr
    from nurse.tts.kokoro_stream import _load_pipeline as _warmup_tts
    _warmup_asr()
    _warmup_tts()

    console.print("Running timed turn…")
    from nurse.llm.prompt import build_messages
    from nurse.llm.client import LLMClient
    from nurse.llm.tools import ToolDispatcher
    from nurse.rag.retrieve import retrieve
    from nurse.memory.longterm import LongTermMemory
    from nurse.safety.filter import check_and_escalate

    long_term = LongTermMemory(patient)
    dispatcher = ToolDispatcher(patient)
    llm = LLMClient(dispatcher)
    rag_context = retrieve(text)
    messages = build_messages([], text, patient_summary=long_term.load_for_prompt(), rag_context=rag_context)
    _, messages = check_and_escalate(text, messages)

    t0 = time.perf_counter()
    tokens = list(llm.stream_response(messages))
    t_llm = time.perf_counter()
    response = "".join(tokens)

    from nurse.tts.kokoro_stream import synthesize
    audio = synthesize(response)
    t_tts = time.perf_counter()

    console.print(f"\n[bold]Response:[/bold] {response}")
    console.print(f"[bold]LLM:[/bold] {(t_llm - t0)*1000:.0f}ms")
    console.print(f"[bold]TTS:[/bold] {(t_tts - t_llm)*1000:.0f}ms")
    console.print(f"[bold]Total (no ASR):[/bold] {(t_tts - t0)*1000:.0f}ms")


if __name__ == "__main__":
    app()
