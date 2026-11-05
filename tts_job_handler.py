from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from threading import Event, Thread
from typing import Any

from jobs import Job
from persistent_worker import JobExecution
from tts_service import TTSService


@contextlib.contextmanager
def _keep_lease(
    execution: JobExecution, interval_seconds: float = 30.0
) -> Generator[None, None, None]:
    """Mantém o lease do job vivo em background durante a inferência na CPU."""
    stop_event = Event()

    def _heartbeat_loop():
        while not stop_event.wait(interval_seconds):
            try:
                execution.heartbeat()
            except Exception:
                break

    worker_thread = Thread(target=_heartbeat_loop, daemon=True)
    worker_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)


def make_tts_handler(
    engine: TTSService,
) -> Callable[[Job, JobExecution], dict[str, Any]]:
    """Cria o handler registrado para kind='tts_generate'."""

    def handle(job: Job, execution: JobExecution) -> dict[str, Any]:
        execution.raise_if_cancelled()
        payload = job.input

        # Validação obrigatória de campos
        ref_audio_path = payload.get("ref_audio_path")
        ref_text = payload.get("ref_text")
        output_path = payload.get("output_path")
        text = payload.get("text")
        chunks = payload.get("chunks")
        temperature = float(payload.get("temperature", 0.65))

        if not isinstance(ref_audio_path, str) or not ref_audio_path:
            raise ValueError(
                "Campo obrigatório 'ref_audio_path' inválido ou ausente."
            )
        if not isinstance(ref_text, str) or not ref_text:
            raise ValueError("Campo obrigatório 'ref_text' inválido ou ausente.")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError(
                "Campo obrigatório 'output_path' inválido ou ausente."
            )

        if not chunks and (not isinstance(text, str) or not text):
            raise ValueError(
                "É obrigatório informar 'text' ou uma lista de 'chunks'."
            )

        with _keep_lease(execution):
            if chunks and isinstance(chunks, list):
                audio_gerado = engine.synthesize_chunks(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    chunks=chunks,
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                )
            else:
                audio_gerado = engine.synthesize(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    text_to_speak=str(text),
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                )

        return {"audio_path": audio_gerado}

    return handle