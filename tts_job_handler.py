from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Generator
from threading import Event, Thread
from typing import Any

from jobs import Job
from persistent_worker import JobExecution
from tts_service import TTSService

logger = logging.getLogger("tts_job_handler")


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
                logger.exception(
                    "Falha ao renovar heartbeat do job %s. "
                    "A inferência pode continuar, mas o lease pode expirar.",
                    execution._job.id,
                )
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
        logger.info(
            "HANDLER_START job_id=%s kind=%s tentativa=%s",
            job.id, job.kind, job.attempt
        )
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

        logger.info(
            "HANDLER_VALIDATED job_id=%s ref_audio=%s output=%s mode=%s "
            "chunks=%s temperature=%s",
            job.id, ref_audio_path, output_path,
            "chunks" if chunks and isinstance(chunks, list) else "direct",
            len(chunks) if isinstance(chunks, list) else 0,
            temperature,
        )

        with _keep_lease(execution):
            if chunks and isinstance(chunks, list):
                logger.info(
                    "SYNTHESIS_CHUNKS_BEGIN job_id=%s total_chunks=%s",
                    job.id, len(chunks)
                )
                audio_gerado = engine.synthesize_chunks(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    chunks=chunks,
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                )
            else:
                logger.info("SYNTHESIS_DIRECT_BEGIN job_id=%s", job.id)
                audio_gerado = engine.synthesize(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    text_to_speak=str(text),
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                )

        logger.info(
            "HANDLER_SUCCESS job_id=%s audio_path=%s exists=%s",
            job.id, audio_gerado, __import__("os").path.isfile(audio_gerado)
        )
        return {"audio_path": audio_gerado}

    return handle