from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Generator
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

from jobs import Job
from persistent_worker import JobCancelled, JobExecution

if TYPE_CHECKING:
    from tts_service import TTSService

logger = logging.getLogger("tts_job_handler")


@contextlib.contextmanager
def _keep_lease(
    execution: JobExecution, interval_seconds: float = 20.0
) -> Generator[None, None, None]:
    """Mantém a reserva viva inclusive durante o carregamento do modelo."""
    stop_event = Event()
    heartbeat_errors: list[BaseException] = []

    def heartbeat_loop() -> None:
        while not stop_event.wait(interval_seconds):
            try:
                execution.heartbeat()
                execution.raise_if_cancelled()
            except BaseException as error:
                heartbeat_errors.append(error)
                if isinstance(error, JobCancelled):
                    logger.warning("Cancelamento detectado durante a execução do job %s.", execution.job_id)
                else:
                    logger.exception(
                        "LEASE_HEARTBEAT_FAILED job_id=%s. A execução perdeu a confirmação da reserva.",
                        execution.job_id,
                    )
                stop_event.set()
                return

    execution.heartbeat()
    execution.raise_if_cancelled()
    thread = Thread(
        target=heartbeat_loop,
        daemon=True,
        name=f"job-heartbeat-{execution.job_id[:8]}",
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=3.0)

    if heartbeat_errors:
        error = heartbeat_errors[0]
        if isinstance(error, JobCancelled):
            raise error
        raise RuntimeError(
            f"Lease do job {execution.job_id} foi perdido durante a execução."
        ) from error


def make_tts_handler(
    engine_factory: Callable[[], "TTSService"],
) -> Callable[[Job, JobExecution], dict[str, Any]]:
    """Cria o handler de síntese; o modelo só é carregado após reservar um job."""

    def handle(job: Job, execution: JobExecution) -> dict[str, Any]:
        logger.info(
            "HANDLER_START job_id=%s kind=%s attempt=%s/%s",
            job.id, job.kind, job.attempt, job.max_attempts,
        )
        execution.raise_if_cancelled()
        payload = job.input or {}

        ref_audio_path = payload.get("ref_audio_path")
        ref_text = payload.get("ref_text")
        output_path = payload.get("output_path")
        text = payload.get("text")
        chunks = payload.get("chunks")
        temperature = float(payload.get("temperature", 0.65))

        if not isinstance(ref_audio_path, str) or not ref_audio_path:
            raise ValueError("Campo obrigatório 'ref_audio_path' inválido ou ausente.")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise ValueError("Campo obrigatório 'ref_text' inválido ou ausente.")
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("Campo obrigatório 'output_path' inválido ou ausente.")
        if not os.path.isfile(ref_audio_path):
            raise FileNotFoundError(f"Áudio de referência não encontrado: {ref_audio_path}")

        valid_chunks = isinstance(chunks, list) and bool(chunks)
        if not valid_chunks and (not isinstance(text, str) or not text.strip()):
            raise ValueError("É obrigatório informar 'text' ou uma lista não vazia de 'chunks'.")
        if valid_chunks:
            for index, chunk in enumerate(chunks):
                if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                    raise ValueError(f"Chunk {index + 1} inválido: esperado objeto com campo text.")
                if chunk.get("lang", "pt") is None:
                    raise ValueError(f"Chunk {index + 1} possui idioma inválido.")

        logger.info(
            "HANDLER_VALIDATED job_id=%s mode=%s chunks=%s temperature=%s",
            job.id, "chunks" if valid_chunks else "direct",
            len(chunks) if valid_chunks else 0, temperature,
        )

        # O heartbeat começa ANTES de carregar torch/Qwen. Uma inicialização longa
        # não deixa o lease expirar e, se falhar, o traceback fica associado ao job.
        with _keep_lease(execution):
            logger.info("TTS_ENGINE_LOAD_BEGIN job_id=%s", job.id)
            engine = engine_factory()
            logger.info("TTS_ENGINE_READY job_id=%s", job.id)
            execution.raise_if_cancelled()

            if valid_chunks:
                logger.info("SYNTHESIS_CHUNKS_BEGIN job_id=%s total_chunks=%s", job.id, len(chunks))
                audio_path = engine.synthesize_chunks(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    chunks=chunks,
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                    progress_callback=execution.report_progress,
                )
            else:
                logger.info("SYNTHESIS_DIRECT_BEGIN job_id=%s", job.id)
                execution.report_progress({"phase": "synthesizing", "event": "direct_started"})
                audio_path = engine.synthesize(
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    text_to_speak=text,
                    output_path=output_path,
                    temperature=temperature,
                    checkpoint=execution.raise_if_cancelled,
                )
                execution.report_progress({"phase": "finalizing", "event": "direct_completed"})

            execution.raise_if_cancelled()
            execution.heartbeat()

        if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"O motor informou um áudio de saída que não existe: {audio_path!r}"
            )

        logger.info("HANDLER_SUCCESS job_id=%s output_exists=true", job.id)
        return {"audio_path": audio_path}

    return handle
