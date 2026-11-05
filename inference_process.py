from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from persistent_worker import PermanentJobError


class InferenceTimeoutError(PermanentJobError):
    """Inferência Qwen sem atividade pelo limite configurado."""


def _inference_child(
    connection: Connection, model_path: str, dict_file: str, log_path: str
) -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("qwen_inference_child")
    engine = None

    try:
        while True:
            try:
                command = connection.recv()
            except EOFError:
                break

            if command.get("operation") == "close":
                break

            try:
                if engine is None:
                    connection.send({
                        "type": "progress",
                        "data": {"phase": "loading_model", "event": "engine_loading"},
                    })
                    from tts_service import TTSService

                    engine = TTSService(model_path=model_path, dict_file=dict_file)
                    connection.send({
                        "type": "progress",
                        "data": {"phase": "loading_model", "event": "engine_ready"},
                    })

                operation = command["operation"]
                args = command["args"]

                if operation == "chunks":
                    output_path = engine.synthesize_chunks(
                        ref_audio_path=args["ref_audio_path"],
                        ref_text=args["ref_text"],
                        chunks=args["chunks"],
                        output_path=args["output_path"],
                        temperature=args["temperature"],
                        checkpoint=None,
                        progress_callback=lambda data: connection.send({
                            "type": "progress", "data": data
                        }),
                    )
                elif operation == "direct":
                    connection.send({
                        "type": "progress",
                        "data": {"phase": "synthesizing", "event": "direct_started"},
                    })
                    output_path = engine.synthesize(
                        ref_audio_path=args["ref_audio_path"],
                        ref_text=args["ref_text"],
                        text_to_speak=args["text_to_speak"],
                        output_path=args["output_path"],
                        temperature=args["temperature"],
                        checkpoint=None,
                    )
                    connection.send({
                        "type": "progress",
                        "data": {"phase": "finalizing", "event": "direct_completed"},
                    })
                else:
                    raise ValueError(f"Operação de inferência desconhecida: {operation}")

                connection.send({"type": "result", "output": output_path})
            except BaseException as error:
                logger.exception("Falha durante a inferência.")
                try:
                    connection.send({
                        "type": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    })
                except (BrokenPipeError, EOFError, OSError):
                    break
    finally:
        try:
            connection.close()
        except OSError:
            pass


class InferenceProcessClient:
    """Cliente para um processo Qwen persistente, isolado do worker da fila."""

    def __init__(
        self,
        model_path: str,
        dict_file: str,
        *,
        timeout_seconds: float | None = None,
        log_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.dict_file = dict_file
        self.timeout_seconds = timeout_seconds or float(
            os.environ.get("QWEN_INFERENCE_TIMEOUT_SECONDS", "900")
        )
        self.log_path = log_path or str(
            Path(__file__).resolve().parent / "worker_inference.log"
        )
        self._context = multiprocessing.get_context("spawn")
        self._process = None
        self._connection: Connection | None = None

    def _start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop(force=True)
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_inference_child,
            args=(child, self.model_path, self.dict_file, self.log_path),
            name="qwen-inference",
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent

    def _stop(self, *, force: bool) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None

        if connection is not None:
            if not force:
                try:
                    connection.send({"operation": "close"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
            try:
                connection.close()
            except OSError:
                pass

        if process is not None:
            if process.is_alive() and force:
                process.terminate()
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            else:
                process.join(timeout=1)

    def close(self) -> None:
        self._stop(force=False)

    def _run(
        self,
        operation: str,
        args: dict[str, Any],
        *,
        checkpoint: Callable[[], None] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> str:
        self._start()
        assert self._connection is not None
        assert self._process is not None
        connection = self._connection
        process = self._process
        connection.send({"operation": operation, "args": args})
        last_activity = time.monotonic()

        try:
            while True:
                if checkpoint is not None:
                    checkpoint()

                if connection.poll(0.5):
                    message = connection.recv()
                    last_activity = time.monotonic()
                    message_type = message.get("type")

                    if message_type == "progress":
                        if progress_callback is not None:
                            progress_callback(message.get("data") or {})
                    elif message_type == "result":
                        return str(message["output"])
                    elif message_type == "error":
                        details = message.get("traceback", "")
                        raise RuntimeError(
                            f"Falha na inferência Qwen: {message.get('error', 'erro desconhecido')}\n{details}"
                        )
                    else:
                        raise RuntimeError(
                            f"Mensagem inválida recebida do processo Qwen: {message_type!r}"
                        )

                if not process.is_alive():
                    raise RuntimeError(
                        f"Processo de inferência Qwen encerrou inesperadamente "
                        f"(exit_code={process.exitcode})."
                    )

                inactive_for = time.monotonic() - last_activity
                if inactive_for >= self.timeout_seconds:
                    raise InferenceTimeoutError(
                        "A inferência Qwen foi encerrada após "
                        f"{self.timeout_seconds:.0f} segundos sem progresso."
                    )
        except BaseException:
            # Cancelamento, perda do lease, timeout ou falha de protocolo:
            # encerra o processo isolado sem derrubar o consumidor da fila.
            self._stop(force=True)
            raise

    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        text_to_speak: str,
        output_path: str,
        temperature: float = 0.65,
        checkpoint: Callable[[], None] | None = None,
    ) -> str:
        return self._run(
            "direct",
            {
                "ref_audio_path": ref_audio_path,
                "ref_text": ref_text,
                "text_to_speak": text_to_speak,
                "output_path": output_path,
                "temperature": temperature,
            },
            checkpoint=checkpoint,
            progress_callback=None,
        )

    def synthesize_chunks(
        self,
        ref_audio_path: str,
        ref_text: str,
        chunks: list[dict[str, Any]],
        output_path: str,
        temperature: float = 0.65,
        checkpoint: Callable[[], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        return self._run(
            "chunks",
            {
                "ref_audio_path": ref_audio_path,
                "ref_text": ref_text,
                "chunks": chunks,
                "output_path": output_path,
                "temperature": temperature,
            },
            checkpoint=checkpoint,
            progress_callback=progress_callback,
        )
