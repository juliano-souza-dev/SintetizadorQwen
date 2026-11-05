"""Fila sequencial de síntese para o Qwen3-TTS.

O Worker é uma thread do mesmo processo da aplicação. Isso mantém uma única
instância do modelo carregada e evita recarregamentos/concorrência no modelo.
"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class JobQueue:
    """Executa trabalhos em série e mantém seus estados em memória."""

    def __init__(
        self,
        processor: Callable[[dict[str, Any]], str],
        max_history: int = 100,
    ) -> None:
        self._processor = processor
        self._max_history = max(10, max_history)
        self._pending: queue.Queue[str | None] = queue.Queue()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="qwen-tts-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, payload: dict[str, Any]) -> str:
        """Adiciona uma solicitação à fila e retorna seu identificador."""
        with self._lock:
            if self._closed:
                raise RuntimeError("A fila de síntese está encerrada.")

            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": _now(),
                "started_at": None,
                "finished_at": None,
                "result": None,
                "error": None,
                "payload": dict(payload),
            }
            self._pending.put(job_id)
            self._trim_history_locked()
            return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public_copy(job) if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
            return [self._public_copy(job) for job in jobs]

    def _public_copy(self, job: dict[str, Any]) -> dict[str, Any]:
        # Não expõe o payload completo (caminhos e textos) na tabela da UI.
        return {
            key: value
            for key, value in job.items()
            if key != "payload"
        }

    def _run(self) -> None:
        while True:
            job_id = self._pending.get()
            try:
                if job_id is None:
                    return

                with self._lock:
                    job = self._jobs.get(job_id)
                    if job is None:
                        continue
                    job["status"] = "running"
                    job["started_at"] = _now()
                    payload = dict(job["payload"])

                try:
                    result_path = self._processor(payload)
                    with self._lock:
                        job = self._jobs.get(job_id)
                        if job is not None:
                            job["status"] = "completed"
                            job["result"] = result_path
                            job["finished_at"] = _now()
                except Exception as exc:
                    error_detail = traceback.format_exc()
                    with self._lock:
                        job = self._jobs.get(job_id)
                        if job is not None:
                            job["status"] = "failed"
                            job["error"] = str(exc)
                            job["finished_at"] = _now()
                    try:
                        from tts_service import registrar_log
                        registrar_log(
                            f"[JOB {job_id}] Falha no processamento: "
                            f"{exc}\n{error_detail}"
                        )
                    except Exception:
                        pass
            finally:
                self._pending.task_done()

    def _trim_history_locked(self) -> None:
        if len(self._jobs) <= self._max_history:
            return
        terminal = [
            job for job in self._jobs.values()
            if job["status"] in {"completed", "failed"}
        ]
        terminal.sort(key=lambda item: item["created_at"])
        while len(self._jobs) > self._max_history and terminal:
            oldest = terminal.pop(0)
            self._jobs.pop(oldest["id"], None)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending.put(None)
        if wait:
            self._worker.join()
