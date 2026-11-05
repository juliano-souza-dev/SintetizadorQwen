from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Any, Protocol

from jobs import Job

JobHandler = Callable[[Job, "JobExecution"], dict[str, Any]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobCancelled(Exception):
    pass


class JobRepository(Protocol):
    def enqueue(
        self, *, kind: str, input: dict[str, Any], idempotency_key: str | None, max_attempts: int
    ) -> Job: ...
    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        allowed_kinds: tuple[str, ...] | None = None,
        excluded_kinds: tuple[str, ...] = (),
    ) -> Job | None: ...
    def heartbeat(self, *, job_id: str, worker_id: str, now: datetime) -> bool: ...
    def cancellation_requested(self, *, job_id: str, worker_id: str) -> bool: ...
    def request_cancellation(self, *, job_id: str, now: datetime) -> Job | None: ...
    def succeed(
        self, *, job_id: str, worker_id: str, output: dict[str, Any], now: datetime
    ) -> bool: ...
    def fail(self, *, job_id: str, worker_id: str, error_message: str, now: datetime) -> bool: ...
    def cancel(self, *, job_id: str, worker_id: str, now: datetime) -> bool: ...
    def recover_abandoned(self, *, before: datetime, now: datetime) -> int: ...
    def get(self, job_id: str) -> Job | None: ...


class JobExecution:
    def __init__(self, repository: JobRepository, job: Job, worker_id: str) -> None:
        self._repository, self._job, self._worker_id = repository, job, worker_id

    def heartbeat(self) -> None:
        if not self._repository.heartbeat(
            job_id=str(self._job.id), worker_id=self._worker_id, now=_utcnow()
        ):
            raise RuntimeError("job lease is no longer owned by this worker")

    def raise_if_cancelled(self) -> None:
        if self._repository.cancellation_requested(
            job_id=str(self._job.id), worker_id=self._worker_id
        ):
            raise JobCancelled()


class PersistentWorker:
    def __init__(
        self,
        repository: JobRepository,
        worker_id: str,
        handlers: dict[str, JobHandler],
        heartbeat_timeout: timedelta = timedelta(minutes=2),
        logger: logging.Logger | None = None,
        allowed_kinds: tuple[str, ...] | None = None,
        excluded_kinds: tuple[str, ...] = (),
    ) -> None:
        self._repository = repository
        self._worker_id = worker_id
        self._handlers = handlers
        self._allowed_kinds = allowed_kinds
        self._excluded_kinds = excluded_kinds
        self._heartbeat_timeout = heartbeat_timeout
        self._logger = logger or logging.getLogger("worker_tts")

    def run_once(self) -> bool:
        now = _utcnow()
        try:
            recovered = self._repository.recover_abandoned(
                before=now - self._heartbeat_timeout, now=now
            )
            if recovered:
                self._logger.warning("Jobs recuperados de timeout: %s", recovered)

            if self._allowed_kinds is None and not self._excluded_kinds:
                job = self._repository.claim_next(worker_id=self._worker_id, now=now)
            else:
                job = self._repository.claim_next(
                    worker_id=self._worker_id,
                    now=now,
                    allowed_kinds=self._allowed_kinds,
                    excluded_kinds=self._excluded_kinds,
                )
        except Exception:
            self._logger.exception(
                "ERRO AO CONSULTAR/REIVINDICAR JOB NA FILA. "
                "O worker continuará tentando após o próximo intervalo."
            )
            return False

        if job is None:
            return False

        import json
        import traceback
        from pathlib import Path

        job_id = str(job.id)
        log_dir = Path(__file__).resolve().parent / "logs" / "jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        job_log_path = log_dir / f"{job_id}.log"
        job_handler = logging.FileHandler(job_log_path, encoding="utf-8")
        job_handler.setLevel(logging.DEBUG)
        job_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(job_handler)

        execution = JobExecution(self._repository, job, self._worker_id)
        phase = "job_start"
        self._logger.info(
            "JOB_START id=%s kind=%s attempt=%s/%s worker=%s log=%s",
            job_id, job.kind, job.attempt, job.max_attempts,
            self._worker_id, job_log_path
        )
        try:
            self._logger.info(
                "JOB_INPUT id=%s payload=%s",
                job_id,
                json.dumps(job.input, ensure_ascii=False, default=str)
            )
            phase = "cancellation_check_before_handler"
            execution.raise_if_cancelled()

            phase = "handler_lookup"
            handler = self._handlers.get(job.kind)
            if handler is None:
                raise ValueError(f"Tipo de job não suportado: {job.kind}")

            phase = "handler_execution"
            self._logger.info("JOB_HANDLER_BEGIN id=%s phase=%s", job_id, phase)
            output = handler(job, execution)
            self._logger.info(
                "JOB_HANDLER_END id=%s output=%s",
                job_id, json.dumps(output, ensure_ascii=False, default=str)
            )

            phase = "cancellation_check_after_handler"
            execution.raise_if_cancelled()

            phase = "persist_success"
            persisted = self._repository.succeed(
                job_id=job_id, worker_id=self._worker_id,
                output=output, now=_utcnow()
            )
            if not persisted:
                self._logger.error(
                    "JOB_SUCCESS_NOT_PERSISTED id=%s worker=%s. "
                    "O handler terminou, mas o repositório recusou a transição.",
                    job_id, self._worker_id
                )
            else:
                self._logger.info("JOB_SUCCEEDED id=%s", job_id)

        except JobCancelled:
            phase = "persist_cancelled"
            try:
                persisted = self._repository.cancel(
                    job_id=job_id, worker_id=self._worker_id, now=_utcnow()
                )
                self._logger.warning(
                    "JOB_CANCELLED id=%s persisted=%s", job_id, persisted
                )
            except Exception:
                self._logger.exception(
                    "JOB_CANCEL_PERSIST_FAILED id=%s", job_id
                )

        except Exception as error:
            error_trace = traceback.format_exc()
            error_details = (
                f"Fase: {phase}\\n"
                f"Tipo: {type(error).__name__}\\n"
                f"Erro: {error}\\n\\n"
                f"Traceback completo:\\n{error_trace}"
            )
            self._logger.error(
                "JOB_FAILED id=%s phase=%s type=%s error=%s\\n%s",
                job_id, phase, type(error).__name__, error, error_trace
            )
            try:
                persisted = self._repository.fail(
                    job_id=job_id,
                    worker_id=self._worker_id,
                    error_message=error_details[-30000:],
                    now=_utcnow(),
                )
                self._logger.error(
                    "JOB_FAILURE_PERSISTED id=%s persisted=%s", job_id, persisted
                )
            except Exception:
                self._logger.exception(
                    "JOB_FAILURE_COULD_NOT_BE_PERSISTED id=%s. "
                    "O traceback permanece no log individual do job.",
                    job_id
                )
        finally:
            self._logger.info(
                "JOB_END id=%s log_file=%s", job_id, job_log_path
            )
            root_logger.removeHandler(job_handler)
            job_handler.flush()
            job_handler.close()

        return True

    def run_forever(self, *, poll_interval_seconds: float, stop_event: Event) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds deve ser positivo")
        self._logger.info("Worker iniciado e monitorando a fila...")
        while not stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception:
                # Última barreira: nenhuma exceção inesperada pode matar silenciosamente o loop.
                self._logger.exception(
                    "ERRO NÃO TRATADO NO LOOP PRINCIPAL DO WORKER. "
                    "O processo continuará ativo e tentará novamente."
                )
                processed = False
            if not processed:
                stop_event.wait(poll_interval_seconds)
        self._logger.info("Worker finalizado.")
