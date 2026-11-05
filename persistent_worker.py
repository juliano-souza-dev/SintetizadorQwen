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
        recovered = self._repository.recover_abandoned(
            before=now - self._heartbeat_timeout, now=now
        )
        if recovered:
            self._logger.info(f"Jobs recuperados de timeout: {recovered}")

        if self._allowed_kinds is None and not self._excluded_kinds:
            job = self._repository.claim_next(worker_id=self._worker_id, now=now)
        else:
            job = self._repository.claim_next(
                worker_id=self._worker_id,
                now=now,
                allowed_kinds=self._allowed_kinds,
                excluded_kinds=self._excluded_kinds,
            )

        if job is None:
            return False

        execution = JobExecution(self._repository, job, self._worker_id)
        self._logger.info(f"Iniciando job: {job.id} (kind={job.kind}, tentativa={job.attempt})")
        try:
            execution.raise_if_cancelled()
            handler = self._handlers.get(job.kind)
            if handler is None:
                raise ValueError(f"Tipo de job nao suportado: {job.kind}")

            output = handler(job, execution)
            execution.raise_if_cancelled()

            self._repository.succeed(
                job_id=str(job.id), worker_id=self._worker_id, output=output, now=_utcnow()
            )
            self._logger.info(f"Job concluido com sucesso: {job.id}")
        except JobCancelled:
            self._repository.cancel(job_id=str(job.id), worker_id=self._worker_id, now=_utcnow())
            self._logger.warning(f"Job cancelado: {job.id}")
        except Exception as error:
            self._repository.fail(
                job_id=str(job.id),
                worker_id=self._worker_id,
                error_message=str(error),
                now=_utcnow(),
            )
            self._logger.error(f"Job falhou: {job.id} - {error}", exc_info=True)

        return True

    def run_forever(self, *, poll_interval_seconds: float, stop_event: Event) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds deve ser positivo")
        self._logger.info("Worker iniciado e monitorando a fila...")
        while not stop_event.is_set():
            if not self.run_once():
                stop_event.wait(poll_interval_seconds)
        self._logger.info("Worker finalizado.")
