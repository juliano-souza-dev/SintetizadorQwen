from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from jobs import Job

JobHandler = Callable[[Job, "JobExecution"], dict[str, Any]]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobCancelled(Exception):
    pass


class JobRepository(Protocol):
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
    def succeed(
        self, *, job_id: str, worker_id: str, output: dict[str, Any], now: datetime
    ) -> bool: ...
    def fail(
        self, *, job_id: str, worker_id: str, error_message: str, now: datetime
    ) -> bool: ...
    def cancel(self, *, job_id: str, worker_id: str, now: datetime) -> bool: ...
    def recover_abandoned(self, *, before: datetime, now: datetime) -> int: ...


class JobExecution:
    def __init__(self, repository: JobRepository, job: Job, worker_id: str) -> None:
        self._repository = repository
        self._job = job
        self._worker_id = worker_id

    @property
    def job_id(self) -> str:
        return str(self._job.id)

    def heartbeat(self) -> None:
        if not self._repository.heartbeat(
            job_id=self.job_id, worker_id=self._worker_id, now=_utcnow()
        ):
            raise RuntimeError("O lease deste job não pertence mais a este worker.")

    def raise_if_cancelled(self) -> None:
        if self._repository.cancellation_requested(
            job_id=self.job_id, worker_id=self._worker_id
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
                self._logger.warning("JOBS_RECOVERED count=%s", recovered)

            job = self._repository.claim_next(
                worker_id=self._worker_id,
                now=now,
                allowed_kinds=self._allowed_kinds,
                excluded_kinds=self._excluded_kinds,
            )
        except Exception:
            self._logger.exception(
                "QUEUE_POLL_FAILED worker=%s. O consumidor continuará tentando.",
                self._worker_id,
            )
            return False

        if job is None:
            return False

        job_id = str(job.id)
        execution = JobExecution(self._repository, job, self._worker_id)
        started = time.monotonic()
        log_path = Path(__file__).resolve().parent / "logs" / "jobs" / f"{job_id}.log"
        job_handler: logging.Handler | None = None
        root_logger = logging.getLogger()

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            job_handler = logging.FileHandler(log_path, encoding="utf-8")
            job_handler.setLevel(logging.DEBUG)
            job_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            root_logger.addHandler(job_handler)
        except OSError:
            self._logger.exception(
                "JOB_LOG_FILE_UNAVAILABLE job_id=%s path=%s. "
                "A execução continuará no log principal.",
                job_id, log_path,
            )

        phase = "job_claimed"
        self._logger.info(
            "JOB_STARTED id=%s kind=%s attempt=%s/%s worker=%s log=%s",
            job_id, job.kind, job.attempt, job.max_attempts, self._worker_id, log_path,
        )
        try:
            # Não registra texto nem caminhos privados. Registra somente metadados úteis.
            payload = job.input or {}
            chunks = payload.get("chunks")
            self._logger.info(
                "JOB_INPUT_SUMMARY id=%s mode=%s chunks=%s text_chars=%s "
                "reference_exists=%s temperature=%s",
                job_id,
                "chunks" if isinstance(chunks, list) and chunks else "direct",
                len(chunks) if isinstance(chunks, list) else 0,
                len(payload.get("text", "")) if isinstance(payload.get("text"), str) else 0,
                Path(payload["ref_audio_path"]).is_file()
                if isinstance(payload.get("ref_audio_path"), str) else False,
                payload.get("temperature", 0.65),
            )

            phase = "lease_validation"
            execution.heartbeat()
            execution.raise_if_cancelled()

            phase = "handler_lookup"
            handler = self._handlers.get(job.kind)
            if handler is None:
                raise ValueError(f"Tipo de job não suportado: {job.kind}")

            phase = "handler_execution"
            self._logger.info("JOB_HANDLER_BEGIN id=%s", job_id)
            output = handler(job, execution)
            self._logger.info(
                "JOB_HANDLER_END id=%s output_keys=%s",
                job_id, sorted(output.keys()) if isinstance(output, dict) else "invalid",
            )

            phase = "post_handler_validation"
            execution.raise_if_cancelled()
            if not isinstance(output, dict):
                raise TypeError("O handler do job deve retornar um dicionário.")

            phase = "persist_success"
            persisted = self._repository.succeed(
                job_id=job_id, worker_id=self._worker_id, output=output, now=_utcnow()
            )
            if not persisted:
                raise RuntimeError(
                    "O handler terminou, mas o repositório recusou a conclusão: "
                    "o lease expirou, foi perdido ou o job foi cancelado."
                )
            self._logger.info("JOB_SUCCEEDED id=%s duration_ms=%s", job_id, int(
                (time.monotonic() - started) * 1000
            ))

        except JobCancelled:
            phase = "persist_cancelled"
            try:
                persisted = self._repository.cancel(
                    job_id=job_id, worker_id=self._worker_id, now=_utcnow()
                )
                self._logger.warning("JOB_CANCELLED id=%s persisted=%s", job_id, persisted)
            except Exception:
                self._logger.exception("JOB_CANCEL_PERSIST_FAILED id=%s", job_id)

        except Exception as error:
            error_trace = traceback.format_exc()
            error_details = (
                f"Fase: {phase}\n"
                f"Tipo: {type(error).__name__}\n"
                f"Erro: {error}\n\n"
                f"Traceback completo:\n{error_trace}"
            )
            self._logger.error(
                "JOB_FAILED id=%s phase=%s type=%s error=%s\n%s",
                job_id, phase, type(error).__name__, error, error_trace,
            )
            try:
                persisted = self._repository.fail(
                    job_id=job_id,
                    worker_id=self._worker_id,
                    error_message=error_details[-30000:],
                    now=_utcnow(),
                )
                self._logger.error("JOB_FAILURE_PERSISTED id=%s persisted=%s", job_id, persisted)
            except Exception:
                self._logger.exception("JOB_FAILURE_COULD_NOT_BE_PERSISTED id=%s", job_id)

        finally:
            self._logger.info(
                "JOB_FINISHED id=%s elapsed_ms=%s log=%s",
                job_id, int((time.monotonic() - started) * 1000), log_path,
            )
            if job_handler is not None:
                root_logger.removeHandler(job_handler)
                job_handler.flush()
                job_handler.close()

        return True

    def run_forever(self, *, poll_interval_seconds: float, stop_event: Event) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds deve ser positivo.")
        self._logger.info("WORKER_STARTED worker=%s", self._worker_id)
        while not stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception:
                self._logger.exception(
                    "WORKER_LOOP_UNEXPECTED_ERROR worker=%s. O loop continuará ativo.",
                    self._worker_id,
                )
                processed = False
            if not processed:
                stop_event.wait(poll_interval_seconds)
        self._logger.info("WORKER_STOPPED worker=%s", self._worker_id)
