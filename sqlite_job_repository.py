from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs import Job

UTC = timezone.utc


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


class SqliteJobRepository:
    """Fila durável SQLite com reserva exclusiva, leases e retries."""

    def __init__(self, db_path: str = "data/jobs.db") -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    input TEXT NOT NULL,
                    output TEXT,
                    error_message TEXT,
                    worker_id TEXT,
                    idempotency_key TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    cancel_requested_at TEXT,
                    lease_heartbeat_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    progress TEXT
                )
                """
            )

            # Migração incremental: preserva bancos locais criados por versões anteriores.
            existing = {
                row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            migrations = {
                "idempotency_key": "TEXT",
                "cancellation_requested": "INTEGER NOT NULL DEFAULT 0",
                "cancel_requested_at": "TEXT",
                "lease_heartbeat_at": "TEXT",
                "started_at": "TEXT",
                "finished_at": "TEXT",
                "progress": "TEXT",
            }
            for column, definition in migrations.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(status, kind, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(status, lease_heartbeat_at)"
            )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        keys = set(row.keys())
        return Job(
            id=uuid.UUID(row["id"]),
            kind=row["kind"],
            status=row["status"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            input=json.loads(row["input"] or "{}"),
            output=json.loads(row["output"]) if row["output"] else None,
            error_message=row["error_message"],
            worker_id=row["worker_id"],
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            created_at=_iso_to_dt(row["created_at"]),
            updated_at=_iso_to_dt(row["updated_at"]),
            started_at=_iso_to_dt(row["started_at"]) if "started_at" in keys else None,
            finished_at=_iso_to_dt(row["finished_at"]) if "finished_at" in keys else None,
            heartbeat_at=_iso_to_dt(row["lease_heartbeat_at"]) if "lease_heartbeat_at" in keys else None,
            cancel_requested_at=(
                _iso_to_dt(row["cancel_requested_at"])
                if "cancel_requested_at" in keys
                else None
            ),
        )

    def enqueue(
        self,
        *,
        kind: str,
        input: dict[str, Any],
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> Job:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind é obrigatório.")
        if max_attempts < 1:
            raise ValueError("max_attempts deve ser maior ou igual a 1.")

        job_id = str(uuid.uuid4())
        now = _utcnow().isoformat()
        with self._connection() as conn:
            # BEGIN IMMEDIATE serializa produtores que usam a mesma chave idempotente.
            conn.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = conn.execute(
                    "SELECT * FROM jobs WHERE idempotency_key = ? ORDER BY created_at LIMIT 1",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return self._row_to_job(existing)

            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, attempt, max_attempts, input,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?)
                """,
                (job_id, kind.strip(), max_attempts, _dump(input), idempotency_key, now, now),
            )
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise RuntimeError("Job foi inserido, mas não pôde ser recuperado.")
            return self._row_to_job(row)

    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        allowed_kinds: tuple[str, ...] | None = None,
        excluded_kinds: tuple[str, ...] = (),
    ) -> Job | None:
        now_str = now.isoformat()
        with self._connection() as conn:
            # Uma única transação de escrita impede dois workers de reservarem o mesmo job.
            conn.execute("BEGIN IMMEDIATE")
            query = "SELECT id FROM jobs WHERE status IN ('queued', 'retryable')"
            params: list[Any] = []

            if allowed_kinds is not None:
                if not allowed_kinds:
                    return None
                placeholders = ",".join("?" for _ in allowed_kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(allowed_kinds)
            if excluded_kinds:
                placeholders = ",".join("?" for _ in excluded_kinds)
                query += f" AND kind NOT IN ({placeholders})"
                params.extend(excluded_kinds)

            query += " ORDER BY created_at ASC, id ASC LIMIT 1"
            candidate = conn.execute(query, params).fetchone()
            if candidate is None:
                return None

            job_id = candidate["id"]
            result = conn.execute(
                """
                UPDATE jobs SET
                    status = 'running',
                    attempt = attempt + 1,
                    worker_id = ?,
                    lease_heartbeat_at = ?,
                    started_at = ?,
                    finished_at = NULL,
                    cancellation_requested = 0,
                    cancel_requested_at = NULL,
                    output = NULL,
                    error_message = NULL,
                    updated_at = ?
                WHERE id = ? AND status IN ('queued', 'retryable')
                """,
                (worker_id, now_str, now_str, now_str, job_id),
            )
            if result.rowcount != 1:
                return None
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._row_to_job(row) if row else None

    def heartbeat(self, *, job_id: str, worker_id: str, now: datetime) -> bool:
        now_str = now.isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET lease_heartbeat_at = ?, updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (now_str, now_str, job_id, worker_id),
            )
            return result.rowcount == 1

    def update_progress(
        self, *, job_id: str, worker_id: str, progress_data: dict[str, Any]
    ) -> bool:
        now_str = _utcnow().isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET output = ?, progress = ?, updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (_dump(progress_data), _dump(progress_data), now_str, job_id, worker_id),
            )
            return result.rowcount == 1

    def cancellation_requested(self, *, job_id: str, worker_id: str) -> bool:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT cancellation_requested FROM jobs
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (job_id, worker_id),
            ).fetchone()
            return bool(row["cancellation_requested"]) if row else False

    def request_cancellation(self, *, job_id: str, now: datetime) -> Job | None:
        now_str = now.isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    cancellation_requested = 1,
                    cancel_requested_at = COALESCE(cancel_requested_at, ?),
                    status = CASE
                        WHEN status IN ('queued', 'retryable') THEN 'cancelled'
                        ELSE status
                    END,
                    finished_at = CASE
                        WHEN status IN ('queued', 'retryable') THEN ?
                        ELSE finished_at
                    END,
                    updated_at = ?
                WHERE id = ? AND status IN ('queued', 'retryable', 'running')
                """,
                (now_str, now_str, now_str, job_id),
            )
        return self.get(job_id)

    def succeed(
        self, *, job_id: str, worker_id: str, output: dict[str, Any], now: datetime
    ) -> bool:
        now_str = now.isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET
                    status = 'succeeded',
                    output = ?,
                    error_message = NULL,
                    finished_at = ?,
                    lease_heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                  AND cancellation_requested = 0
                """,
                (_dump(output), now_str, now_str, now_str, job_id, worker_id),
            )
            return result.rowcount == 1

    def fail(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_message: str,
        now: datetime,
    ) -> bool:
        now_str = now.isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET
                    status = CASE
                        WHEN cancellation_requested = 1 THEN 'cancelled'
                        WHEN attempt < max_attempts THEN 'retryable'
                        ELSE 'failed'
                    END,
                    error_message = ?,
                    finished_at = ?,
                    lease_heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (error_message, now_str, now_str, now_str, job_id, worker_id),
            )
            return result.rowcount == 1

    def cancel(self, *, job_id: str, worker_id: str, now: datetime) -> bool:
        now_str = now.isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET
                    status = 'cancelled',
                    finished_at = ?,
                    lease_heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (now_str, now_str, now_str, job_id, worker_id),
            )
            return result.rowcount == 1

    def recover_abandoned(self, *, before: datetime, now: datetime) -> int:
        before_str, now_str = before.isoformat(), now.isoformat()
        with self._connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET
                    status = CASE
                        WHEN cancellation_requested = 1 THEN 'cancelled'
                        WHEN attempt < max_attempts THEN 'retryable'
                        ELSE 'failed'
                    END,
                    error_message = CASE
                        WHEN cancellation_requested = 1 THEN 'Job cancelado após perda do worker.'
                        WHEN attempt < max_attempts THEN 'Lease do worker expirou; job liberado para nova tentativa.'
                        ELSE 'Lease expirou e o limite de tentativas foi atingido.'
                    END,
                    worker_id = NULL,
                    finished_at = ?,
                    updated_at = ?
                WHERE status = 'running'
                  AND (lease_heartbeat_at IS NULL OR lease_heartbeat_at < ?)
                """,
                (now_str, now_str, before_str),
            )
            return result.rowcount

    def get(self, job_id: str) -> Job | None:
        try:
            normalized_id = str(uuid.UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError):
            return None
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (normalized_id,)).fetchone()
            return self._row_to_job(row) if row else None

    def list(
        self, *, offset: int, limit: int, status: str | None = None
    ) -> tuple[list[Job], int]:
        if offset < 0 or limit < 1:
            raise ValueError("offset deve ser >= 0 e limit deve ser >= 1.")
        with self._connection() as conn:
            if status:
                total = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
                ).fetchone()[0]
                rows = conn.execute(
                    """
                    SELECT * FROM jobs WHERE status = ?
                    ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
                    """,
                    (status, limit, offset),
                ).fetchall()
            else:
                total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                rows = conn.execute(
                    """
                    SELECT * FROM jobs
                    ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [self._row_to_job(row) for row in rows], total

    def retry(self, *, job_id: str, now: datetime) -> Job | None:
        now_str = now.isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET
                    status = 'queued',
                    attempt = 0,
                    worker_id = NULL,
                    lease_heartbeat_at = NULL,
                    cancellation_requested = 0,
                    cancel_requested_at = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    output = NULL,
                    progress = NULL,
                    error_message = NULL,
                    updated_at = ?
                WHERE id = ? AND status IN ('failed', 'cancelled')
                """,
                (now_str, job_id),
            )
        return self.get(job_id)

    def latest_status_for_project(self, project_id: str) -> str | None:
        return None

    def latest_for_project(
        self, project_id: str, kind: str, status: str | None = None
    ) -> Job | None:
        return None
