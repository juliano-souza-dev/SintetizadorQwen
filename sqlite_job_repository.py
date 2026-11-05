from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs import Job

UTC = timezone.utc


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso_to_dt(val: str | None) -> datetime | None:
    if not val:
        return None
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    return datetime.fromisoformat(val)


class SqliteJobRepository:
    """Implementação durável do protocolo JobRepository baseada em SQLite local."""

    def __init__(self, db_path: str = "data/jobs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT,
                    error_message TEXT,
                    worker_id TEXT,
                    idempotency_key TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    lease_heartbeat_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            # Coluna de progresso dinâmico (caso não exista em bancos antigos)
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN progress TEXT")
            except Exception:
                pass

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs (status, kind)"
            )

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        output_data = json.loads(row["output"]) if row["output"] else None
        # Se houver progresso guardado, injeta no output ou metadados para leitura na UI
        job = Job(
            id=uuid.UUID(row["id"]),
            kind=row["kind"],
            status=row["status"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            input=json.loads(row["input"]),
            output=output_data,
            error_message=row["error_message"],
            worker_id=row["worker_id"],
            created_at=_iso_to_dt(row["created_at"]),
            updated_at=_iso_to_dt(row["updated_at"]),
        )
        return job

    def enqueue(
        self,
        *,
        kind: str,
        input: dict[str, Any],
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> Job:
        job_id = str(uuid.uuid4())
        now = _utcnow().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, attempt, max_attempts, input,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    kind,
                    max_attempts,
                    json.dumps(input),
                    idempotency_key,
                    now,
                    now,
                ),
            )
        return self.get(job_id)  # type: ignore

    def claim_next(
        self,
        *,
        worker_id: str,
        now: datetime,
        allowed_kinds: tuple[str, ...] | None = None,
        excluded_kinds: tuple[str, ...] = (),
    ) -> Job | None:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = "SELECT * FROM jobs WHERE status = 'queued'"
            params: list[Any] = []

            if allowed_kinds is not None:
                placeholders = ",".join("?" for _ in allowed_kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(allowed_kinds)
            if excluded_kinds:
                placeholders = ",".join("?" for _ in excluded_kinds)
                query += f" AND kind NOT IN ({placeholders})"
                params.extend(excluded_kinds)

            query += " ORDER BY created_at ASC LIMIT 1"
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None

            job_id = row["id"]
            attempt = row["attempt"] + 1
            conn.execute(
                """
                UPDATE jobs SET 
                    status = 'running',
                    attempt = ?,
                    worker_id = ?,
                    lease_heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (attempt, worker_id, now_str, now_str, job_id),
            )
            return self.get(job_id)

    def heartbeat(self, *, job_id: str, worker_id: str, now: datetime) -> bool:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    lease_heartbeat_at = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (now_str, now_str, job_id, worker_id),
            )
            return cursor.rowcount > 0

    def update_progress(self, *, job_id: str, worker_id: str, progress_data: dict[str, Any]) -> bool:
        """Permite atualizar o estado de progresso intermediário (ex: chunks concluídos)."""
        now_str = _utcnow().isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    output = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (json.dumps(progress_data), now_str, job_id, worker_id),
            )
            return cursor.rowcount > 0

    def cancellation_requested(self, *, job_id: str, worker_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT cancellation_requested FROM jobs WHERE id = ? AND worker_id = ?",
                (job_id, worker_id),
            )
            row = cursor.fetchone()
            return bool(row["cancellation_requested"]) if row else False

    def request_cancellation(self, *, job_id: str, now: datetime) -> Job | None:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE jobs SET 
                    cancellation_requested = 1,
                    updated_at = ?
                WHERE id = ? AND status IN ('queued', 'running')
                """,
                (now_str, job_id),
            )
            conn.execute(
                """
                UPDATE jobs SET 
                    status = 'cancelled',
                    updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now_str, job_id),
            )
        return self.get(job_id)

    def succeed(
        self, *, job_id: str, worker_id: str, output: dict[str, Any], now: datetime
    ) -> bool:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    status = 'succeeded',
                    output = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (json.dumps(output), now_str, job_id, worker_id),
            )
            return cursor.rowcount > 0

    def fail(
        self,
        *,
        job_id: str,
        worker_id: str,
        error_message: str,
        now: datetime,
    ) -> bool:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    status = 'failed',
                    error_message = ?,
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (error_message, now_str, job_id, worker_id),
            )
            return cursor.rowcount > 0

    def cancel(self, *, job_id: str, worker_id: str, now: datetime) -> bool:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    status = 'cancelled',
                    updated_at = ?
                WHERE id = ? AND worker_id = ? AND status = 'running'
                """,
                (now_str, job_id, worker_id),
            )
            return cursor.rowcount > 0

    def recover_abandoned(self, *, before: datetime, now: datetime) -> int:
        before_str = before.isoformat()
        now_str = now.isoformat()
        with self._get_conn() as conn:
            # Retorna jobs 'running' cuja lease expirou de volta para 'queued' para reiniciar do início
            cursor = conn.execute(
                """
                UPDATE jobs SET 
                    status = CASE WHEN attempt < max_attempts THEN 'queued' ELSE 'failed' END,
                    error_message = CASE WHEN attempt < max_attempts THEN 'servidor reiniciado - retomando job' ELSE 'excedeu tentativas' END,
                    worker_id = NULL,
                    updated_at = ?
                WHERE status = 'running'
                  AND (lease_heartbeat_at IS NULL OR lease_heartbeat_at < ?)
                """,
                (now_str, before_str),
            )
            return cursor.rowcount

    def get(self, job_id: str) -> Job | None:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            return self._row_to_job(row) if row else None

    def list(
        self, *, offset: int, limit: int, status: str | None = None
    ) -> tuple[list[Job], int]:
        with self._get_conn() as conn:
            if status:
                c = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
                )
                total = c.fetchone()[0]
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                c = conn.execute("SELECT COUNT(*) FROM jobs")
                total = c.fetchone()[0]
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [self._row_to_job(r) for r in rows], total

    def retry(self, *, job_id: str, now: datetime) -> Job | None:
        now_str = now.isoformat()
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'queued', error_message = NULL, updated_at = ? WHERE id = ?",
                (now_str, job_id),
            )
        return self.get(job_id)

    def latest_status_for_project(self, project_id: str) -> str | None:
        return None

    def latest_for_project(
        self, project_id: str, kind: str, status: str | None = None
    ) -> Job | None:
        return None