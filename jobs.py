from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class Job:
    id: UUID
    kind: str
    status: str
    attempt: int
    max_attempts: int
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error_message: str | None = None
    worker_id: str | None = None
    idempotency_key: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    cancel_requested_at: datetime | None = None
