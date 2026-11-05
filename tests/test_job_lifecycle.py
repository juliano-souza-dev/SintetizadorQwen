from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Thread

from sqlite_job_repository import SqliteJobRepository

UTC = timezone.utc


class JobLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repository = SqliteJobRepository(
            str(Path(self.temp_dir.name) / "jobs.db")
        )

    def now(self) -> datetime:
        return datetime.now(UTC)

    def test_failure_retries_until_max_attempts(self) -> None:
        job = self.repository.enqueue(
            kind="tts_generate", input={"text": "hello"}, max_attempts=2
        )

        first = self.repository.claim_next(worker_id="worker-a", now=self.now())
        self.assertEqual(first.id, job.id)
        self.assertEqual(first.attempt, 1)
        self.assertTrue(self.repository.fail(
            job_id=str(job.id), worker_id="worker-a",
            error_message="first failure", now=self.now()
        ))
        self.assertEqual(self.repository.get(str(job.id)).status, "retryable")

        second = self.repository.claim_next(worker_id="worker-a", now=self.now())
        self.assertEqual(second.id, job.id)
        self.assertEqual(second.attempt, 2)
        self.assertTrue(self.repository.fail(
            job_id=str(job.id), worker_id="worker-a",
            error_message="second failure", now=self.now()
        ))
        self.assertEqual(self.repository.get(str(job.id)).status, "failed")
        self.assertIsNone(self.repository.claim_next(worker_id="worker-a", now=self.now()))

    def test_permanent_failure_does_not_schedule_retry(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={}, max_attempts=3)
        self.repository.claim_next(worker_id="worker-a", now=self.now())

        self.assertTrue(self.repository.fail_permanently(
            job_id=str(job.id),
            worker_id="worker-a",
            error_message="inference inactivity timeout",
            now=self.now(),
        ))
        failed = self.repository.get(str(job.id))
        self.assertEqual(failed.status, "failed")
        self.assertIn("inactivity timeout", failed.error_message)
        self.assertIsNone(self.repository.claim_next(worker_id="worker-a", now=self.now()))

    def test_abandoned_job_is_released_only_after_lease_expiry(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={}, max_attempts=2)
        claimed_at = self.now()
        claimed = self.repository.claim_next(worker_id="dead-worker", now=claimed_at)
        self.assertEqual(claimed.id, job.id)

        self.assertEqual(self.repository.recover_abandoned(
            before=claimed_at - timedelta(seconds=1), now=self.now()
        ), 0)
        self.assertEqual(self.repository.recover_abandoned(
            before=claimed_at + timedelta(seconds=1), now=self.now()
        ), 1)

        recovered = self.repository.get(str(job.id))
        self.assertEqual(recovered.status, "retryable")
        self.assertIsNone(recovered.worker_id)

        reclaimed = self.repository.claim_next(worker_id="healthy-worker", now=self.now())
        self.assertEqual(reclaimed.id, job.id)
        self.assertEqual(reclaimed.attempt, 2)

    def test_heartbeat_prevents_recovery(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={}, max_attempts=2)
        now = self.now()
        self.repository.claim_next(worker_id="worker-a", now=now)
        heartbeat_at = now + timedelta(seconds=30)
        self.assertTrue(self.repository.heartbeat(
            job_id=str(job.id), worker_id="worker-a", now=heartbeat_at
        ))
        self.assertEqual(self.repository.recover_abandoned(
            before=heartbeat_at - timedelta(seconds=1), now=heartbeat_at
        ), 0)

    def test_cancellation_is_cooperative_for_running_job(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())

        requested = self.repository.request_cancellation(
            job_id=str(job.id), now=self.now()
        )
        self.assertEqual(requested.status, "running")
        self.assertTrue(self.repository.cancellation_requested(
            job_id=str(job.id), worker_id="worker-a"
        ))
        self.assertTrue(self.repository.cancel(
            job_id=str(job.id), worker_id="worker-a", now=self.now()
        ))
        self.assertEqual(self.repository.get(str(job.id)).status, "cancelled")

    def test_queued_job_can_be_cancelled_without_worker(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        cancelled = self.repository.request_cancellation(
            job_id=str(job.id), now=self.now()
        )
        self.assertEqual(cancelled.status, "cancelled")
        self.assertIsNone(self.repository.claim_next(worker_id="worker-a", now=self.now()))

    def test_progress_is_persisted_without_overwriting_job_output(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        progress = {"phase": "synthesizing", "current_chunk": 2, "total_chunks": 5, "chunk_history": [{"text_chars": 20, "duration_seconds": 8.5}]}
        self.assertTrue(self.repository.update_progress(job_id=str(job.id), worker_id="worker-a", progress_data=progress))
        updated = self.repository.get(str(job.id))
        self.assertEqual(updated.progress, progress)
        self.assertIsNone(updated.output)

    def test_deleting_running_job_requests_cancellation_then_allows_removal(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        requested = self.repository.request_deletion(job_id=str(job.id), now=self.now())
        self.assertEqual(requested.status, "running")
        self.assertTrue(requested.deletion_requested)
        self.assertTrue(self.repository.cancellation_requested(job_id=str(job.id), worker_id="worker-a"))
        self.assertTrue(self.repository.cancel(job_id=str(job.id), worker_id="worker-a", now=self.now()))
        cancelled = self.repository.get(str(job.id))
        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(cancelled.deletion_requested)
        self.assertTrue(self.repository.delete_job(job_id=str(job.id)))
        self.assertIsNone(self.repository.get(str(job.id)))

    def test_running_job_cannot_be_deleted_before_worker_stops(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        requested = self.repository.request_deletion(job_id=str(job.id), now=self.now())
        self.assertEqual(requested.status, "running")
        self.assertTrue(requested.deletion_requested)
        self.assertTrue(self.repository.cancellation_requested(
            job_id=str(job.id), worker_id="worker-a"
        ))

        self.assertFalse(self.repository.delete_job(job_id=str(job.id)))
        self.assertIsNotNone(self.repository.get(str(job.id)))
        self.assertTrue(self.repository.cancel(
            job_id=str(job.id), worker_id="worker-a", now=self.now()
        ))
        self.assertTrue(self.repository.delete_job(job_id=str(job.id)))
        self.assertIsNone(self.repository.get(str(job.id)))

    def test_queued_job_deletion_removes_it_without_worker(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        requested = self.repository.request_deletion(job_id=str(job.id), now=self.now())
        self.assertEqual(requested.status, "cancelled")
        self.assertTrue(requested.deletion_requested)
        self.assertTrue(self.repository.delete_job(job_id=str(job.id)))
        self.assertIsNone(self.repository.get(str(job.id)))

    def test_missing_job_is_reported_as_cancellation_to_old_worker(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        self.assertTrue(self.repository.delete_job(job_id=str(job.id)))
        self.assertTrue(self.repository.cancellation_requested(
            job_id=str(job.id), worker_id="worker-a"
        ))

    def test_lost_worker_ownership_is_reported_as_cancellation(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        self.repository.request_cancellation(job_id=str(job.id), now=self.now())
        self.assertTrue(self.repository.cancellation_requested(
            job_id=str(job.id), worker_id="different-worker"
        ))
    def test_idempotency_key_returns_existing_job(self) -> None:
        first = self.repository.enqueue(
            kind="tts_generate", input={"text": "first"}, idempotency_key="request-1"
        )
        duplicate = self.repository.enqueue(
            kind="tts_generate", input={"text": "different"}, idempotency_key="request-1"
        )
        self.assertEqual(first.id, duplicate.id)
        self.assertEqual(duplicate.input["text"], "first")

    def test_manual_retry_resets_attempt_counter_and_lease(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={}, max_attempts=1)
        self.repository.claim_next(worker_id="worker-a", now=self.now())
        self.repository.fail(
            job_id=str(job.id), worker_id="worker-a",
            error_message="failed", now=self.now()
        )
        retried = self.repository.retry(job_id=str(job.id), now=self.now())
        self.assertEqual(retried.status, "queued")
        self.assertEqual(retried.attempt, 0)
        self.assertIsNone(retried.worker_id)
        claimed = self.repository.claim_next(worker_id="worker-b", now=self.now())
        self.assertEqual(claimed.attempt, 1)

    def test_two_workers_cannot_claim_the_same_job(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        barrier = Barrier(2)
        claims = []

        def claim(worker_id: str) -> None:
            barrier.wait()
            claims.append(self.repository.claim_next(
                worker_id=worker_id, now=self.now()
            ))

        threads = [
            Thread(target=claim, args=("worker-a",)),
            Thread(target=claim, args=("worker-b",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        claimed_ids = [item.id for item in claims if item is not None]
        self.assertEqual(claimed_ids, [job.id])

    def test_existing_database_is_migrated_without_losing_jobs(self) -> None:
        legacy_path = Path(self.temp_dir.name) / "legacy.db"
        now = self.now().isoformat()
        with sqlite3.connect(legacy_path) as conn:
            conn.execute(
                """
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    attempt INTEGER NOT NULL, max_attempts INTEGER NOT NULL,
                    input TEXT NOT NULL, output TEXT, error_message TEXT,
                    worker_id TEXT, idempotency_key TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    lease_heartbeat_at TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO jobs (
                    id, kind, status, attempt, max_attempts, input, created_at, updated_at
                ) VALUES (?, 'tts_generate', 'queued', 0, 3, '{}', ?, ?)
                """,
                ("00000000-0000-0000-0000-000000000001", now, now),
            )

        migrated = SqliteJobRepository(str(legacy_path))
        job = migrated.get("00000000-0000-0000-0000-000000000001")
        self.assertEqual(job.status, "queued")
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.finished_at)


if __name__ == "__main__":
    unittest.main()
