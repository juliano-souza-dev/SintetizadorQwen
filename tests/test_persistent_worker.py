from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from persistent_worker import PermanentJobError, PersistentWorker
from sqlite_job_repository import SqliteJobRepository

UTC = timezone.utc


class PersistentWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.repository = SqliteJobRepository(str(root / "jobs.db"))
        self.log_dir = root / "job-logs"
        self.root_logger = logging.getLogger()
        self.previous_root_level = self.root_logger.level
        self.root_logger.setLevel(logging.INFO)
        self.addCleanup(self.root_logger.setLevel, self.previous_root_level)

    def now(self) -> datetime:
        return datetime.now(UTC)

    def test_handler_exception_is_persisted_with_traceback_and_job_log(self) -> None:
        job = self.repository.enqueue(
            kind="tts_generate", input={"text": "private text"}, max_attempts=2
        )

        def fail(*_args):
            raise ValueError("synthetic model initialization failure")

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": fail},
            logger=logging.getLogger("test.persistent_worker.failure"),
            job_log_dir=self.log_dir,
        )
        self.assertTrue(worker.run_once())

        failed_attempt = self.repository.get(str(job.id))
        self.assertEqual(failed_attempt.status, "retryable")
        self.assertIn("Fase: handler_execution", failed_attempt.error_message)
        self.assertIn("ValueError: synthetic model initialization failure", failed_attempt.error_message)
        self.assertIn("Traceback completo:", failed_attempt.error_message)

        log_file = self.log_dir / f"{job.id}.log"
        self.assertTrue(log_file.is_file())
        log_text = log_file.read_text(encoding="utf-8")
        self.assertIn("JOB_STARTED", log_text)
        self.assertIn("JOB_HANDLER_BEGIN", log_text)
        self.assertIn("JOB_FAILED", log_text)
        self.assertNotIn("private text", log_text)

    def test_permanent_handler_error_does_not_retry(self) -> None:
        job = self.repository.enqueue(
            kind="tts_generate", input={}, max_attempts=3
        )

        def timeout(*_args):
            raise PermanentJobError("inference inactivity timeout")

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": timeout},
            logger=logging.getLogger("test.persistent_worker.permanent"),
            job_log_dir=self.log_dir,
        )
        self.assertTrue(worker.run_once())
        failed = self.repository.get(str(job.id))
        self.assertEqual(failed.status, "failed")
        self.assertIn("inference inactivity timeout", failed.error_message)
        self.assertIsNone(
            self.repository.claim_next(worker_id="test-worker", now=self.now())
        )

    def test_success_is_persisted_only_after_handler_returns(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})
        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": lambda *_: {"audio_path": "generated.wav"}},
            logger=logging.getLogger("test.persistent_worker.success"),
            job_log_dir=self.log_dir,
        )
        self.assertTrue(worker.run_once())
        succeeded = self.repository.get(str(job.id))
        self.assertEqual(succeeded.status, "succeeded")
        self.assertEqual(succeeded.output["audio_path"], "generated.wav")

    def test_handler_cannot_mark_job_succeeded_after_cancellation(self) -> None:
        job = self.repository.enqueue(kind="tts_generate", input={})

        def cancel_during_handler(current, execution):
            self.repository.request_cancellation(
                job_id=str(current.id), now=self.now()
            )
            execution.raise_if_cancelled()
            return {"result": "should not be persisted"}

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": cancel_during_handler},
            logger=logging.getLogger("test.persistent_worker.cancel"),
            job_log_dir=self.log_dir,
        )
        self.assertTrue(worker.run_once())
        self.assertEqual(self.repository.get(str(job.id)).status, "cancelled")

    def test_next_job_runs_after_regular_failure(self) -> None:
        failed_job = self.repository.enqueue(
            kind="tts_generate", input={}, max_attempts=1
        )
        next_job = self.repository.enqueue(kind="tts_generate", input={})
        calls = []

        def handler(job, _execution):
            calls.append(str(job.id))
            if str(job.id) == str(failed_job.id):
                raise ValueError("synthetic generation failure")
            return {"audio_path": "next-job.wav"}

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": handler},
            logger=logging.getLogger("test.persistent_worker.next_after_failure"),
            job_log_dir=self.log_dir,
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(self.repository.get(str(failed_job.id)).status, "failed")
        self.assertTrue(worker.run_once())
        self.assertEqual(calls, [str(failed_job.id), str(next_job.id)])
        self.assertEqual(self.repository.get(str(next_job.id)).status, "succeeded")

    def test_next_job_runs_after_permanent_failure(self) -> None:
        failed_job = self.repository.enqueue(kind="tts_generate", input={})
        next_job = self.repository.enqueue(kind="tts_generate", input={})
        calls = []

        def handler(job, _execution):
            calls.append(str(job.id))
            if str(job.id) == str(failed_job.id):
                raise PermanentJobError("synthetic permanent inference failure")
            return {"audio_path": "next-job.wav"}

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": handler},
            logger=logging.getLogger("test.persistent_worker.next_after_permanent_failure"),
            job_log_dir=self.log_dir,
        )

        self.assertTrue(worker.run_once())
        self.assertEqual(self.repository.get(str(failed_job.id)).status, "failed")
        self.assertTrue(worker.run_once())
        self.assertEqual(calls, [str(failed_job.id), str(next_job.id)])
        self.assertEqual(self.repository.get(str(next_job.id)).status, "succeeded")

    def test_next_job_runs_after_running_job_is_deleted(self) -> None:
        deleted_job = self.repository.enqueue(kind="tts_generate", input={})
        next_job = self.repository.enqueue(kind="tts_generate", input={})
        calls = []

        def handler(job, execution):
            calls.append(str(job.id))
            if str(job.id) == str(deleted_job.id):
                requested = self.repository.request_deletion(
                    job_id=str(job.id), now=self.now()
                )
                self.assertEqual(requested.status, "running")
                execution.raise_if_cancelled()
            return {"audio_path": "next-job.wav"}

        worker = PersistentWorker(
            self.repository,
            "test-worker",
            {"tts_generate": handler},
            logger=logging.getLogger("test.persistent_worker.next_after_deletion"),
            job_log_dir=self.log_dir,
        )

        self.assertTrue(worker.run_once())
        cancelled = self.repository.get(str(deleted_job.id))
        self.assertEqual(cancelled.status, "cancelled")
        self.assertTrue(cancelled.deletion_requested)
        self.assertTrue(self.repository.delete_job(job_id=str(deleted_job.id)))

        self.assertTrue(worker.run_once())
        self.assertEqual(calls, [str(deleted_job.id), str(next_job.id)])
        self.assertEqual(self.repository.get(str(next_job.id)).status, "succeeded")


if __name__ == "__main__":
    unittest.main()
