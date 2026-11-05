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


if __name__ == "__main__":
    unittest.main()
