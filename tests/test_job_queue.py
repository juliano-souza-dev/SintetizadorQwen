import time
import unittest
from threading import Lock

from job_queue import JobQueue


class JobQueueTests(unittest.TestCase):
    def wait_for_terminal(self, manager, job_ids, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jobs = [manager.get(job_id) for job_id in job_ids]
            if all(job and job["status"] in {"completed", "failed"} for job in jobs):
                return jobs
            time.sleep(0.02)
        self.fail("Os Jobs não chegaram a um estado final dentro do prazo.")

    def test_processes_jobs_sequentially_and_returns_result(self):
        processed = []
        lock = Lock()

        def processor(payload):
            time.sleep(0.05)
            with lock:
                processed.append(payload["value"])
            return f"result-{payload['value']}"

        manager = JobQueue(processor)
        try:
            first = manager.submit({"value": 1})
            second = manager.submit({"value": 2})
            jobs = self.wait_for_terminal(manager, [first, second])

            self.assertEqual(processed, [1, 2])
            self.assertEqual([job["status"] for job in jobs], ["completed", "completed"])
            self.assertEqual(jobs[0]["result"], "result-1")
            self.assertEqual(jobs[1]["result"], "result-2")
            self.assertNotIn("payload", manager.get(first))
        finally:
            manager.shutdown(wait=True)

    def test_processor_exception_marks_job_as_failed(self):
        def processor(_payload):
            raise RuntimeError("falha simulada")

        manager = JobQueue(processor)
        try:
            job_id = manager.submit({"value": "erro"})
            job = self.wait_for_terminal(manager, [job_id])[0]

            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "falha simulada")
            self.assertIsNotNone(job["finished_at"])
        finally:
            manager.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
