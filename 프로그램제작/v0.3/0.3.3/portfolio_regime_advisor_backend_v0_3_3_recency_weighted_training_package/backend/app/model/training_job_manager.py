from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass
class TrainingJob:
    job_id: str
    status: str = "PENDING"
    progress: float = 0.0
    message: str = "Queued"
    result: Optional[dict] = None


class TrainingJobManager:
    """In-memory training job manager for MVP.

    Job state is intentionally process-local and disappears after server restart.
    Model artifacts/registry entries are still persisted by TrainingService.
    """

    def __init__(self, max_concurrent_jobs: int = 1):
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent_jobs)

    def create_job(self, target: Callable[[TrainingJob], dict]) -> TrainingJob:
        job = TrainingJob(job_id=str(uuid.uuid4()))
        with self._lock:
            self.jobs[job.job_id] = job

        def runner():
            acquired = self._semaphore.acquire(blocking=False)
            if not acquired:
                job.status = "FAILED"
                job.message = "Another training job is already running. Retry after it completes."
                job.progress = 0.0
                return
            try:
                job.status = "RUNNING"
                job.message = "Training started"
                job.progress = 0.05
                result = target(job)
                job.result = result
                job.progress = 1.0
                job.status = "COMPLETED"
                job.message = "Training completed"
            except Exception as exc:
                job.status = "FAILED"
                job.message = str(exc)
                job.progress = min(job.progress, 0.99)
            finally:
                self._semaphore.release()

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)
