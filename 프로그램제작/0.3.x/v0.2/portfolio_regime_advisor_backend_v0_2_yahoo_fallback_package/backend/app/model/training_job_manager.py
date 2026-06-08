from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class TrainingJob:
    job_id: str
    status: str = "PENDING"
    progress: float = 0.0
    message: str = "Queued"
    result: Optional[dict] = None


class TrainingJobManager:
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create_job(self, target: Callable[[TrainingJob], dict]) -> TrainingJob:
        job = TrainingJob(job_id=str(uuid.uuid4()))
        with self._lock:
            self.jobs[job.job_id] = job

        def runner():
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

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)
