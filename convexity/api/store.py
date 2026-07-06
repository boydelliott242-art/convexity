"""In-memory scan-job store with on-disk persistence of completed results.

The HTTP API runs scans as background tasks and tracks each one as a
:class:`~convexity.api.schemas.ScanJob` in a process-local, thread-safe store keyed
by job id. Completed :class:`~convexity.core.models.ScanResult` payloads are also
persisted as JSON under ``<settings.data_dir>/scans`` so the most recent scan
survives a restart and can be served by ``GET /api/scans/latest``.

Design notes
------------
* **Import-safe.** Importing this module performs no I/O and starts no scan: the
  store is a plain object created on demand. Disk is only touched when a scan
  completes (write) or when :meth:`ScanJobStore.load_persisted` is called
  explicitly (read) — typically from the app's startup hook.
* **Thread-safe.** Scans run on FastAPI's background-task threadpool, and the
  progress callback mutates a job from that worker thread while HTTP handler
  threads read it. All mutation and reads go through a single re-entrant lock so a
  poller never observes a torn update.
* **Bounded.** Only the most recent ``max_jobs`` jobs are retained in memory; the
  oldest completed/failed jobs are evicted first so a long-lived process does not
  grow without bound. Persisted result files are likewise pruned to ``max_files``.
* **Honest.** A failed scan is recorded as ``FAILED`` with its error message rather
  than being hidden; nothing is fabricated to make a scan look successful.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import uuid
from collections import OrderedDict
from typing import List, Optional, Tuple

from convexity.api.schemas import ScanJob, ScanProgress, ScanStatus
from convexity.core.config import Settings, get_settings
from convexity.core.logging import get_logger
from convexity.core.models import ScanParams, ScanResult

_log = get_logger(__name__)

# Subdirectory (under settings.data_dir) where completed scan results are stored.
_SCANS_SUBDIR = "scans"
# Filename prefix for persisted result JSON files.
_FILE_PREFIX = "scan-"
_FILE_SUFFIX = ".json"


def _utcnow() -> _dt.datetime:
    """Return a timezone-aware UTC ``now`` (single source for job timestamps)."""
    return _dt.datetime.now(_dt.timezone.utc)


def _fraction(done: int, total: int) -> float:
    """Compute a clamped 0..1 progress fraction, tolerating a zero/unknown total."""
    if total <= 0:
        return 0.0
    frac = float(done) / float(total)
    if frac < 0.0:
        return 0.0
    if frac > 1.0:
        return 1.0
    return frac


class ScanJobStore:
    """Thread-safe registry of scan jobs plus persistence of completed results.

    A single instance is created per application (see :func:`get_store`). It owns
    the lifecycle of every :class:`ScanJob`: creation, per-stage progress updates,
    successful completion (which also persists the result to disk) and failure.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        max_jobs: int = 100,
        max_files: int = 50,
    ) -> None:
        """Create an empty store bound to ``settings`` for its persistence path.

        Args:
            settings: Active settings; defaults to the cached process settings. Only
                ``data_dir`` is used, to locate the on-disk ``scans`` directory.
            max_jobs: Maximum number of in-memory jobs to retain (oldest evicted).
            max_files: Maximum number of persisted result files to keep on disk.
        """
        self._settings: Settings = settings if settings is not None else get_settings()
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, ScanJob] = OrderedDict()
        self._latest_completed_id: Optional[str] = None
        # The most recent ScanResult known to the store, even if its originating
        # job has been evicted or was loaded from disk on startup.
        self._latest_result: Optional[ScanResult] = None
        self._max_jobs = max(1, int(max_jobs))
        self._max_files = max(1, int(max_files))

    # ------------------------------------------------------------------ #
    # Paths                                                              #
    # ------------------------------------------------------------------ #
    @property
    def scans_dir(self) -> str:
        """Absolute path to the on-disk directory holding persisted results."""
        return os.path.abspath(os.path.join(self._settings.data_dir, _SCANS_SUBDIR))

    def _ensure_dir(self) -> str:
        """Create (if needed) and return the persisted-scans directory path."""
        path = self.scans_dir
        os.makedirs(path, exist_ok=True)
        return path

    def _file_for(self, job_id: str) -> str:
        """Return the on-disk path a given job's result is persisted to."""
        safe = "".join(ch for ch in job_id if ch.isalnum() or ch in "-_")
        return os.path.join(self.scans_dir, f"{_FILE_PREFIX}{safe}{_FILE_SUFFIX}")

    # ------------------------------------------------------------------ #
    # Job lifecycle                                                      #
    # ------------------------------------------------------------------ #
    def create(self, params: ScanParams) -> ScanJob:
        """Register a new ``PENDING`` job for ``params`` and return it.

        The job is assigned a fresh uuid4 id and inserted into the store; eviction
        of the oldest finished jobs runs afterwards to keep the store bounded.
        """
        job_id = uuid.uuid4().hex
        job = ScanJob(
            id=job_id,
            status=ScanStatus.PENDING,
            progress=ScanProgress(stage="queued", message="Scan queued."),
            created_at=_utcnow(),
        )
        with self._lock:
            self._jobs[job_id] = job
            self._jobs.move_to_end(job_id)
            self._evict_locked()
        # The scan parameters are not duplicated onto the lightweight job record:
        # the caller (the route) already holds ``params`` to drive the scan, and
        # they are echoed back inside the eventual ``ScanResult.params``.
        return job

    def mark_running(self, job_id: str) -> None:
        """Transition a job to ``RUNNING`` and stamp its start time."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = ScanStatus.RUNNING
            job.started_at = _utcnow()
            job.progress = ScanProgress(
                stage="starting", message="Scan starting…"
            )

    def update_progress(
        self,
        job_id: str,
        stage: str,
        done: int,
        total: int,
        message: str,
    ) -> None:
        """Record a progress tick for a running job.

        Designed to be passed (bound) as the pipeline's ``progress`` callback. It
        is best-effort: an unknown job id is ignored so a late tick from a finished
        scan can never raise.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress = ScanProgress(
                stage=stage,
                done=int(done),
                total=int(total),
                fraction=_fraction(int(done), int(total)),
                message=message,
            )

    def mark_completed(self, job_id: str, result: ScanResult) -> None:
        """Attach ``result`` to a job, mark it ``COMPLETED`` and persist to disk.

        Persistence failures are logged but never raised: a completed scan is still
        served from memory even if the disk write fails.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.status = ScanStatus.COMPLETED
                job.finished_at = _utcnow()
                job.result = result
                job.progress = ScanProgress(
                    stage="done",
                    done=1,
                    total=1,
                    fraction=1.0,
                    message="Scan complete.",
                )
            self._latest_completed_id = job_id
            self._latest_result = result
        # Persist outside the critical section's mutation logic but still safe:
        # the write only reads the immutable ``result`` and ``job_id``.
        self._persist(job_id, result)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job ``FAILED`` with a human-readable ``error`` message."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = ScanStatus.FAILED
            job.finished_at = _utcnow()
            job.error = error
            job.progress = ScanProgress(
                stage="failed", message=f"Scan failed: {error}"
            )

    # ------------------------------------------------------------------ #
    # Reads                                                              #
    # ------------------------------------------------------------------ #
    def get(self, job_id: str) -> Optional[ScanJob]:
        """Return the job for ``job_id`` (a shared reference), or ``None``."""
        with self._lock:
            return self._jobs.get(job_id)

    def latest_result(self) -> Optional[ScanResult]:
        """Return the most recent completed :class:`ScanResult`, if any.

        This survives in-memory eviction and reflects a result loaded from disk on
        startup, so ``GET /api/scans/latest`` works even after a restart.
        """
        with self._lock:
            return self._latest_result

    def active_count(self) -> int:
        """Number of jobs currently ``PENDING`` or ``RUNNING``."""
        with self._lock:
            return sum(
                1
                for j in self._jobs.values()
                if j.status in (ScanStatus.PENDING, ScanStatus.RUNNING)
            )

    def active_jobs(self) -> List[ScanJob]:
        """All jobs currently ``PENDING`` or ``RUNNING``, newest first.

        Lets a reconnecting dashboard rediscover an in-flight scan (e.g. after a
        page reload or a browser that gave up watching) without knowing its id.
        """
        with self._lock:
            active = [
                j
                for j in self._jobs.values()
                if j.status in (ScanStatus.PENDING, ScanStatus.RUNNING)
            ]
        return list(reversed(active))

    def total_count(self) -> int:
        """Total number of jobs currently retained in memory."""
        with self._lock:
            return len(self._jobs)

    def persisted_count(self) -> int:
        """Number of persisted result files currently on disk (0 if none)."""
        try:
            return len(self._list_files())
        except OSError:
            return 0

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def _persist(self, job_id: str, result: ScanResult) -> None:
        """Write ``result`` to disk as JSON, then prune old files. Best-effort."""
        try:
            self._ensure_dir()
            path = self._file_for(job_id)
            payload = result.model_dump(mode="json")
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            self._prune_files()
        except Exception as exc:  # pragma: no cover - persistence is best-effort
            _log.warning("could not persist scan result %s: %s", job_id, exc)

    def _list_files(self) -> List[str]:
        """Return persisted result file paths, newest first by modification time."""
        directory = self.scans_dir
        if not os.path.isdir(directory):
            return []
        entries: List[Tuple[float, str]] = []
        for name in os.listdir(directory):
            if name.startswith(_FILE_PREFIX) and name.endswith(_FILE_SUFFIX):
                full = os.path.join(directory, name)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:  # pragma: no cover - race with concurrent prune
                    continue
                entries.append((mtime, full))
        entries.sort(key=lambda pair: pair[0], reverse=True)
        return [full for _mtime, full in entries]

    def _prune_files(self) -> None:
        """Delete the oldest persisted files beyond ``max_files``. Best-effort."""
        files = self._list_files()
        for stale in files[self._max_files:]:
            try:
                os.remove(stale)
            except OSError:  # pragma: no cover - file may already be gone
                pass

    def load_persisted(self) -> Optional[ScanResult]:
        """Load the most recent persisted result into memory, returning it.

        Intended to be called once on application startup so ``latest`` is
        populated before any new scan runs. A missing directory, an empty
        directory, or a corrupt file is handled gracefully (logged, ``None``
        returned) — the API stays import- and startup-safe with no prior scan.
        """
        try:
            files = self._list_files()
        except OSError as exc:  # pragma: no cover - defensive
            _log.warning("could not list persisted scans: %s", exc)
            return None

        for path in files:
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                result = ScanResult.model_validate(payload)
            except Exception as exc:
                _log.warning("skipping unreadable persisted scan %s: %s", path, exc)
                continue
            with self._lock:
                self._latest_result = result
            _log.info("loaded most recent persisted scan from %s", path)
            return result

        _log.info("no persisted scans found under %s", self.scans_dir)
        return None

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #
    def _evict_locked(self) -> None:
        """Evict oldest *finished* jobs once the store exceeds ``max_jobs``.

        Must be called with ``self._lock`` held. Active (pending/running) jobs are
        never evicted; if every job is active and the cap is exceeded the store is
        allowed to grow rather than dropping in-flight work.
        """
        while len(self._jobs) > self._max_jobs:
            evicted = False
            for jid, job in list(self._jobs.items()):
                if job.status in (ScanStatus.COMPLETED, ScanStatus.FAILED):
                    del self._jobs[jid]
                    evicted = True
                    break
            if not evicted:
                break


# A process-wide default store, created lazily so importing this module has no
# side effects (no directory creation, no disk read) until the app actually uses it.
_DEFAULT_STORE: Optional[ScanJobStore] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def get_store() -> ScanJobStore:
    """Return the process-wide :class:`ScanJobStore`, creating it on first use.

    The store is created lazily and cached so every route and the startup hook
    share one registry. Creation does no I/O, keeping import and first-call safe.
    """
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_STORE_LOCK:
            if _DEFAULT_STORE is None:
                _DEFAULT_STORE = ScanJobStore()
    return _DEFAULT_STORE


__all__ = ["ScanJobStore", "get_store"]
