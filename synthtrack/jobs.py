"""Jobs: persistent records on disk + a single background worker (GPU is serial)."""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Job:
    id: str
    name: str
    instrument_id: str
    lights_dir: str
    source: str = "manual"            # manual | auto
    status: str = "queued"            # queued | running | done | failed | cancelled
    stage: str = ""
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    params: dict = field(default_factory=dict)      # per-job overrides of SearchSettings
    n_frames: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    files: list[str] = field(default_factory=list)  # explicit frames (one field of a multi-field folder)
    field: dict = field(default_factory=dict)       # field of view this job processes (fields.py)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # decoded large artefacts (candidates.json: tens of MB, 1.5 s to parse and 3 s to write with
        # indentation) – a GUI decision must not re-read the file; validated by mtime + size
        self._cache: dict[Path, tuple[int, int, object]] = {}

    def job_dir(self, job_id: str) -> Path:
        return self.dir / job_id

    def create(self, name: str, instrument_id: str, lights_dir: str, source: str = "manual",
               params: dict | None = None) -> Job:
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job = Job(id=jid, name=name, instrument_id=instrument_id, lights_dir=lights_dir,
                  source=source, params=params or {})
        self.job_dir(jid).mkdir(parents=True, exist_ok=True)
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        with self._lock:
            _atomic_write(self.job_dir(job.id) / "job.json", json.dumps(job.to_dict(), indent=1))

    def get(self, job_id: str) -> Job:
        p = self.job_dir(job_id) / "job.json"
        if not p.exists():
            raise KeyError(job_id)
        with self._lock:
            d = json.loads(_read_text(p))
        known = Job.__dataclass_fields__
        return Job(**{k: v for k, v in d.items() if k in known})

    def list(self) -> list[Job]:
        out = []
        for p in self.dir.glob("*/job.json"):
            try:
                out.append(self.get(p.parent.name))
            except Exception:
                continue
        out.sort(key=lambda j: j.created, reverse=True)
        return out

    def delete(self, job_id: str) -> None:
        """Remove the job directory; raise if Windows keeps something locked."""
        import gc
        d = self.job_dir(job_id)
        with self._lock:
            for k in range(10):
                shutil.rmtree(d, ignore_errors=True)
                if not d.exists():
                    return
                gc.collect()
                time.sleep(0.1 * (k + 1))
        left = [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]
        raise PermissionError("Úlohu nelze úplně smazat, soubory drží jiný proces: " + ", ".join(left[:5]))

    def cleanup_work(self, job_id: str) -> None:
        """Drop the scratch directory of a finished/failed/cancelled job."""
        import os
        if os.environ.get("SYNTHTRACK_KEEP_WORK") or (self.job_dir(job_id) / "work" / "KEEP").exists():
            return                     # diagnostics: the pipeline marks work/ to be kept (keep_work)
        shutil.rmtree(self.job_dir(job_id) / "work", ignore_errors=True)

    # ---- artefacts ---------------------------------------------------
    def write_json(self, job_id: str, name: str, data) -> None:
        text = _dumps(data)
        with self._lock:
            p = self.job_dir(job_id) / name
            _atomic_write(p, text)
            self._remember(p, data)

    def read_json(self, job_id: str, name: str, default=None):
        """Decoded artefact. Large files are cached: the result is shared – change it only
        through ``update_json`` / ``write_json``."""
        p = self.job_dir(job_id) / name
        with self._lock:
            hit = self._cached(p)
            if hit is not None:
                return hit
            if not p.exists():
                return default
            data = json.loads(_read_text(p))
            self._remember(p, data)
            return data

    def update_json(self, job_id: str, name: str, update, default=None):
        """Atomically read, mutate and replace a JSON artefact.

        ``update`` receives the decoded value and returns the value to write.
        Keeping the lock across the whole transaction prevents two GUI requests
        from silently overwriting each other's candidate decisions.
        """
        p = self.job_dir(job_id) / name
        with self._lock:
            data = self._cached(p)
            if data is None:
                data = json.loads(_read_text(p)) if p.exists() else default
            data = update(data)
            _atomic_write(p, _dumps(data))
            self._remember(p, data)
            return data

    def _cached(self, p: Path):
        e = self._cache.get(p)
        if e is None:
            return None
        try:
            st = p.stat()
        except OSError:
            self._cache.pop(p, None)
            return None
        if (st.st_mtime_ns, st.st_size) != e[:2]:
            self._cache.pop(p, None)
            return None
        return e[2]

    def _remember(self, p: Path, data) -> None:
        try:
            st = p.stat()
        except OSError:
            return
        if st.st_size < 1_000_000:          # small files: reading is cheap
            self._cache.pop(p, None)
            return
        self._cache.pop(p, None)
        self._cache[p] = (st.st_mtime_ns, st.st_size, data)
        while len(self._cache) > 3:          # the jobs being reviewed
            self._cache.pop(next(iter(self._cache)))

    def log(self, job_id: str, msg: str) -> None:
        with self._lock, open(self.job_dir(job_id) / "log.txt", "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S ") + msg + "\n")

    def read_log(self, job_id: str) -> str:
        p = self.job_dir(job_id) / "log.txt"
        if not p.exists():
            return ""
        with self._lock:
            return _read_text(p)


def _retry_io(fn, tries: int = 20):
    """Windows: a file being read by another thread/process (web UI, antivirus,
    search indexer) cannot be replaced or opened for a moment -> retry briefly."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(min(0.02 * (k + 1), 0.25))


def _atomic_write(p: Path, text: str) -> None:
    tmp = p.with_name(p.name + f".{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    try:
        _retry_io(lambda: tmp.replace(p))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _dumps(data) -> str:
    # compact: 5× faster than indent=1 and half the size for 7 600 candidates (88P: 3.2 → 0.7 s)
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _read_text(p: Path) -> str:
    return _retry_io(lambda: p.read_text(encoding="utf-8"))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not serialisable: {type(o)}")


class Cancelled(Exception):
    pass


class JobRunner:
    """One worker thread; jobs run strictly one after another."""

    def __init__(self, store: JobStore, run_fn):
        self.store = store
        self.run_fn = run_fn          # run_fn(job, ctx) – see pipeline.run_job
        self.q: queue.Queue[str] = queue.Queue()
        self.current: str | None = None
        self._cancel: set[str] = set()
        self._thread = threading.Thread(target=self._loop, name="synthtrack-worker", daemon=True)
        self._thread.start()
        # jobs interrupted by a restart go back to the queue
        for j in store.list():
            if j.status in ("queued", "running"):
                j.status, j.stage, j.progress = "queued", "", 0.0
                store.save(j)
                self.q.put(j.id)

    def submit(self, job: Job) -> None:
        self.q.put(job.id)

    def cancel(self, job_id: str) -> None:
        self._cancel.add(job_id)
        try:
            j = self.store.get(job_id)
            if j.status == "queued":
                j.status = "cancelled"
                self.store.save(j)
        except KeyError:
            pass

    def _loop(self):
        while True:
            jid = self.q.get()
            try:
                job = self.store.get(jid)
            except KeyError:
                continue
            if job.status != "queued" or jid in self._cancel:
                continue
            self.current = jid
            ctx = RunContext(self, job)
            job.status, job.started, job.error = "running", time.time(), None
            self.store.save(job)
            try:
                self.run_fn(job, ctx)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
                self.store.log(jid, "Zrušeno uživatelem.")
            except Exception as e:  # noqa: BLE001 – job failures must not kill the worker
                job.status = "failed"
                job.error = str(e)
                self.store.log(jid, "CHYBA: " + str(e))
                self.store.log(jid, traceback.format_exc())
            finally:
                # always release memory-mapped scratch arrays and delete them –
                # also after an error or cancel (Windows keeps mapped files locked)
                ctx.release_memmaps()
                import gc
                gc.collect()
                self.store.cleanup_work(jid)
            job.finished = time.time()
            job.progress = 1.0 if job.status == "done" else job.progress
            self.store.save(job)
            self.current = None


class RunContext:
    """Passed to the pipeline: logging, progress, cancellation."""

    def __init__(self, runner: JobRunner | None, job: Job, store: JobStore | None = None):
        self.runner = runner
        self.job = job
        self.store = store or runner.store
        self._last_save = 0.0
        self._memmaps: list = []

    def track_memmap(self, mm):
        """Register a scratch memmap so it is closed whenever the job ends."""
        self._memmaps.append(mm)
        return mm

    def release_memmaps(self) -> None:
        for mm in self._memmaps:
            base = mm
            while base is not None and not hasattr(base, "_mmap"):
                base = getattr(base, "base", None)
            m = getattr(base, "_mmap", None)
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        self._memmaps.clear()

    def log(self, msg: str) -> None:
        self.store.log(self.job.id, msg)

    def warn(self, msg: str) -> None:
        self.job.warnings.append(msg)
        self.log("VAROVÁNÍ: " + msg)

    def stage(self, name: str, progress: float) -> None:
        self.job.stage = name
        self.job.progress = progress
        self.store.save(self.job)
        self.log(f"== {name}")
        self.check()

    def progress(self, p: float) -> None:
        self.job.progress = p
        now = time.time()
        if now - self._last_save > 1.0:
            self.store.save(self.job)
            self._last_save = now

    def check(self) -> None:
        if self.runner and self.job.id in self.runner._cancel:
            raise Cancelled()

    def should_stop(self) -> bool:
        return bool(self.runner and self.job.id in self.runner._cancel)
