"""Automatic mode: watch a root folder and start a job for every complete set.

A "set" is any directory (the root itself or a sub-directory at any depth)
that contains FITS frames. It is considered complete when it holds at least
``frames_per_set`` frames and no file in it has changed for
``settle_seconds``. Polling is used on purpose: it works on network shares
and on Windows/Linux alike, where file-system events are unreliable.

Each processed directory is remembered in ``watch_state.json`` so it is not
processed twice; a directory that receives more frames later can be re-run
manually.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from astropy.io import fits

from .fields import _is_calibration_dir, create_field_jobs
from .fitsmeta import list_fits
from .instruments import InstrumentStore
from .jobs import JobRunner, JobStore
from .settings import Settings


class Watcher:
    def __init__(self, settings: Settings, jobs: JobStore, runner: JobRunner):
        self.settings = settings
        self.jobs = jobs
        self.runner = runner
        self.state_file = settings.root / "watch_state.json"
        self.state = self._load()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_scan: float | None = None
        self.pending: dict[str, dict] = {}       # dir -> {"count", "age_s"}
        self.messages: list[str] = []

    # ---- state -------------------------------------------------------
    def _load(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                pass
        return {"processed": {}}

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=1))

    # ---- control -----------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="synthtrack-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {"running": self.running, "last_scan": self.last_scan, "pending": self.pending,
                "processed": len(self.state["processed"]), "messages": self.messages[-20:],
                "config": self.settings.watch.__dict__}

    def forget(self, directory: str) -> None:
        self.state["processed"].pop(directory, None)
        self._save()

    # ---- loop --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception as e:  # noqa: BLE001
                self._msg(f"Chyba skenování: {e}")
            self._stop.wait(max(2, self.settings.watch.poll_seconds))

    def _msg(self, m: str) -> None:
        self.messages.append(time.strftime("%H:%M:%S ") + m)
        self.messages = self.messages[-200:]

    def scan_once(self) -> list[str]:
        cfg = self.settings.watch
        root = Path(cfg.root)
        self.last_scan = time.time()
        started = []
        if not cfg.root or not root.is_dir():
            self.pending = {}
            return started
        dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
        # never treat calibration libraries as light sets
        masters = [Path(i.masters_dir).resolve() for i in InstrumentStore(self.settings.instruments_dir).list()
                   if i.masters_dir]
        dirs = [d for d in dirs if not any(d.resolve() == m or m in d.resolve().parents for m in masters)]
        pending = {}
        now = time.time()
        for d in dirs:
            key = str(d)
            if key in self.state["processed"]:
                continue
            files = list_fits(d)
            if not files or _is_calibration_dir(files):
                continue
            newest = max(f.stat().st_mtime for f in files)
            age = now - newest
            pending[key] = {"count": len(files), "age_s": round(age)}
            if len(files) >= cfg.frames_per_set and age >= cfg.settle_seconds:
                inst_id = cfg.instrument or self._instrument_for(files[0])
                if not inst_id:
                    self._msg(f"{d}: žádný profil přístroje neodpovídá hlavičce – přeskočeno.")
                    self.state["processed"][key] = {"time": now, "job": None, "error": "no instrument"}
                    continue
                # pre-sorted fields (root/<field>/…) are separate folders -> separate jobs
                name = f"{root.name}/{d.relative_to(root).as_posix()}" if d != root else d.name
                inst = InstrumentStore(self.settings.instruments_dir).get(inst_id)
                jobs, fs, skipped = create_field_jobs(self.jobs, inst, key, name, source="auto",
                                                      split=self.settings.field_split,
                                                      tolerance_fov=self.settings.field_tolerance_fov)
                if fs.multiple:
                    self._msg(f"{d}: {len(fs.fields)} zorných polí – " + "; ".join(f.describe() for f in fs.fields))
                for m in skipped:
                    self._msg(f"{d}: {m}")
                for job in jobs:
                    for note in fs.notes + skipped:
                        self.jobs.log(job.id, note)
                    self.runner.submit(job)
                    self._msg(f"Spuštěna úloha {job.id} pro {d} – {job.name} ({job.n_frames or len(files)} snímků)")
                    started.append(job.id)
                self.state["processed"][key] = {"time": now, "job": jobs[0].id if jobs else None,
                                                "jobs": [j.id for j in jobs]}
                pending.pop(key, None)
        self.pending = pending
        self._save()
        return started

    def _instrument_for(self, fits_path: Path) -> str | None:
        try:
            hdr = fits.getheader(fits_path)
        except Exception:
            return None
        inst = InstrumentStore(self.settings.instruments_dir).find_for_header(hdr)
        return inst.id if inst else None
