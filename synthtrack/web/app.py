"""Local web GUI + REST API (FastAPI). Start with ``synthtrack serve``."""
from __future__ import annotations

import json
import os
import string
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .. import classifier as CL
from ..confidence import annotate as annotate_confidence
from ..astrometry import find_astap
from ..export import export_csv, export_mpc80
from ..fields import create_tree_jobs, find_light_dirs, rel_name, split_fields
from ..fitsmeta import frame_groups, list_fits, read_meta
from ..gpu import get_backend, live_status
from ..instruments import InstrumentStore, instrument_from_dict
from ..jobs import JobRunner, JobStore
from ..masters import MasterLibrary
from ..mpcdata import COMETS_NAME, MPCORB_NAME, MpcStore
from ..pipeline import run_job
from ..settings import Settings
from ..watcher import Watcher

STATIC = Path(__file__).parent / "static"


def _np_default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def create_app(settings: Settings, start_worker: bool = True) -> FastAPI:
    app = FastAPI(title="SynthTrack", version=__version__)
    instruments = InstrumentStore(settings.instruments_dir)
    jobs = JobStore(settings.jobs_dir)
    runner = JobRunner(jobs, lambda job, ctx: run_job(job, ctx, settings)) if start_worker else None
    watcher = Watcher(settings, jobs, runner) if runner else None
    mpc = MpcStore(settings.catalogs_dir)
    cat_task: dict = {"running": False, "kind": None, "done": 0, "total": None, "error": None, "message": ""}

    def _eph_name():
        from ..ephem import load_ephemeris
        try:
            return load_ephemeris(settings.ephemeris, [settings.catalogs_dir]).name
        except Exception as e:  # noqa: BLE001
            return f"chyba: {e}"

    if watcher and settings.watch.enabled:
        watcher.start()

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def status():
        be = get_backend(settings.gpu.backend, settings.gpu.device, settings.gpu.memory_fraction)
        cats = {}
        for k in (MPCORB_NAME, COMETS_NAME):
            st = mpc.status(k).to_dict()
            st["stale"] = st["age_days"] is not None and st["age_days"] > settings.mpcorb_max_age_days
            cats[k] = st
        return {
            "version": __version__,
            "backend": be.info(),
            "gpus": live_status(),
            "gpu_memory_fraction": settings.gpu.memory_fraction,
            "catalogs": cats,
            "catalog_task": cat_task,
            "astap": find_astap(settings.astap.executable),
            "ephemeris": _eph_name(),
            "current_job": runner.current if runner else None,
            "watch": watcher.status() if watcher else None,
            "online": settings.online.__dict__,
            "ucac4": _ucac4_status(),
        }

    def _ucac4_status():
        from ..ucac4 import find_root
        if not settings.ucac4_dir:
            return {"set": False, "ok": False, "path": ""}
        root = find_root(settings.ucac4_dir)
        n = len([f for f in root.iterdir() if f.name.lower().startswith("z")]) if root else 0
        return {"set": True, "ok": root is not None, "path": str(root or settings.ucac4_dir), "zones": n}

    # ---------------------------------------------------------------- settings
    @app.get("/api/settings")
    def get_settings():
        return settings.to_dict()

    @app.put("/api/settings")
    def put_settings(data: dict = Body(...)):
        settings.update(data)
        settings.save()
        return settings.to_dict()

    # ------------------------------------------------------------- instruments
    @app.get("/api/instruments")
    def list_instruments():
        return [i.summary() for i in instruments.list()]

    @app.post("/api/instruments")
    def save_instrument(data: dict = Body(...)):
        try:
            inst = instrument_from_dict(data)
        except (TypeError, ValueError) as e:
            raise HTTPException(400, str(e))
        return instruments.save(inst).summary()

    @app.delete("/api/instruments/{inst_id}")
    def delete_instrument(inst_id: str):
        instruments.delete(inst_id)
        return {"ok": True}

    def _inst(inst_id):
        try:
            return instruments.get(inst_id)
        except KeyError:
            raise HTTPException(404, f"Přístroj {inst_id} neexistuje")

    @app.get("/api/instruments/{inst_id}/masters")
    def masters(inst_id: str):
        inst = _inst(inst_id)
        lib = MasterLibrary(inst, settings.instruments_dir / f"{inst.id}.masters.json")
        return [{"type": e.type, **e.meta.to_dict()} for e in lib.scan()]

    @app.post("/api/instruments/{inst_id}/preview")
    def preview(inst_id: str, data: dict = Body(...)):
        """Which masters would be used for the frames in ``lights_dir``."""
        inst = _inst(inst_id)
        root = data.get("lights_dir", "")
        dirs = find_light_dirs(root, _master_dirs()) if root else []
        if not dirs:
            return {"frames": 0, "selection": None}
        lib = MasterLibrary(inst, settings.instruments_dir / f"{inst.id}.masters.json")
        lib.scan()

        def split_of(files):
            return split_fields(files, inst, settings.field_tolerance_fov).to_dict() if settings.field_split else None

        if len(dirs) == 1 and rel_name(root, dirs[0]) == "":
            files = list_fits(dirs[0])
            meta = read_meta(files[len(files) // 2], inst)
            return {"frames": len(files), "meta": meta.to_dict(), "fields": split_of(files),
                    "groups": frame_groups(files, inst),
                    "selection": {k: v.to_dict() for k, v in lib.select_all(meta).items()}}
        # pre-sorted fields: one sub-folder = one field (each may still be split further)
        out, metas = [], []
        for d in dirs:
            files = list_fits(d)
            m = read_meta(files[len(files) // 2], inst)
            metas.append(m)
            out.append({"path": str(d), "rel": rel_name(root, d) or ".", "frames": len(files),
                        "exptime": m.exptime, "filter": m.filter, "fields": split_of(files)})
        meta = metas[0]
        for d, m in zip(out, metas):       # masters per folder (they may differ in exposure / filter)
            sel = lib.select_all(m)
            bad = [f"{v.kind}: {v.reason}" for k, v in sel.items() if k != "bias" and not v.ok]
            d["masters_ok"], d["masters_note"] = not bad, "; ".join(bad)
        allfiles = [f for d in dirs for f in list_fits(d)]
        return {"frames": sum(d["frames"] for d in out), "meta": meta.to_dict(), "fields": None, "dirs": out,
                "groups": frame_groups(allfiles, inst),
                "selection": {k: v.to_dict() for k, v in lib.select_all(meta).items()}}

    @app.post("/api/identify")
    def identify(data: dict = Body(...)):
        """Which instrument profile the frames in ``lights_dir`` belong to (header rules)."""
        root = data.get("lights_dir", "")
        dirs = find_light_dirs(root, _master_dirs()) if root else []
        if not dirs:
            return {"instrument_id": None, "reason": "v adresáři nejsou FITS snímky"}
        res = []
        for d in dirs[:20]:
            inst, why = instruments.identify_files(list_fits(d))
            res.append((inst.id if inst else None, why))
        bad = [why for i, why in res if i is None]
        if bad:
            return {"instrument_id": None, "reason": bad[0]}
        ids = sorted({i for i, _ in res})
        if len(ids) > 1:
            return {"instrument_id": None, "reason": "podadresáře jsou z různých přístrojů: " + ", ".join(ids)}
        return {"instrument_id": ids[0], "reason": res[0][1]}

    def _master_dirs():
        return [i.masters_dir for i in instruments.list() if i.masters_dir]

    # ------------------------------------------------------------------ browse
    @app.get("/api/browse")
    def browse(path: str = ""):
        if not path:
            if os.name == "nt":
                roots = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            else:
                roots = ["/", str(Path.home())]
            return {"path": "", "parent": None, "dirs": [{"name": r, "path": r, "fits": 0} for r in roots],
                    "fits": 0}
        p = Path(path)
        if not p.is_dir():
            raise HTTPException(400, f"{path} není adresář")
        dirs = []
        try:
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if c.is_dir() and not c.name.startswith("."):
                    dirs.append({"name": c.name, "path": str(c)})
        except PermissionError:
            pass
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "dirs": dirs, "fits": len(list_fits(p))}

    # -------------------------------------------------------------------- jobs
    def _job(job_id):
        try:
            return jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, "Úloha neexistuje")

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_dict() for j in jobs.list()]

    @app.post("/api/jobs")
    def create_job(data: dict = Body(...)):
        _inst(data["instrument_id"])
        lights = data["lights_dir"]
        if not find_light_dirs(lights, _master_dirs()):
            raise HTTPException(400, f"V {lights} ani v jeho podadresářích nejsou žádné FITS snímky")
        name = data.get("name") or f"{Path(lights).parent.name}/{Path(lights).name}"
        split = bool(data.get("split", settings.field_split))
        select = data.get("fields")          # 1-based field indices chosen in the GUI (None = all)
        created, splits, skipped = create_tree_jobs(
            jobs, _inst(data["instrument_id"]), lights, name, "manual", data.get("params") or {},
            split=split, tolerance_fov=settings.field_tolerance_fov, dirs=data.get("dirs"),
            select=[int(i) for i in select] if select is not None else None, exclude=_master_dirs())
        if not created:
            raise HTTPException(400, "Nic ke zpracování: " + (" ".join(skipped) or "nevybrán žádný adresář."))
        notes = {}
        for d, fs in splits:
            notes[str(d)] = fs.notes
        for j in created:
            for note in notes.get(str(Path(j.lights_dir)), []) + skipped:
                jobs.log(j.id, note)
            if runner:
                runner.submit(j)
        return {"jobs": [j.to_dict() for j in created], "skipped": skipped,
                "fields": splits[0][1].to_dict() if len(splits) == 1 else None,
                "dirs": [{"path": str(d), "fields": fs.to_dict()} for d, fs in splits]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _job(job_id).to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        if runner and runner.current == job_id:
            raise HTTPException(409, "Úloha právě běží – nejdřív ji zrušte")
        try:
            CL.archive_job(settings.root, jobs.job_dir(job_id))    # labels stay for the classifier
        except Exception:  # noqa: BLE001 – never block deleting
            pass
        try:
            jobs.delete(job_id)
        except PermissionError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.post("/api/jobs/delete-finished")
    def delete_finished():
        n = 0
        for j in jobs.list():
            if j.status in ("done", "failed", "cancelled") and not (runner and runner.current == j.id):
                try:
                    CL.archive_job(settings.root, jobs.job_dir(j.id))
                except Exception:  # noqa: BLE001
                    pass
                try:
                    jobs.delete(j.id)
                    n += 1
                except PermissionError:
                    pass
        return {"deleted": n}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if runner:
            runner.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(job_id: str, data: dict = Body(default={})):
        old = _job(job_id)
        params = {**old.params, **(data.get("params") or {})}
        job = jobs.create(old.name, data.get("instrument_id", old.instrument_id), old.lights_dir, old.source, params)
        if old.files or old.field:          # same field of view as the original job
            job.files, job.field, job.n_frames = list(old.files), dict(old.field), old.n_frames
            jobs.save(job)
        if runner:
            runner.submit(job)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/candidates")
    def candidates(job_id: str):
        _job(job_id)
        # computed on every read: works for older jobs and follows a retrained classifier
        data = annotate_confidence(jobs.read_json(job_id, "candidates.json", []))
        # serialised directly: FastAPI's generic encoder needs seconds for thousands of candidates
        return Response(json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=_np_default),
                        media_type="application/json")

    @app.patch("/api/jobs/{job_id}/candidates/{cid}")
    def update_candidate(job_id: str, cid: int, data: dict = Body(...)):
        changed = []

        def mutate(cands):
            for c in cands:
                if c["id"] != cid:
                    continue
                for k in ("status", "trksub", "note"):
                    if k in data:
                        c[k] = data[k]
                if "reviewed" in data:            # undo restores the previous state exactly
                    c["reviewed"] = bool(data["reviewed"])
                elif "status" in data:
                    c["reviewed"] = c["status"] != "unreviewed"   # a human decision (classifier weight)
                changed.append(c)
                return cands
            raise KeyError(cid)

        try:
            jobs.update_json(job_id, "candidates.json", mutate, [])
        except KeyError:
            raise HTTPException(404, "Kandidát neexistuje")
        return annotate_confidence(changed)[0]

    # ------------------------------------------------------------ sky cut-outs
    @app.get("/api/jobs/{job_id}/candidates/{cid}/sky")
    def sky_cutout(job_id: str, cid: int, survey: str = "DSS2c", fov: float | None = None):
        from fastapi.responses import Response
        from .. import sky as SKY
        if not settings.online.sky_images:
            raise HTTPException(403, "Náhledy z přehlídek (DSS) jsou vypnuté – zapněte je v Nastavení: "
                                     "„Online: náhledy DSS“.")
        job = _job(job_id)
        c = next((x for x in jobs.read_json(job_id, "candidates.json", []) if x["id"] == cid), None)
        if c is None:
            raise HTTPException(404, "Kandidát neexistuje")
        if c.get("ra") is None:
            raise HTTPException(400, "Kandidát nemá souřadnice (snímky bez WCS) – výřez nelze stáhnout.")
        if survey not in SKY.SURVEYS:
            survey = "DSS2c"
        if not fov:
            size = (c.get("images") or {}).get("frames", {}).get("size") or 64
            fov = size * float((job.summary or {}).get("pixel_scale") or 1.0) / 3600.0
        fov = float(min(max(fov, 0.01), 2.0))
        cache = jobs.job_dir(job_id) / "img" / f"sky3_{cid}_{survey}_{int(round(fov * 3600))}.img"   # sky3: STScI north-up (16. 11.), colour only from CDS (17. 11.)
        meta = cache.with_suffix(".txt")
        if cache.exists() and meta.exists():
            ct, src = meta.read_text(encoding="utf-8").split("\n", 1)
            return Response(cache.read_bytes(), media_type=ct, headers={"X-Sky-Source": src.strip()})
        try:
            data, ct, src = SKY.fetch(c["ra"], c["dec"], fov, survey)
        except RuntimeError as e:
            raise HTTPException(502, str(e))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        meta.write_text(f"{ct}\n{src}", encoding="utf-8")
        return Response(data, media_type=ct, headers={"X-Sky-Source": src})

    # ------------------------------------------------------------ VSX variables in the DSS view
    _vsx_cache: dict = {}

    def _vsx_table():
        """ASTAP variable_stars_13.csv (AAVSO VSX to ~18 mag), loaded once."""
        astap_exe = find_astap(settings.astap.executable)
        cands = []
        if settings.astap.database_dir:
            cands.append(Path(settings.astap.database_dir))
        if astap_exe:
            cands.append(Path(astap_exe).parent)
        path = next((d / "variable_stars_13.csv" for d in cands if (d / "variable_stars_13.csv").exists()), None)
        if path is None:
            return None
        if _vsx_cache.get("path") != str(path):
            from .. import twilight as TW
            _vsx_cache.clear()
            _vsx_cache.update(path=str(path), data=TW.load_variables(path, max_mag=None))
        return _vsx_cache["data"]

    @app.get("/api/jobs/{job_id}/candidates/{cid}/vsx")
    def vsx_near(job_id: str, cid: int, fov: float = 0.1):
        """Variable stars in the DSS cut-out: AAVSO VSX → VizieR B/vsx mirrors → ASTAP (offline).
        Each with name, type, range, period, link and offsets east/north (deg); cached per cut-out."""
        import math
        from .. import vsx as VSX
        c = next((x for x in jobs.read_json(job_id, "candidates.json", []) if x["id"] == cid), None)
        if c is None or c.get("ra") is None:
            raise HTTPException(404, "Kandidát nemá souřadnice")
        fov = float(min(max(fov, 0.005), 2.0))
        half = 0.75 * fov
        ra0, de0 = float(c["ra"]), float(c["dec"])
        cache = jobs.job_dir(job_id) / "img" / f"vsx_{cid}_{int(round(fov * 3600))}.json"
        if cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except ValueError:
                pass
        note = None
        if settings.online.sky_images:
            try:
                stars, src = VSX.query(ra0, de0, half * math.sqrt(2.0))
                res = {"stars": VSX.offsets(stars, ra0, de0), "source": src}
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(res), encoding="utf-8")
                return res
            except RuntimeError as e:
                note = f"online nedostupné ({e}) – použit ASTAP"
        var = _vsx_table()
        if var is None:
            raise HTTPException(404, "VSX ani VizieR neodpovídají a katalog proměnných ASTAP "
                                     "(variable_stars_13.csv) nenalezen – nastavte složku ASTAP v Nastavení.")
        stars = VSX.from_astap(var, ra0, de0, half)
        return {"stars": VSX.offsets(stars, ra0, de0), "source": "ASTAP (offline, VSX do ~18 mag)", "note": note}

    # ------------------------------------------------------------ ranging (12–48 h ephemeris)
    def _fov_deg(job, cands) -> list[float] | None:
        """Field of view (width, height) in degrees: FITS size × pixel scale, else candidate extent."""
        scale = float((job.summary or {}).get("pixel_scale") or 0)
        if not scale:
            return None
        w = h = 0
        try:
            from astropy.io import fits
            f = next((p for p in (job.files or []) if Path(p).exists()), None)
            if f:
                hd = fits.getheader(f)
                w, h = int(hd.get("NAXIS1", 0)), int(hd.get("NAXIS2", 0))
        except Exception:                 # noqa: BLE001 – the plot works without it
            w = h = 0
        if not (w and h) and cands:
            w = int(max(c["x"] for c in cands) + 1)
            h = int(max(c["y"] for c in cands) + 1)
        return [w * scale / 3600, h * scale / 3600] if w and h else None

    @app.get("/api/jobs/{job_id}/candidates/{cid}/ranging")
    def ranging(job_id: str, cid: int, model: str = "all", hours: str = "12,18,24,36,48", rho_min: float = 0.003):
        from .. import ranging as RGN
        job = _job(job_id)
        S = job.summary or {}
        cands = jobs.read_json(job_id, "candidates.json", [])
        c = next((x for x in cands if x["id"] == cid), None)
        if c is None:
            raise HTTPException(404, "Kandidát neexistuje")
        if S.get("mjd_ref") is None or not S.get("site"):
            raise HTTPException(400, "Úloha nemá čas reference nebo polohu observatoře – ranging nelze spočítat.")
        if model not in RGN.MODELS:
            model = "all"
        try:
            hh = [float(x) for x in hours.split(",") if x.strip()][:12]
        except ValueError:
            hh = [12, 18, 24, 36, 48]
        hh = [h for h in hh if 0 < h <= 24 * 30] or [12, 18, 24, 36, 48]
        try:
            out = RGN.run(c, float(S["mjd_ref"]), S["site"], hours=hh, model=model,
                          pixel_scale=S.get("pixel_scale"), seed=cid,
                          rho_min=float(min(max(rho_min, 0.001), 5.0)))
        except ValueError as e:
            raise HTTPException(400, f"Ranging nelze spočítat: {e}")
        out["fov_deg"] = _fov_deg(job, cands)
        out["rho_min"] = float(min(max(rho_min, 0.001), 5.0))
        out["models"] = RGN.MODELS
        return Response(json.dumps(out, default=_np_default, ensure_ascii=False), media_type="application/json")

    # ------------------------------------------------------------ classifier
    def _rescore_all(model) -> int:
        n = 0
        for j in jobs.list():
            if j.status != "done":
                continue
            cands = jobs.read_json(j.id, "candidates.json", None)
            if not cands:
                continue
            m = CL.load(settings.root, j.instrument_id) or model
            CL.score(m, cands, CL.job_context(j.summary or {}))
            jobs.write_json(j.id, "candidates.json", cands)
            n += 1
        return n

    @app.get("/api/classifier")
    def classifier_status():
        m = CL.load(settings.root)
        keys = ("trained", "n_pos", "n_neg", "n_injected", "jobs", "cv_auc", "threshold_95", "hidden_at_95")
        return {"model": None if m is None else {k: m.get(k) for k in keys}}

    @app.post("/api/classifier/train")
    def classifier_train():
        try:
            m = CL.train(settings.root)
        except ValueError as e:
            raise HTTPException(400, str(e))
        n = _rescore_all(m)
        return {"report": m["report"] + f" Skóre přepočteno v {n} úlohách.", "cv_auc": m.get("cv_auc"),
                "threshold_95": m.get("threshold_95")}

    @app.get("/api/jobs/{job_id}/known")
    def known(job_id: str):
        return jobs.read_json(job_id, "known.json", [])

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def log(job_id: str):
        return jobs.read_log(job_id)

    @app.get("/api/jobs/{job_id}/img/{name}")
    def image(job_id: str, name: str):
        p = (jobs.job_dir(job_id) / "img" / name).resolve()
        if jobs.job_dir(job_id).resolve() not in p.parents or not p.exists():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/api/jobs/{job_id}/export/{fmt}", response_class=PlainTextResponse)
    def export(job_id: str, fmt: str):
        job = _job(job_id)
        cands = jobs.read_json(job_id, "candidates.json", [])
        if fmt == "csv":
            return export_csv(cands)
        if fmt == "mpc80":
            fr = jobs.read_json(job_id, "frames.json", {})
            obs_code = job.summary.get("observatory_code") or settings.observatory_code
            return export_mpc80(cands, job.summary.get("mjd_ref", 0), fr.get("mjd", []), obs_code,
                                job.summary.get("filter", "G")[:1], summary=job.summary)
        raise HTTPException(400, "Formát: csv | mpc80")

    # ---------------------------------------------------------------- catalogs
    def _run_catalog(kind: str, url: str | None, path: str | None):
        cat_task.update(running=True, kind=kind, done=0, total=None, error=None, message="Stahuji…")
        try:
            def prog(done, total):
                cat_task.update(done=done, total=total)
            if url:
                mpc.download(kind, url, progress=prog)
            else:
                cat_task["message"] = "Importuji…"
                mpc.import_file(kind, path)
            cat_task["message"] = "Hotovo"
        except Exception as e:  # noqa: BLE001
            cat_task.update(error=str(e), message="Chyba")
        finally:
            cat_task["running"] = False

    @app.post("/api/catalogs/update")
    def update_catalog(data: dict = Body(...)):
        kind = data.get("kind", MPCORB_NAME)
        if not settings.online.mpc_download:
            raise HTTPException(403, "Stahování z MPC je v Nastavení vypnuté (offline režim).")
        if cat_task["running"]:
            raise HTTPException(409, "Aktualizace už běží")
        url = settings.online.mpcorb_url if kind == MPCORB_NAME else settings.online.comets_url
        threading.Thread(target=_run_catalog, args=(kind, url, None), daemon=True).start()
        return {"started": True}

    @app.post("/api/catalogs/import")
    def import_catalog(data: dict = Body(...)):
        kind, path = data.get("kind", MPCORB_NAME), data.get("path", "")
        if not Path(path).is_file():
            raise HTTPException(400, f"Soubor {path} neexistuje")
        if cat_task["running"]:
            raise HTTPException(409, "Aktualizace už běží")
        threading.Thread(target=_run_catalog, args=(kind, None, path), daemon=True).start()
        return {"started": True}

    # ------------------------------------------------------------------- watch
    @app.get("/api/watch")
    def watch_status():
        return watcher.status() if watcher else {}

    @app.post("/api/watch")
    def watch_config(data: dict = Body(...)):
        settings.update({"watch": data})
        settings.save()
        if watcher:
            if settings.watch.enabled:
                watcher.start()
            else:
                watcher.stop()
        return watcher.status() if watcher else {}

    @app.post("/api/watch/scan")
    def watch_scan():
        return {"started": watcher.scan_once() if watcher else []}

    @app.post("/api/watch/forget")
    def watch_forget(data: dict = Body(...)):
        if watcher:
            watcher.forget(data["dir"])
        return {"ok": True}

    # ------------------------------------------------------------------ static
    @app.get("/")
    def index():
        # cache-busting: the browser (or the desktop web view, where Ctrl+F5 may not exist) kept an
        # old style.css / app.js after an update – the files get their modification time as ?v=
        from fastapi.responses import HTMLResponse
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for name in ("style.css", "app.js"):
            try:
                v = int((STATIC / name).stat().st_mtime)
            except OSError:
                continue
            html = html.replace(f'/static/{name}"', f'/static/{name}?v={v}"')
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.settings = settings
    app.state.runner = runner
    app.state.watcher = watcher
    return app
