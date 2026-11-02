"""Command line interface.

  synthtrack app                                           desktop window (no browser)
  synthtrack shortcut                                      desktop / Start-menu shortcut
  synthtrack serve   [--host 127.0.0.1] [--port 8765]     web GUI in the browser (also remote)
  synthtrack run     --instrument ID --lights DIR           no GUI; DIR or its sub-folders, one job per field
  synthtrack masters --instrument ID                        list master library
  synthtrack catalogs update|import [--kind mpcorb|comets] [--file PATH]
  synthtrack gpu                                            backend / device info
  synthtrack demo    [--out DIR]                            synthetic test sequence + profile

Global: --data DIR (default ~/SynthTrackData or $SYNTHTRACK_DATA)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .settings import load_settings


def main(argv=None):
    # A Windows service/shortcut may inherit a legacy code page that cannot
    # encode Czech text or the arcsecond symbol and used to abort a CLI job
    # during its first progress message.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass
    ap = argparse.ArgumentParser(prog="synthtrack", description="GPU synthetic tracking pipeline")
    ap.add_argument("--data", help="datový adresář (nastavení, přístroje, úlohy, katalogy)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="spustit webové GUI")
    s.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = přístup z jiných počítačů v síti")
    s.add_argument("--port", type=int, default=8765)

    r = sub.add_parser("run", help="zpracovat jeden adresář bez GUI")
    r.add_argument("--instrument", required=True)
    r.add_argument("--lights", required=True)
    r.add_argument("--param", action="append", default=[], help="přepsání parametru, např. snr_threshold=6")
    r.add_argument("--no-split", action="store_true", help="nerozdělovat adresář podle zorných polí")
    r.add_argument("--field", type=int, action="append",
                   help="zpracovat jen toto pole (1, 2, …; lze opakovat; jen pro jeden adresář)")

    m = sub.add_parser("masters", help="vypsat knihovnu master snímků")
    m.add_argument("--instrument", required=True)

    c = sub.add_parser("catalogs", help="MPCORB / CometEls")
    c.add_argument("action", choices=["update", "import", "status"])
    c.add_argument("--kind", choices=["mpcorb", "comets"], default="mpcorb")
    c.add_argument("--file")

    sub.add_parser("gpu", help="informace o GPU backendu")
    sub.add_parser("app", help="spustit jako samostatnou aplikaci v okně (bez prohlížeče)")
    sub.add_parser("shortcut", help="vytvořit zástupce SynthTrack na ploše a v nabídce Start")

    d = sub.add_parser("demo", help="vytvořit syntetickou testovací sekvenci a profil přístroje")
    d.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    settings = load_settings(a.data)

    if a.cmd == "serve":
        import uvicorn

        from .web.app import create_app
        print(f"SynthTrack GUI: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}")
        print(f"Datový adresář: {settings.root}")
        uvicorn.run(create_app(settings), host=a.host, port=a.port, log_level="warning")

    elif a.cmd == "run":
        from .fields import create_tree_jobs
        from .instruments import InstrumentStore
        from .jobs import JobStore, RunContext
        from .pipeline import run_job

        params = dict(p.split("=", 1) for p in a.param)
        store = JobStore(settings.jobs_dir)
        istore = InstrumentStore(settings.instruments_dir)
        inst = istore.get(a.instrument)
        jobs, splits, skipped = create_tree_jobs(
            store, inst, a.lights, Path(a.lights).name, "manual", params,
            split=settings.field_split and not a.no_split, tolerance_fov=settings.field_tolerance_fov,
            select=a.field, exclude=[i.masters_dir for i in istore.list() if i.masters_dir])
        if len(splits) > 1:
            print(f"{len(splits)} adresářů se snímky (předem roztříděná pole):")
        for d, fs in splits:
            if len(splits) > 1:
                print(f"  {d}: {fs.n_frames} snímků")
            for note in fs.notes:
                print("  " + note)
            if fs.multiple:
                for f in fs.fields:
                    print("    " + f.describe())
        for note in skipped:
            print(note)
        if not jobs:
            print("CHYBA: žádné pole ke zpracování", file=sys.stderr)
            sys.exit(1)

        class Ctx(RunContext):
            def log(self, msg):
                super().log(msg)
                print(msg, flush=True)

        failed = 0
        for k, job in enumerate(jobs, 1):     # fields one after another
            if len(jobs) > 1:
                print(f"\n===== {job.name} ({k}/{len(jobs)}) =====", flush=True)
            ctx = Ctx(None, job, store)
            job.status = "running"
            try:
                run_job(job, ctx, settings)
                job.status = "done"
            except Exception as e:
                job.status, job.error = "failed", str(e)
                failed += 1
                print("CHYBA:", e, file=sys.stderr)
            finally:
                ctx.release_memmaps()
                store.cleanup_work(job.id)
            store.save(job)
            if job.status == "done":
                print(json.dumps({k2: v for k2, v in job.summary.items() if k2 not in ("masters", "field")}, indent=1))
            print(f"Výsledky: {store.job_dir(job.id)}")
        if failed:
            sys.exit(1)

    elif a.cmd == "masters":
        from .instruments import InstrumentStore
        from .masters import MasterLibrary

        inst = InstrumentStore(settings.instruments_dir).get(a.instrument)
        for e in MasterLibrary(inst, settings.instruments_dir / f"{inst.id}.masters.json").scan():
            mm = e.meta
            print(f"{e.type:6s} exp={mm.exptime!s:>6} filtr={mm.filter:5s} T={mm.temperature!s:>6} "
                  f"bin={mm.xbin}x{mm.ybin} gain={mm.gain} {mm.path}")

    elif a.cmd == "catalogs":
        from .mpcdata import MpcStore

        st = MpcStore(settings.catalogs_dir)
        if a.action == "status":
            for k in ("mpcorb", "comets"):
                print(st.status(k).to_dict())
        elif a.action == "import":
            if not a.file:
                ap.error("--file je povinný pro import")
            print(st.import_file(a.kind, a.file).to_dict())
        else:
            if not settings.online.mpc_download:
                print("Stahování z MPC je vypnuté. Povolte online.mpc_download v settings.toml nebo v GUI.")
                sys.exit(2)
            url = settings.online.mpcorb_url if a.kind == "mpcorb" else settings.online.comets_url
            print(st.download(a.kind, url, progress=lambda d, t: print(f"\r{d / 2**20:.1f} MB", end="")).to_dict())

    elif a.cmd == "app":
        from .desktop import run_app
        run_app(settings)

    elif a.cmd == "shortcut":
        from .desktop import create_shortcuts
        made = create_shortcuts(a.data)
        for m in made:
            print("Vytvořen zástupce:", m)
        if not made:
            print("Zástupce se nepodařilo vytvořit.")
            sys.exit(1)

    elif a.cmd == "gpu":
        from .gpu import get_backend
        print(json.dumps(get_backend(settings.gpu.backend, settings.gpu.device,
                                     settings.gpu.memory_fraction).info(), indent=1))

    elif a.cmd == "demo":
        from .instruments import Instrument, InstrumentStore
        from .simulate import Mover, make_sequence

        out = Path(a.out) if a.out else settings.root / "demo"
        make_sequence(out, n=20, shape=(512, 512),
                      movers=[Mover(100, 120, 1.2, 0.6, 600), Mover(380, 300, -0.5, -0.9, 400),
                              Mover(250, 420, 0.3, -0.2, 900)],
                      drift_px_per_frame=(0.25, -0.15), gradient=0.25, vignetting=0.3, hot_pixels=30)
        inst = Instrument(id="demo", name="Demo (syntetická data)", focal_length_mm=1000, pixel_size_um=4.85,
                          masters_dir=str(out / "masters"), saturation_adu=65000)
        InstrumentStore(settings.instruments_dir).save(inst)
        print(f"Demo data: {out / 'lights'}\nProfil přístroje: demo\n"
              f"Spusťte: synthtrack run --instrument demo --lights \"{out / 'lights'}\"")


if __name__ == "__main__":
    main()
