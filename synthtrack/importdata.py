"""Move a SynthTrack data folder to another computer (e.g. Windows → Linux).

``export_data`` (run on the old machine) collects what a new installation needs:
settings, instrument profiles, catalogues, classifier models and the master
darks/flats of every instrument into one folder (``masters/<instrument id>/``).
Jobs and their results are not included.

``import_data`` (run on the new machine) copies that folder into the data
directory and fixes everything that holds a path of the old machine: each
profile's ``masters_dir`` points to ``<data>/masters/<id>`` when it was exported,
and settings with a path that does not exist here (ASTAP, UCAC4, watch root) are
cleared with a note, so the GUI shows what to fill in.
"""
from __future__ import annotations

import shutil
from pathlib import Path

try:                      # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
import tomli_w

CATALOG_FILES = ("mpcorb.npz", "mpcorb.json", "comets.npz", "comets.json", "CometEls.txt")
FITS_EXT = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz")


def _is_fits(p: Path) -> bool:
    return p.is_file() and p.name.lower().endswith(FITS_EXT)


def export_data(src: Path, dst: Path, masters: bool = True, log=print,
                path_map: dict[str, str] | None = None) -> Path:
    """Copy the parts of data folder ``src`` a new installation needs into ``dst``.

    ``path_map``: prefix replacements for ``masters_dir`` when the export runs where the
    profile paths look different (e.g. a Windows drive mounted elsewhere)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "settings.toml").exists():
        shutil.copy2(src / "settings.toml", dst / "settings.toml")
    (dst / "instruments").mkdir(exist_ok=True)
    for p in sorted((src / "instruments").glob("*.toml")):
        shutil.copy2(p, dst / "instruments" / p.name)
        if not masters:
            continue
        d = tomllib.loads(p.read_text(encoding="utf-8"))
        raw = str(d.get("masters_dir", "") or "")
        for a, b in (path_map or {}).items():
            if raw.lower().startswith(a.lower()):
                raw = b + raw[len(a):].replace("\\", "/")
                break
        md = Path(raw)
        if not raw or not md.is_dir():
            log(f"{p.stem}: adresář masterů {md} nenalezen – nepřibalen")
            continue
        out = dst / "masters" / p.stem
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(md.rglob("*")):
            if _is_fits(f):
                rel = f.relative_to(md)
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out / rel)
                n += 1
        log(f"{p.stem}: {n} masterů z {md}")
    (dst / "catalogs").mkdir(exist_ok=True)
    for name in CATALOG_FILES:
        if (src / "catalogs" / name).exists():
            shutil.copy2(src / "catalogs" / name, dst / "catalogs" / name)
    if (src / "models").is_dir():
        shutil.copytree(src / "models", dst / "models", dirs_exist_ok=True)
    return dst


def import_data(src: Path, dst: Path, log=print) -> list[str]:
    """Install an exported folder ``src`` into data folder ``dst``; returns notes for the user."""
    src, dst = Path(src), Path(dst)
    if not (src / "instruments").is_dir():
        raise FileNotFoundError(f"{src} neobsahuje exportovaná data (chybí instruments/)")
    dst.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    # settings: keep an existing file as backup
    if (src / "settings.toml").exists():
        if (dst / "settings.toml").exists():
            shutil.copy2(dst / "settings.toml", dst / "settings.pred-importem.toml")
        s = tomllib.loads((src / "settings.toml").read_text(encoding="utf-8"))
        for sect, key, what in (("astap", "executable", "ASTAP"), ("astap", "database_dir", "databáze ASTAP"),
                                (None, "ucac4_dir", "UCAC4"), ("watch", "root", "kořen automatického režimu")):
            tbl = s.get(sect, {}) if sect else s
            val = tbl.get(key, "")
            if val and not Path(val).exists():
                tbl[key] = ""
                notes.append(f"{what}: cesta {val} tu neexistuje – vymazána, doplňte v Nastavení")
        if s.get("watch", {}).get("enabled") and not s.get("watch", {}).get("root"):
            s["watch"]["enabled"] = False
        (dst / "settings.toml").write_bytes(tomli_w.dumps(s).encode())
    (dst / "instruments").mkdir(exist_ok=True)
    for p in sorted((src / "instruments").glob("*.toml")):
        d = tomllib.loads(p.read_text(encoding="utf-8"))
        ms = src / "masters" / p.stem
        if ms.is_dir():
            target = dst / "masters" / p.stem
            shutil.copytree(ms, target, dirs_exist_ok=True)
            d["masters_dir"] = str(target)
        elif d.get("masters_dir") and not Path(d["masters_dir"]).exists():
            notes.append(f"{p.stem}: mastery nebyly přibaleny ({d['masters_dir']}) – nastavte adresář v Přístroje")
        (dst / "instruments" / p.name).write_bytes(tomli_w.dumps(d).encode())
        (dst / "instruments" / f"{p.stem}.masters.json").unlink(missing_ok=True)   # cache with old paths
        log(f"přístroj {p.stem}: mastery {d.get('masters_dir', '')}")
    if (src / "catalogs").is_dir():
        (dst / "catalogs").mkdir(exist_ok=True)
        for f in (src / "catalogs").iterdir():
            if f.is_file():
                shutil.copy2(f, dst / "catalogs" / f.name)
    if (src / "models").is_dir():
        shutil.copytree(src / "models", dst / "models", dirs_exist_ok=True)
    for n in notes:
        log("POZOR: " + n)
    return notes
