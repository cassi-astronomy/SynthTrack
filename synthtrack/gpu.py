"""Compute backend selection (vendor neutral).

The tracking code is written against the NumPy API (``xp``). The same code runs
on any GPU that CuPy supports:

* NVIDIA – CuPy built for CUDA (Windows, Linux): ``pip install -e ".[cuda12]"``
* AMD    – CuPy built for ROCm/HIP (Linux, WSL2):  ``pip install -e ".[rocm]"``

and falls back to NumPy on the CPU otherwise. Which platform the installed CuPy
targets is detected at run time (``cupy.cuda.runtime.is_hip``); the rest of the
pipeline only sees ``Backend.xp`` and never needs to know the vendor.

Device status for the header bar comes from ``nvidia-smi`` (NVIDIA) or
``amd-smi`` / ``rocm-smi`` (AMD) when available – all best effort.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

PLATFORM_LABEL = {"cuda": "CUDA", "rocm": "ROCm", "cpu": "CPU"}
# accepted values of settings.gpu.backend
PREFERENCES = ("auto", "cupy", "cuda", "rocm", "numpy")


@dataclass
class DeviceInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    util_pct: int | None = None
    arch: str = ""                  # e.g. "sm_86" (NVIDIA), "gfx1151" (AMD)


@dataclass
class Backend:
    name: str                       # "cupy" | "numpy"
    xp: object                      # module with numpy API
    device: int = 0
    devices: list[DeviceInfo] = field(default_factory=list)
    note: str = ""
    platform: str = "cpu"           # "cuda" | "rocm" | "cpu"

    @property
    def is_gpu(self) -> bool:
        return self.name != "numpy"

    @property
    def label(self) -> str:
        return PLATFORM_LABEL.get(self.platform, self.platform)

    def asnumpy(self, a):
        if self.name == "cupy":
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def asarray(self, a):
        return self.xp.asarray(a)

    def info(self) -> dict:
        return {"backend": self.name, "platform": self.platform, "label": self.label,
                "device": self.device, "note": self.note,
                "devices": [asdict(d) for d in self.devices]}


_cached: Backend | None = None
_cached_config: tuple[str, int, float] | None = None


def get_backend(preference: str = "auto", device: int = 0, memory_fraction: float = 0.6,
                refresh: bool = False) -> Backend:
    """``preference``: auto | cupy (any CuPy GPU) | cuda | rocm (require that platform) | numpy."""
    global _cached, _cached_config
    preference = (preference or "auto").lower()
    if preference not in PREFERENCES:
        preference = "auto"
    config = (preference, int(device), float(memory_fraction))
    if _cached is not None and not refresh and _cached_config == config:
        return _cached
    be = None
    if preference != "numpy":
        be = _try_cupy(device, memory_fraction, preference)
    if be is None:
        if preference == "numpy":
            note = "Vynucen CPU backend."
        elif _cupy_error:
            note = f"GPU nepoužitelné ({_cupy_error}) – běží CPU (NumPy)."
        else:
            note = "CuPy není nainstalované – běží CPU (NumPy)." + _install_hint()
        be = Backend("numpy", np, note=note, devices=smi_devices())
    _cached, _cached_config = be, config
    return be


_cupy_error: str = ""


def _decode(v) -> str:
    return v.decode(errors="replace").strip("\x00 ") if isinstance(v, bytes) else str(v)


def _try_cupy(device: int, memory_fraction: float, preference: str = "auto") -> Backend | None:
    global _cupy_error
    _cupy_error = ""
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    try:
        is_hip = bool(getattr(cp.cuda.runtime, "is_hip", False))
    except Exception:
        is_hip = False
    platform = "rocm" if is_hip else "cuda"
    if preference in ("cuda", "rocm") and preference != platform:
        _cupy_error = (f"vyžádán {PLATFORM_LABEL[preference]}, ale nainstalované CuPy je pro "
                       f"{PLATFORM_LABEL[platform]}")
        return None
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            _cupy_error = f"{PLATFORM_LABEL[platform]}: žádné GPU nenalezeno" + (_rocm_hint() if is_hip else "")
            return None
        if not 0 <= device < n:
            device = 0
        cp.cuda.Device(device).use()
        cp.get_default_memory_pool().set_limit(fraction=float(memory_fraction))
        devs = _cupy_devices(cp, n, is_hip)
        _merge_util(devs)
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        _cupy_error = f"{PLATFORM_LABEL[platform]}: {msg}" + (_rocm_hint() if is_hip else "")
        return None
    # Smoke test: elementwise ops and reductions are JIT-compiled at run time
    # (NVRTC on CUDA, hipRTC on ROCm), sorting uses Thrust/rocThrust. If any of
    # that is missing every GPU call would fail mid-job -> fall back to CPU now.
    try:
        _smoke_test(cp)
    except Exception as e:
        _cupy_error = _explain_smoke_error(e, is_hip, devs[device].arch if devs else "")
        return None
    arch = f" ({devs[device].arch})" if devs and devs[device].arch else ""
    return Backend("cupy", cp, device=device, devices=devs, platform=platform,
                   note=f"{PLATFORM_LABEL[platform]}{arch}, limit paměti {memory_fraction:.0%}")


def _cupy_devices(cp, n: int, is_hip: bool) -> list[DeviceInfo]:
    devs = []
    for i in range(n):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = _decode(props.get("name", f"GPU {i}"))
            if is_hip:
                arch = _decode(props.get("gcnArchName", "")).split(":")[0]
            else:
                arch = f"sm_{props.get('major', '')}{props.get('minor', '')}" if "major" in props else ""
            devs.append(DeviceInfo(i, name, int(total) // 2**20, int(free) // 2**20, arch=arch))
    return devs


def _smoke_test(cp) -> None:
    a = cp.arange(1024, dtype=cp.float32)
    if abs(float((a * 2.0 + 1.0).sum()) - 1048576.0) > 1.0:
        raise RuntimeError("chybný výsledek testovacího výpočtu")
    s = cp.sort(a[::-1].copy())
    if float(s[0]) != 0.0 or float(s[-1]) != 1023.0:
        raise RuntimeError("chybný výsledek testovacího řazení")


def _explain_smoke_error(e: Exception, is_hip: bool, arch: str) -> str:
    msg = str(e).splitlines()[0] if str(e) else type(e).__name__
    low = msg.lower()
    if not is_hip and ("header" in low or "nvrtc" in low):
        return ("chybí hlavičky CUDA/NVRTC, nainstalujte: "
                "pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12")
    if is_hip:
        if "invalid device function" in low or "nobinaryforgpu" in low or "no kernel image" in low:
            return (f"ROCm: CuPy/knihovny nejsou sestavené pro tuto GPU ({arch or '?'}) – "
                    f"použijte novější ROCm nebo sestavte CuPy pro {arch or 'svou architekturu'}")
        if "hiprtc" in low or "header" in low or "comgr" in low:
            return "ROCm: chybí hipRTC/hlavičky – nainstalujte: sudo apt install rocm (nebo rocm-dev)"
        return f"ROCm: {msg}" + _rocm_hint()
    return msg


def _rocm_hint() -> str:
    if sys.platform.startswith("linux"):
        return (" – zkontrolujte rocminfo, členství uživatele ve skupinách render,video "
                "a verzi jádra")
    return ""


# --------------------------------------------------------------------- vendor detection
_VENDOR_IDS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
_vendors_cache: list[tuple[str, str]] | None = None


def detect_gpus() -> list[tuple[str, str]]:
    """Graphics adapters present in the machine as (vendor, name); no GPU library needed."""
    global _vendors_cache
    if _vendors_cache is not None:
        return list(_vendors_cache)
    found: list[tuple[str, str]] = []
    try:
        if sys.platform.startswith("linux"):
            for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
                if "-" in Path(dev).parent.name:          # card0-HDMI-A-1 etc.
                    continue
                try:
                    vid = Path(dev, "vendor").read_text().strip().lower()
                    cls = Path(dev, "class").read_text().strip().lower()
                except OSError:
                    continue
                if not cls.startswith("0x03"):           # display controller
                    continue
                vendor = _VENDOR_IDS.get(vid, vid)
                name = ""
                try:
                    name = Path(dev, "product_name").read_text().strip()
                except OSError:
                    pass
                found.append((vendor, name or vendor))
        elif sys.platform == "win32":
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "(Get-CimInstance Win32_VideoController).Name"],
                               capture_output=True, text=True, timeout=10,
                               stdin=subprocess.DEVNULL, **NO_WINDOW)
            for line in r.stdout.splitlines():
                nm = line.strip()
                if not nm:
                    continue
                up = nm.upper()
                vendor = ("NVIDIA" if "NVIDIA" in up else
                          "AMD" if ("AMD" in up or "RADEON" in up) else
                          "Intel" if "INTEL" in up else "")
                found.append((vendor, nm))
    except Exception:
        found = []
    _vendors_cache = found
    return list(found)


def _install_hint() -> str:
    vendors = {v for v, _ in detect_gpus()}
    if "NVIDIA" in vendors:
        return ' Nalezena NVIDIA GPU – pro zrychlení: pip install -e ".[cuda12]".'
    if "AMD" in vendors:
        if sys.platform == "win32":
            return (" Nalezena AMD GPU – CuPy pro AMD na Windows neexistuje; GPU lze použít "
                    "v Linuxu nebo WSL2 s ROCm.")
        if sys.platform.startswith("linux"):
            return (' Nalezena AMD GPU – nainstalujte ROCm (Ubuntu: sudo apt install rocm) '
                    'a pip install -e ".[rocm]".')
    return ""


# ----------------------------------------------------------------------- device status
# Windows: never flash a console window when the app runs without one (pythonw).
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

_SMI_CACHE: tuple[float, list] = (0.0, [])
_SMI_EXE: dict[str, str | None] = {}
_SMI_TTL_S = 4.0


def _exe(name: str) -> str | None:
    if name not in _SMI_EXE:
        p = shutil.which(name)
        if p is None and sys.platform.startswith("linux") and os.path.exists(f"/opt/rocm/bin/{name}"):
            p = f"/opt/rocm/bin/{name}"
        _SMI_EXE[name] = p
    return _SMI_EXE[name]


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=5,
                          stdin=subprocess.DEVNULL, **NO_WINDOW).stdout


def smi_devices() -> list[DeviceInfo]:
    """Best effort status via nvidia-smi / amd-smi / rocm-smi (works without CuPy); cached."""
    global _SMI_CACHE
    now = time.monotonic()
    if now - _SMI_CACHE[0] < _SMI_TTL_S:
        return [replace(d) for d in _SMI_CACHE[1]]
    devs: list[DeviceInfo] = []
    for fn in (_query_nvidia_smi, _query_amd_smi, _query_rocm_smi):
        try:
            devs = fn()
        except Exception:
            devs = []
        if devs:
            break
    _SMI_CACHE = (now, devs)
    return [replace(d) for d in devs]


nvidia_smi_devices = smi_devices          # backwards compatible name


def _query_nvidia_smi() -> list[DeviceInfo]:
    exe = _exe("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"])
    return parse_nvidia_smi(out)


def parse_nvidia_smi(out: str) -> list[DeviceInfo]:
    devs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                util = int(parts[4]) if parts[4].isdigit() else None
                devs.append(DeviceInfo(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), util))
            except ValueError:
                continue
    return devs


def _query_amd_smi() -> list[DeviceInfo]:
    exe = _exe("amd-smi")
    if not exe:
        return []
    static = _run([exe, "static", "--asic", "--json"])
    metric = _run([exe, "metric", "--usage", "--mem-usage", "--json"])
    return parse_amd_smi(static, metric)


def _num(v) -> float | None:
    """amd-smi values are numbers, strings, or {"value": x, "unit": u}."""
    unit = ""
    if isinstance(v, dict):
        unit = str(v.get("unit", "")).lower()
        v = v.get("value")
    try:
        x = float(str(v).split()[0])
    except (TypeError, ValueError, IndexError):
        return None
    if unit in ("b", "bytes"):
        x /= 2**20
    elif unit in ("kb", "kib"):
        x /= 2**10
    elif unit in ("gb", "gib"):
        x *= 2**10
    return x


def _gpu_list(js) -> list[dict]:
    if isinstance(js, dict):
        for k in ("gpu_data", "gpus", "data"):
            if isinstance(js.get(k), list):
                return js[k]
        return [js]
    return js if isinstance(js, list) else []


def parse_amd_smi(static_json: str, metric_json: str) -> list[DeviceInfo]:
    try:
        st = _gpu_list(json.loads(static_json)) if static_json.strip() else []
        me = _gpu_list(json.loads(metric_json)) if metric_json.strip() else []
    except ValueError:
        return []
    names = {}
    for i, g in enumerate(st):
        idx = int(g.get("gpu", i))
        asic = g.get("asic", {}) if isinstance(g.get("asic"), dict) else {}
        names[idx] = (str(asic.get("market_name") or asic.get("marketing_name") or f"AMD GPU {idx}"),
                      str(asic.get("target_graphics_version") or ""))
    devs = []
    for i, g in enumerate(me):
        idx = int(g.get("gpu", i))
        usage = g.get("usage", {}) if isinstance(g.get("usage"), dict) else {}
        mem = g.get("mem_usage", {}) if isinstance(g.get("mem_usage"), dict) else {}
        util = _num(usage.get("gfx_activity", usage.get("gfx_usage")))
        total = _num(mem.get("total_vram"))
        used = _num(mem.get("used_vram"))
        free = _num(mem.get("free_vram"))
        # APU (shared memory): the BIOS carve-out is tiny, the GPU really uses GTT
        gtt_t, gtt_u = _num(mem.get("total_gtt")), _num(mem.get("used_gtt"))
        if gtt_t and total is not None and total * 8 < gtt_t:
            total, used, free = (total + gtt_t, (used or 0) + (gtt_u or 0), None)
        if total is None:
            continue
        if free is None:
            free = total - (used or 0)
        name, arch = names.get(idx, (f"AMD GPU {idx}", ""))
        devs.append(DeviceInfo(idx, name, int(total), int(max(free, 0)),
                               None if util is None else int(util), arch=arch))
    return devs


def _query_rocm_smi() -> list[DeviceInfo]:
    exe = _exe("rocm-smi")
    if not exe:
        return []
    out = _run([exe, "--showproductname", "--showuse", "--showmeminfo", "vram", "gtt", "--json"])
    return parse_rocm_smi(out)


def parse_rocm_smi(out: str) -> list[DeviceInfo]:
    try:
        js = json.loads(out[out.find("{"):]) if "{" in out else {}
    except ValueError:
        return []
    devs = []
    for key, d in js.items():
        if not (isinstance(d, dict) and key.lower().startswith("card")):
            continue
        try:
            idx = int(key[4:])
        except ValueError:
            continue

        def get(*names):
            for n in names:
                for k, v in d.items():
                    if k.lower() == n.lower():
                        return v
            return None
        total = _num(get("VRAM Total Memory (B)"))
        used = _num(get("VRAM Total Used Memory (B)"))
        gtt_t = _num(get("GTT Total Memory (B)"))
        gtt_u = _num(get("GTT Total Used Memory (B)"))
        if total is not None:
            total /= 2**20
            used = (used or 0) / 2**20
            if gtt_t and total * 8 < gtt_t / 2**20:      # APU
                total += gtt_t / 2**20
                used += (gtt_u or 0) / 2**20
        else:
            continue
        util = _num(get("GPU use (%)"))
        name = get("Card Series", "Card series", "Card Model", "Device Name") or f"AMD GPU {idx}"
        arch = get("GFX Version") or ""
        devs.append(DeviceInfo(idx, str(name), int(total), int(max(total - used, 0)),
                               None if util is None else int(util), arch=str(arch)))
    return devs


def _merge_util(devs: list[DeviceInfo]) -> None:
    smi = {d.index: d for d in smi_devices()}
    for d in devs:
        if d.index in smi:
            d.util_pct = smi[d.index].util_pct


def live_status() -> list[dict]:
    """Current per-GPU status for the header bar."""
    be = _cached or get_backend()
    if be.platform == "rocm":
        # memory from the HIP runtime (correct also for APUs with shared memory),
        # utilisation from amd-smi/rocm-smi
        try:
            devs = _cupy_devices(be.xp, len(be.devices), True)
            _merge_util(devs)
            return [asdict(d) for d in devs]
        except Exception:
            pass
    devs = smi_devices() or be.devices
    return [asdict(d) for d in devs]
