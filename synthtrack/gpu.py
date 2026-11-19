"""Compute backend selection.

The tracking code is written against the NumPy API (``xp``). With CuPy
installed (NVIDIA, Windows or Linux) the very same code runs on the GPU;
otherwise it falls back to NumPy on the CPU. Other backends (e.g. AMD via
PyTorch/ROCm or OpenCL) can be added behind the same interface later.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np


@dataclass
class DeviceInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    util_pct: int | None = None


@dataclass
class Backend:
    name: str                       # "cupy" | "numpy"
    xp: object                      # module with numpy API
    device: int = 0
    devices: list[DeviceInfo] = field(default_factory=list)
    note: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.name != "numpy"

    def asnumpy(self, a):
        if self.name == "cupy":
            return self.xp.asnumpy(a)
        return np.asarray(a)

    def asarray(self, a):
        return self.xp.asarray(a)

    def info(self) -> dict:
        return {"backend": self.name, "device": self.device, "note": self.note,
                "devices": [asdict(d) for d in self.devices]}


_cached: Backend | None = None
_cached_config: tuple[str, int, float] | None = None


def get_backend(preference: str = "auto", device: int = 0, memory_fraction: float = 0.6,
                refresh: bool = False) -> Backend:
    global _cached, _cached_config
    config = (preference, int(device), float(memory_fraction))
    if _cached is not None and not refresh and _cached_config == config:
        return _cached
    be = None
    if preference in ("auto", "cupy"):
        be = _try_cupy(device, memory_fraction)
    if be is None:
        if preference == "numpy":
            note = "Vynucen CPU backend."
        elif _cupy_error:
            note = f"GPU nepoužitelné ({_cupy_error}) – běží CPU (NumPy)."
        else:
            note = "CuPy/CUDA nedostupné – běží CPU (NumPy)."
        be = Backend("numpy", np, note=note, devices=nvidia_smi_devices())
    _cached, _cached_config = be, config
    return be


_cupy_error: str = ""


def _try_cupy(device: int, memory_fraction: float) -> Backend | None:
    global _cupy_error
    _cupy_error = ""
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    try:
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            return None
        cp.cuda.Device(device).use()
        cp.get_default_memory_pool().set_limit(fraction=float(memory_fraction))
        devs = []
        for i in range(n):
            with cp.cuda.Device(i):
                free, total = cp.cuda.runtime.memGetInfo()
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
                devs.append(DeviceInfo(i, name, total // 2**20, free // 2**20))
        _merge_util(devs)
    except Exception as e:
        _cupy_error = f"CUDA: {e}"
        return None
    # Smoke test: elementwise ops and reductions are JIT-compiled by NVRTC and need
    # the CUDA headers. Without them every GPU call fails mid-job -> fall back to CPU now.
    try:
        a = cp.arange(1024, dtype=cp.float32)
        if abs(float((a * 2.0 + 1.0).sum()) - 1048576.0) > 1.0:
            raise RuntimeError("chybný výsledek testovacího výpočtu")
    except Exception as e:
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        if "header" in msg.lower() or "nvrtc" in msg.lower():
            msg = 'chybí hlavičky CUDA/NVRTC, nainstalujte: pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12'
        _cupy_error = msg
        return None
    return Backend("cupy", cp, device=device, devices=devs,
                   note=f"CUDA, limit paměti {memory_fraction:.0%}")


# Windows: never flash a console window when the app runs without one (pythonw).
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}

_SMI_CACHE: tuple[float, list] = (0.0, [])
_SMI_EXE: list = []          # [path or None] once looked up
_SMI_TTL_S = 4.0


def nvidia_smi_devices() -> list[DeviceInfo]:
    """Best effort status via nvidia-smi (works even without CuPy); cached for a few seconds."""
    global _SMI_CACHE
    now = time.monotonic()
    if now - _SMI_CACHE[0] < _SMI_TTL_S:
        return [replace(d) for d in _SMI_CACHE[1]]
    if not _SMI_EXE:
        _SMI_EXE.append(shutil.which("nvidia-smi"))
    exe = _SMI_EXE[0]
    if not exe:
        _SMI_CACHE = (now, [])
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
            stdin=subprocess.DEVNULL, **NO_WINDOW).stdout
    except Exception:
        _SMI_CACHE = (now, [])
        return []
    devs = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                devs.append(DeviceInfo(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), int(parts[4])))
            except ValueError:
                continue
    _SMI_CACHE = (now, devs)
    return [replace(d) for d in devs]


def _merge_util(devs: list[DeviceInfo]) -> None:
    smi = {d.index: d for d in nvidia_smi_devices()}
    for d in devs:
        if d.index in smi:
            d.util_pct = smi[d.index].util_pct


def live_status() -> list[dict]:
    """Current per-GPU status for the header bar."""
    be = _cached or get_backend()
    devs = nvidia_smi_devices() or be.devices
    return [asdict(d) for d in devs]
