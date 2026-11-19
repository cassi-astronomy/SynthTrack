"""GPU backend selection is vendor neutral (CUDA / ROCm via CuPy, CPU fallback).

No GPU is needed: a fake ``cupy`` module backed by NumPy stands in for both platforms.
"""
import json
import sys
import types

import numpy as np
import pytest

from synthtrack import gpu


def _fake_cupy(is_hip: bool, arch: str = "gfx1151", fail_sort: bool = False, n: int = 1):
    cp = types.ModuleType("cupy")
    for k in ("arange", "float32", "asarray"):
        setattr(cp, k, getattr(np, k))
    cp.asnumpy = np.asarray

    def sort(a):
        if fail_sort:
            raise RuntimeError("hipErrorNoBinaryForGpu: Unable to find code object for all current devices!")
        return np.sort(a)
    cp.sort = sort

    class Device:
        def __init__(self, i=0): self.i = i
        def use(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def props(i):
        p = {"name": b"AMD Radeon 8060S Graphics" if is_hip else b"NVIDIA GeForce RTX 3070 Laptop GPU"}
        if is_hip:
            p["gcnArchName"] = f"{arch}:sramecc-:xnack-".encode()
        else:
            p.update(major=8, minor=6)
        return p
    runtime = types.SimpleNamespace(is_hip=is_hip, getDeviceCount=lambda: n,
                                    memGetInfo=lambda: (30 * 2**30, 32 * 2**30),
                                    getDeviceProperties=props)
    cp.cuda = types.SimpleNamespace(runtime=runtime, Device=Device)
    pool = types.SimpleNamespace(set_limit=lambda fraction: None)
    cp.get_default_memory_pool = lambda: pool
    return cp


@pytest.fixture
def no_smi(monkeypatch):
    monkeypatch.setattr(gpu, "smi_devices", lambda: [])
    monkeypatch.setattr(gpu, "detect_gpus", lambda: [])


@pytest.mark.parametrize("is_hip,platform,arch", [(False, "cuda", "sm_86"), (True, "rocm", "gfx1151")])
def test_cupy_platform_detected(monkeypatch, no_smi, is_hip, platform, arch):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(is_hip))
    be = gpu.get_backend("auto", refresh=True)
    assert be.name == "cupy" and be.is_gpu
    assert be.platform == platform
    assert be.devices[0].arch == arch
    assert be.devices[0].total_mb == 32 * 1024
    info = be.info()
    assert info["label"] == gpu.PLATFORM_LABEL[platform]
    assert arch in info["note"]


def test_platform_mismatch_falls_back_to_cpu(monkeypatch, no_smi):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(is_hip=True))
    be = gpu.get_backend("cuda", refresh=True)
    assert be.name == "numpy" and be.platform == "cpu"
    assert "ROCm" in be.note
    assert gpu.get_backend("rocm", refresh=True).platform == "rocm"


def test_rocm_unsupported_arch_explained(monkeypatch, no_smi):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(is_hip=True, arch="gfx1036", fail_sort=True))
    be = gpu.get_backend("auto", refresh=True)
    assert be.name == "numpy"
    assert "gfx1036" in be.note


def test_forced_numpy_and_unknown_preference(monkeypatch, no_smi):
    monkeypatch.setitem(sys.modules, "cupy", _fake_cupy(is_hip=False))
    assert gpu.get_backend("numpy", refresh=True).name == "numpy"
    assert gpu.get_backend("opencl", refresh=True).name == "cupy"     # unknown -> auto


def test_install_hint_per_vendor(monkeypatch):
    monkeypatch.setattr(gpu, "detect_gpus", lambda: [("AMD", "Radeon 8060S Graphics")])
    monkeypatch.setattr(gpu.sys, "platform", "linux")
    assert "rocm" in gpu._install_hint().lower()
    monkeypatch.setattr(gpu.sys, "platform", "win32")
    assert "WSL2" in gpu._install_hint()
    monkeypatch.setattr(gpu, "detect_gpus", lambda: [("NVIDIA", "RTX 3070")])
    assert "cuda12" in gpu._install_hint()


def test_parse_nvidia_smi():
    d = gpu.parse_nvidia_smi("0, NVIDIA GeForce RTX 3070 Laptop GPU, 8192, 7000, 13\n1, X, 100, 50, [N/A]\n")
    assert [x.index for x in d] == [0, 1]
    assert d[0].util_pct == 13 and d[1].util_pct is None


def test_parse_amd_smi_apu_uses_gtt():
    static = json.dumps([{"gpu": 0, "asic": {"market_name": "AMD Radeon 8060S Graphics",
                                             "target_graphics_version": "gfx1151"}}])
    metric = json.dumps({"gpu_data": [{"gpu": 0,
                                       "usage": {"gfx_activity": {"value": 42, "unit": "%"}},
                                       "mem_usage": {"total_vram": {"value": 512, "unit": "MB"},
                                                     "used_vram": {"value": 100, "unit": "MB"},
                                                     "total_gtt": {"value": 30000, "unit": "MB"},
                                                     "used_gtt": {"value": 2000, "unit": "MB"}}}]})
    d = gpu.parse_amd_smi(static, metric)
    assert len(d) == 1
    assert d[0].name == "AMD Radeon 8060S Graphics" and d[0].arch == "gfx1151"
    assert d[0].util_pct == 42
    assert d[0].total_mb == 30512 and d[0].free_mb == 30512 - 2100


def test_parse_rocm_smi_discrete():
    out = "WARNING: something\n" + json.dumps({"card0": {
        "Card Series": "Radeon RX 7900 XTX", "GPU use (%)": "7",
        "VRAM Total Memory (B)": str(24 * 2**30), "VRAM Total Used Memory (B)": str(4 * 2**30),
        "GTT Total Memory (B)": str(32 * 2**30), "GTT Total Used Memory (B)": "0"}})
    d = gpu.parse_rocm_smi(out)
    assert d[0].name == "Radeon RX 7900 XTX"
    assert d[0].total_mb == 24 * 1024 and d[0].free_mb == 20 * 1024 and d[0].util_pct == 7


def test_parsers_tolerate_garbage():
    assert gpu.parse_amd_smi("not json", "") == []
    assert gpu.parse_rocm_smi("") == []
    assert gpu.parse_nvidia_smi("") == []


@pytest.fixture(autouse=True)
def _reset_cache():
    yield
    gpu._cached = None
    gpu._cached_config = None
