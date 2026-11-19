"""Application settings (data directory, online services, GPU, search defaults).

Settings live in ``<data_dir>/settings.toml``. Everything that touches the
network is off by default – the program is offline-first.
"""
from __future__ import annotations

import os
try:                      # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w


@dataclass
class OnlineSettings:
    """Each online service must be explicitly enabled."""
    mpc_download: bool = False      # manual MPCORB / CometEls update button
    skybot: bool = False            # IMCCE SkyBoT cross-check of identifications
    astrometry_net: bool = False    # nova.astrometry.net fallback plate solver
    sky_images: bool = False        # DSS cut-outs on request (CDS hips2fits, STScI, NASA SkyView)
    twilight_dss_check: bool = True # with sky_images: "zdroj navíc" candidates compared with DSS2 automatically
    twilight_pgc: bool = True       # with sky_images: galaxies of HyperLEDA/PGC (VizieR VII/237) per field, cached
    mpcorb_url: str = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
    comets_url: str = "https://minorplanetcenter.net/iau/MPCORB/CometEls.txt"


@dataclass
class GpuSettings:
    backend: str = "auto"           # auto | cupy (CUDA i ROCm) | cuda | rocm | numpy
    device: int = 0
    memory_fraction: float = 0.6    # max share of GPU memory the pipeline may use


@dataclass
class AstapSettings:
    executable: str = ""            # path to astap / astap.exe ("" = search PATH)
    database_dir: str = ""          # folder with D50/D80/... ("" = ASTAP default)
    search_radius_deg: float = 10.0
    timeout_s: int = 120
    sip: bool = False               # pass -sip (only newer ASTAP versions support it)


@dataclass
class SearchSettings:
    """Default synthetic tracking parameters (overridable per job)."""
    rate_min_arcsec_min: float = 0.1
    rate_max_arcsec_min: float = 5.0
    step_px: float = 0.7            # max trail error over the sequence, px
    snr_threshold: float = 6.0
    near_threshold_margin: float = 0.5   # "druhá šance": detections this far below the threshold that pass every check go to a separate list (not the candidates); 0 = off
    segments: int = 3
    segment_z_max: float = 3.0      # max deviation of a time segment from the expected signal (σ)
    min_efficiency: float = 0.3     # slowest searched speed keeps ≥ this share of the signal
    frame_cap_sigma: float = 4.0    # max contribution of one frame to the sum (cosmics, hot pixels)
    tile_px: int = 1024
    background_mesh_px: int = 64
    bright_star_mask_sigma: float = 40.0
    max_candidates: int = 2000
    identify_tolerance_arcsec: float = 30.0
    comet_tolerance_arcsec: float = 180.0   # comets: wider position tolerance (orbits often off by arc minutes); motion must agree
    pointing_wcs_search_arcsec: float = 300.0  # mount-position WCS: search radius for its zero point on known objects
    forced_mag_limit: float = 18.5  # known objects up to this mag get a forced SNR measurement
    throughput_maps: bool = True    # per-frame transparency maps from stars (extinction, obstruction, cirrus)
    throughput_cell_px: int = 256   # grid cell of the maps (native px)
    throughput_min: float = 0.5     # frame pixels below this throughput are left out of the search
    throughput_drop_frac: float = 0.5   # a frame with this fraction of the field below throughput_min is dropped
    twilight_throughput_min: float = 0.3   # frames with the Sun above −18°: keep them (weighted) down to this throughput
    ucac4_complete_mag: float = 15.5   # a static peak this bright must be in UCAC4; none there → not a star (0 = off)
    cover_local_ratio: float = 0.75   # "zakryté místo": track throughput below this × field median of the frame
    frame_weighting: bool = True        # weight frames by their throughput in the sum (optimal; maps needed)
    auto_accept_known: bool = True  # detections identified with a known object are marked accepted
    slow_star_motion_fwhm: float = 1.2  # track shorter than this (FWHM) …
    slow_star_ratio: float = 4.0        # … on a star this many times brighter than the object = star residual
    star_rescue_mover: float = 0.3  # "reziduum hvězdy" kept as candidate if the mover test gives at least this
    star_beside_fwhm: float = 1.6   # … or if the star is ≥ this × FWHM beside the track and the track is consistent
    star_beside_chi2: float = 1.3
    ml_rescue: float = 0.5          # classifier score that returns a candidate rejected only for a nearby star
                                    # (0.9 → 0.5, 11. 11.: injections near stars 31 → 100 of 144 back, unknowns 0 → 10 of 1374 in 15 jobs)
    # cosmic rays (8-neighbour test, refused when implausibly large) and satellite trails
    mask_transients: bool = True
    static_var_reject: float = -1.0 # mover vs static-variable test: below → rejected ("proměnná hvězda")
    static_var_warn: float = 0.0    # below → warning "spíš proměnná hvězda" (confidence low)
    # object narrower than the stars (hot/warm pixels, cosmic remnants): forward-modelled width test
    sharp_ratio: float = 0.65       # object FWHM below this × star FWHM …
    sharp_dchi2: float = 25.0       # … and the star PSF fits worse by Δχ² ≥ this → "ostřejší než hvězdy"
    sharp_warn_dchi2: float = 9.0   # … Δχ² ≥ this → warning only "užší než hvězdy?"
    sharp_keep_snr: float = 15.0    # bright track (SNR ≥) …
    sharp_keep_chip: float = 0.5    # … whose detector-fixed path carries < this share: never "ostřejší"
    check_snr_min: float = 3.5      # independent re-measurement of the track (median template): SNR at least …
    check_snr_frac: float = 0.45    # … and this share of the search SNR
    check_frames_min: float = 0.55  # share of frames that must carry signal (> 0.5 σ)
    static_rate_factor: float = 0.7 # PSF-fit rate below factor × slowest searched rate = static residual
    antialias_fwhm_px: float = 2.7  # undersampled stars are smoothed to this FWHM before resampling; 0 = off
    antialias_below_fwhm_px: float = 2.3  # ... only when the measured FWHM is below this (interpolation error > ~1.5 %)
    mask_bad_columns: bool = True   # detect weak bad columns/rows of the detector and mask them in every frame
    inpaint_hot_pixels: bool = True   # hot / warm pixels replaced by the median of their neighbours (NaN spread by warp cost bright objects their cores)
    velocity_noise_norm: bool = True   # divide each velocity's SNR map by its empirical noise (MAD over the tile, ≥ 1)
    drift_chip_ratio: float = 0.8   # reject when the best detector-fixed path keeps ≥ this share of the linear-track signal …
    drift_chip_overlap: float = 0.6 # … and stays within 1 FWHM of the track in ≥ this share of frames (warm pixels, drift-like dither)
    chip_pattern: bool = False      # subtract a detector-fixed pattern before the search – OFF: on real 161P data it added detections (93 -> 135)
    # empirical noise threshold per tile: the same search + checks on time-shuffled frames
    # (movers scrambled, noise and star residuals kept); 0 permutations = fixed threshold only
    null_permutations: int = 0      # OFF (7. 11.): in dense tiles it saw 0–2 false where 135–270 passed, elsewhere it cut 12 of 101 found injections; ranking by injections (rank.py) replaces it
    null_false_per_mpx: float = 1.0 # expected false candidates (after all checks) per Mpx above the tile threshold
    null_sample_per_bin: int = 20   # null detections fully vetted per 0.5-σ SNR bin (all above threshold + 2)
    null_max_add: float = 4.0       # tile threshold at most snr_threshold + this
    inject_n: int = 100             # synthetic movers added to the frames (completeness, per-run ranking, classifier)
    inject_comet_fraction: float = 0.2   # share of injections that are comets (nucleus + 1/ρ coma 4–120″, condensed … diffuse)
    # "zdroj navíc" alone (≤ 6 frames): static injections – point sources (nova, comet in outburst)
    # and comets with a coma that do not move within the series; they measure what the catalogue
    # comparison finds and which rule loses real objects. 0 = off
    twilight_inject_n: int = 60
    twilight_inject_snr_min: float = 8.0     # matched-filter SNR of the mean stack (CTA-N 324: the limit ~SNR 15–20)
    twilight_inject_snr_max: float = 60.0
    twilight_inject_comet_fraction: float = 0.5
    search_mode: str = "auto"       # auto | tracking | twilight („zdroj navíc“ proti UCAC4) | both
    twilight_max_frames: int = 6    # auto: at most this many frames (or a shorter series) → twilight
    twilight_max_span_min: float = 0.0  # > 0: also a series shorter than this (with ≤ 2 × max_frames frames)
    keep_work: bool = False         # keep work/reg.npy (registered frames) after the job – diagnostics, ~1 GB per job
    inject_rank_keep: float = 0.9   # per-run ranking (rank.py): keep this fraction of injected movers, set aside lower-scored unknowns; 0 = off
    inject_rank_keep_soft: float = 0.97  # between the two cuts: "hraniční skóre podle injekcí" (own group, look through last)
    inject_snr_min: float = 5.0
    inject_snr_max: float = 15.0


@dataclass
class WatchSettings:
    root: str = ""
    instrument: str = ""            # "" = choose by header match rules
    frames_per_set: int = 20
    settle_seconds: int = 60        # no file change for this long = set complete
    poll_seconds: int = 15
    enabled: bool = False


@dataclass
class Settings:
    data_dir: str = ""
    observatory_code: str = "XXX"
    observer_lat_deg: float = 50.0
    observer_lon_deg: float = 15.0
    observer_alt_m: float = 300.0
    mpcorb_max_age_days: int = 7
    ephemeris: str = "auto"         # auto | builtin | path to .bsp / DE folder
    cpu_workers: int = 0            # threads for frames/tiles (0 = all CPU cores, limited by free RAM)
    field_split: bool = True        # split a folder with several fields of view into one job per field
    field_tolerance_fov: float = 0.25   # same field if pointings differ < this × smaller FOV side
    ucac4_dir: str = ""             # local UCAC4 (folder with u4b/z001…z900) for approximate photometry; "" = off
    online: OnlineSettings = field(default_factory=OnlineSettings)
    gpu: GpuSettings = field(default_factory=GpuSettings)
    astap: AstapSettings = field(default_factory=AstapSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    watch: WatchSettings = field(default_factory=WatchSettings)

    # ---- paths -------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.data_dir)

    @property
    def instruments_dir(self) -> Path:
        return self.root / "instruments"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def catalogs_dir(self) -> Path:
        return self.root / "catalogs"

    def ensure_dirs(self) -> None:
        for p in (self.root, self.instruments_dir, self.jobs_dir, self.catalogs_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ---- persistence -------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("data_dir", None)
        return d

    def save(self) -> None:
        self.ensure_dirs()
        (self.root / "settings.toml").write_bytes(tomli_w.dumps(self.to_dict()).encode())

    @classmethod
    def from_dict(cls, data_dir: str, d: dict) -> "Settings":
        sub = {"online": OnlineSettings, "gpu": GpuSettings, "astap": AstapSettings,
               "search": SearchSettings, "watch": WatchSettings}
        kwargs = {}
        for k, v in d.items():
            if k in sub and isinstance(v, dict):
                known = sub[k].__dataclass_fields__
                kwargs[k] = sub[k](**{kk: vv for kk, vv in v.items() if kk in known})
            elif k in cls.__dataclass_fields__ and k != "data_dir":
                kwargs[k] = v
        return cls(data_dir=data_dir, **kwargs)

    def update(self, d: dict) -> None:
        """Merge a (partial) dict from the GUI into the settings."""
        new = Settings.from_dict(self.data_dir, _deep_merge(self.to_dict(), d))
        self.__dict__.update(new.__dict__)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def default_data_dir() -> Path:
    env = os.environ.get("SYNTHTRACK_DATA")
    if env:
        return Path(env)
    return Path.home() / "SynthTrackData"


def load_settings(data_dir: str | os.PathLike | None = None) -> Settings:
    root = Path(data_dir) if data_dir else default_data_dir()
    path = root / "settings.toml"
    if path.exists():
        s = Settings.from_dict(str(root), tomllib.loads(path.read_text(encoding="utf-8")))
    else:
        s = Settings(data_dir=str(root))
        s.save()
    s.ensure_dirs()
    return s
