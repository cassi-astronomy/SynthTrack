# AGENTS.md – SynthTrack

Instructions for AI coding agents (and humans) working on this repository.
Read this file before changing code. User-facing text is **Czech**; code,
identifiers, docstrings and commit messages are **English**.

## What this is

SynthTrack is a GPU-capable **synthetic tracking** pipeline that finds
asteroids and comets in sequences of short exposures (shift-and-add along
hypothesised motions), identifies them against MPCORB/CometEls and lists
unknown candidates. It runs on **Windows and Linux**, offline by default,
with a local web GUI (FastAPI + vanilla JS) and a CLI.

## Quick commands

```bash
pip install -e ".[dev]"            # + ".[cuda12]" for NVIDIA GPUs (CuPy)
pytest                              # must stay green (≈25 s, no network, no GPU needed)
ruff check synthtrack tests --select F,E9
synthtrack demo && synthtrack run --instrument demo --lights ~/SynthTrackData/demo/lights
synthtrack serve                    # GUI on http://localhost:8765
synthtrack app                      # same GUI in a native window (needs ".[desktop]")
synthtrack gpu                      # which backend is active
```

## Layout

| Module | Responsibility |
|---|---|
| `settings.py` | `Settings` dataclasses, `settings.toml` in the data dir. Online services off by default. |
| `instruments.py` | Instrument profiles (TOML): optics, camera, masters folder, per-instrument MPC code + observatory coordinates, FITS key map, filter aliases, crop, site-from-header, registration options. |
| `fitsmeta.py` | FITS loading (float32, BZERO), header normalisation (`FrameMeta`), `parse_section`, sanity of temperatures. |
| `fields.py` | Split a folder into fields of view by pointing (header WCS centre, else RA/DEC keys); tolerance = `field_tolerance_fov` × smaller FOV side; `create_field_jobs` = one job per field (`Job.files`, `Job.field`); `find_light_dirs` / `create_tree_jobs` = pre-sorted fields in sub-folders (one job per light folder, natural order, calibration/hidden/master folders skipped, each folder still split by pointing). Used by GUI, CLI and watcher. |
| `masters.py` | Master library scan + automatic dark/flat/bias selection with human-readable reasons. |
| `calibration.py` | dark/flat, saturation mask, background mesh model ("artificial flat"), σ-normalisation, `nanmedian_last`. |
| `registration.py` | star detection, offset voting, polynomial (order 3) ref↔frame fits, fast coordinate maps, `warp`. |
| `search.py` | **core algorithm**: tile preparation, noise model, `TrackNorm`, `search_tile`, `refine`, `vet`, sub-stack astrometry. |
| `tracking.py` | shared types (`Candidate`, `VelocityGrid`), FWHM estimate, dedupe, phase-correlation fallback. |
| `pipeline.py` | orchestration of one job (memmaps, threads, tiles, identification, outputs). |
| `nbody.py`, `ephem.py`, `identify.py` | orbit propagation (N-body + GR), planetary ephemerides, field prediction and matching. |
| `mpcdata.py` | MPCORB / CometEls parsing, download (only if enabled), `.npz` cache. |
| `jobs.py`, `watcher.py` | persistent jobs + single worker queue; automatic mode (polling). |
| `web/` | REST API (`app.py`) and GUI (`static/`, no build step). |
| `desktop.py` | `synthtrack app`: uvicorn in a thread + pywebview window; `DesktopApi` (native folder/file/save dialogs, exposed as `window.pywebview.api`; keep non-API attributes `_private` or pywebview will try to expose them); `synthtrack shortcut` (.lnk / .desktop). `app.js` must keep working in a plain browser: guard with `desk()`. |
| `simulate.py` | synthetic sequences for tests (optional `throughput(i, xx, yy)` = extinction/obstruction). |
| `defects.py` | weak bad columns / rows (and partial columns from charge traps) from block medians of the detector-frame temporal median; `LineDefects.apply` masks them. |
| `throughput.py` | per-frame throughput maps A_i(p) from star photometry on the registered frames (coarse grid, bilinear); `spread()`, `describe()`. |
| `inject.py` | injection–recovery: `plan` (log-uniform speeds, nominal SNR), `add_to_frames` (into `reg.npy`), `match`, `summarise`. |
| `confidence.py` | readable confidence high/medium/low/none + reasons from the vetting numbers (and `ml_score`); computed on every `GET …/candidates`, so old jobs get it too. |
| `sky.py` | DSS/2MASS reference cut-outs on request (`online.sky_images`): STScI DSS → NASA SkyView → CDS hips2fits/mirror; cached in `img/sky_*`. The GUI renders the local crop with the full 2×2 WCS matrix (arbitrary rotation/scale), not a nearest-90° transform. |
| `classifier.py` | candidate classifier (L2 logistic regression, NumPy) trained from user labels, known objects and injections across all jobs (archived in `<data>/models/examples/` before a job is deleted); `ml_score`. Cross-validation holds out whole jobs, never random candidates from the same field. Common model `<data>/models/classifier.json` + `classifier-<instrument>.json` once an instrument has ≥ 8 real / 30 rejected examples; `load(data_dir, instrument)` falls back to the common one. |

## Pipeline (pipeline.py) – data flow

0. (before the job) `fields.create_field_jobs` splits a multi-field folder into one job per field; the job carries its explicit `files`. A job without `files` that still sees several fields processes only the largest one and warns – never stack different fields.
1. frames (`job.files` or the whole folder) → group by (exptime, filter, binning, size) → sort by mid-exposure time
   Observatory code and fallback coordinates come from this instrument profile; FITS coordinates may override them. Legacy global settings are fallback-only for old profiles.
2. masters selected for the reference (middle) frame; crop to `DATASEC` (`crop = "auto"`)
3. **pass 1** (threads): calibrate → background flatten → σ units → NaN for bad pixels → `work/norm.npy` memmap; star lists; header WCS (sliced to crop/binning)
4. registration by stars (polynomial, handles distortion and dithering); phase correlation fallback
5. hot pixels: detector-fixed outliers in the temporal median of the *dithered* frames → NaN; bad columns/rows (`defects.LineProfiler`, always) → NaN
6. **pass 2** (threads): `warp` every frame onto the reference grid → `work/reg.npy`
7. reference WCS (header, else ASTAP); throughput maps (`throughput.measure`, used only if the field is not uniform); optional injection of synthetic movers into `reg.npy`
8. **tiles** (threads; GPU search serialised by a lock): `prepare_tile` → `search_tile` → peaks at threshold−1 → `refine` (clipped) → threshold → `refine` (raw, half step) → `vet` → sub-stack astrometry → images
9. dedupe, sky coordinates, N-body prediction of known objects, matching, forced SNR on predicted tracks
10. injection matching (status `injected`), classifier score (`ml_score`) if a model exists; `candidates.json`, `known.json`, `frames.json`, `injections.json`, `img/` (incl. `*_frames.png` strips for the GUI viewer); `work/` is deleted

Memory is bounded by tiles, not by sensor size. Never load the full cube
into RAM. Release memmaps (`x = None`, `gc.collect()`) before deleting their
files – Windows cannot delete a mapped file.

## Algorithmic invariants – do not break

* **Template = σ-clipped mean over time** (not median). The detection
  statistic is the GLRT for an unknown static sky.
* **Noise of the sum is `TrackNorm.sigma(speed, subset)`** =
  √(|S| − (1/n)Σ_{i,j∈S} ρ_ij), ρ_ij = exp(−v²Δt²/4σ²). Replacing it by √n
  silently loses slow movers (factor 2–3). `expected()` / `frame_weights()`
  describe the signal share per frame; segment tests must use them.
* Frames entering the stack are **unit noise**: temporal robust σ per pixel
  (+ `noise_model` a + b·S + c·S² for star cores) and an empirical MAD
  re-normalisation after the matched filter (resampling correlates pixels).
* With throughput maps the template model is d_i = A_i·S (`search.amp_clipped_mean`,
  weighted LS), the residual is d_i − A_i·S and frame pixels with A_i <
  `throughput_min` are zero in the search (a partly blocked pupil also moves
  and deforms stars). Without maps (uniform field) the old per-tile scalar
  scaling is used unchanged – keep both paths.
* Undersampled data (measured FWHM < `antialias_below_fwhm_px`, default 2.3 px)
  are smoothed to `antialias_fwhm_px` (default 2.7 px; `registration.antialias`,
  unit noise kept) before `warp`; spline resampling of 2-px stars changes their
  shape with the sub-pixel phase (dark rings, ±30 % peaks) and leaves star
  residuals. Do not smooth more than needed: smoothed noise forms star-sized
  grains (twice the detections on CTA-S1 at FWHM 2.4 px, harder visual review).
* Bad columns/rows (`defects.py`, `mask_bad_columns`): block medians of the
  temporal median in *detector* coordinates; lines ≳ 0.1 σ are NaN in every
  frame before `warp`. Weak columns move with the dither/drift and are fitted
  by straight tracks (CTA-S1 field_02: 82 → 7 candidates in one tile).
* `warp` keeps masked pixels masked (NaN). In `prepare_tile` only holes next
  to bright stars (saturated cores) are dilated into `sat`; isolated holes
  (hot pixels) are just no data in that frame.
* A detection whose PSF-fit rate is below `static_rate_factor` × the slowest
  searched rate is a static source (variable star / star residual) and is
  auto-rejected ("téměř statický zdroj").
* Per-frame contribution is clipped to ±`frame_cap_sigma` for the search
  only; velocity refinement and all reported numbers use `f_raw`.
* `refine` samples tracks **bilinearly**. Integer rounding biases slow
  movers to the minimum speed.
* Reported positions/rates come from **PSF fits in sub-stacks**
  (`measure_substacks` / `fit_motion`), not from the search velocity.
  The sub-stacks are **star-subtracted** with a local template that leaves
  out, per pixel, the frames where the object is within 1.5 FWHM (at least
  n/4 farthest frames kept). Fitting raw frames latched on to neighbouring
  stars in dense fields (1.3 px median error, rate off by ~80 %). The fit is
  accepted only if its track keeps ≥ 80 % of the search SNR on `f_raw`.
* The star-free (local-template) fit cannot reject star residuals: its
  template follows the hypothesised track, so a varying star looks like a
  mover along that very track (31. 10.: passing 397 → 1219 in CTA 304/5).
  Star residuals are decided from the static sky instead: motion < 1.2 FWHM
  over the sequence and the static sky under the track > 4 × the object's
  own peak → "pomalý u hvězdy" (hard); "astrometrie nejistá" is hard;
  ≥ 25 % of the frames with throughput < 0.6 on the track and SNR < 10 →
  "zakryté místo" (1. 11.: the first version used the *minimum* over frames;
  one dim twilight frame then rejected every SNR < 10 detection – 1466 incl.
  5 known asteroids in Coihueco 305/field_01). On 2883 false / 26 real
  (fields 304/4, 304/5) these removed 1855 + 402 false, no real one.
* `search.independent_check`: second measurement of a passing track with a
  per-pixel *median* template (no search, no matched-filter normalisation):
  SNR and share of frames with signal. Compared with the speed efficiency
  `TrackNorm.efficiency(v)` (the median also removes part of a slow mover):
  reject if SNR/eff < 3.5 or SNR < 0.45·SNR_search·eff ("nepotvrzeno
  nezávislým měřením"), or eff ≥ 0.6 and < 55 % frames with signal.
  Simulation: 0–3/40 real movers rejected (0.7–4.5 FWHM, SNR 7–14).
* `search.static_vs_mover`: mover vs. static source with variable flux, both
  with free non-negative per-frame flux (real objects vary a lot in low or
  obstructed series), only the geometry differs; the static position is
  searched only within 1.5 FWHM of the signal-weighted track centre (else a
  variable star elsewhere on a long track rejects real fast movers – see
  `test_pipeline_obstruction_static_and_injection`). T < −1 → "proměnná
  hvězda" (hard), T < 0 → "spíš proměnná hvězda" (confidence low). 304/4–5:
  173 / 426 of 668 false, no real one rejected (Tisiphone, Nicole warned).
  Slow movers ≲ 1 FWHM are genuinely degenerate with a brightening star.
* "reziduum hvězdy" (vet) is overridden when `static_vs_mover` ≥ 0.3
  (`star_rescue_mover`); after scoring, a detection rejected only for
  star-related flags with `ml_score` ≥ 0.9 (`ml_rescue`) is returned
  ("vráceno klasifikátorem"). 305/field_01: Adelinda (T 0.63) and
  Fredegundis (ML 0.998, on a 19× brighter star) – exactly the two known
  objects the safety net reported.
* A slow object's own light can enter the temporal static template. Never call
  that a bright star merely because the static mask overlaps the predicted
  track. `pomalý u hvězdy` requires the static peak to be spatially offset by
  at least 0.6 FWHM; a centred peak is passed to mover/static and independent
  checks. Forced photometry labels the centred case `pomalý zdroj ve statické
  šabloně`, not `jasná hvězda na stopě`.
* `transients.py`: destructive transient masking is optional and defaults to
  `mask_transients = false` because completeness has priority. The search
  already clips each frame to ±4σ and rejects single-frame artifacts during
  vetting. If explicitly enabled, a cosmic must exceed all 8 neighbours and
  both cosmic/trail masks have strict area fuses; never remove a large image
  fraction merely to shorten the rejected list.
* Safety net: identified known objects rejected by vetting → warning +
  `summary.known_rejected` (GUI Souhrn).
* Static-star crowding is reduced only in the GUI: automatic rejects carrying
  the same `static_star_x/y` are a lossless, expandable group. Never cap or
  delete rejected detections; an unknown asteroid/comet may overlap a star.
* Remaining false detections in Milky-Way fields are mostly SNR 6–7 noise
  tails (non-Gaussian). Planned: adaptive per-tile threshold from an
  empirical null (search on time-scrambled frames).
* In `prepare_tile` pixels blocked by the throughput map are excluded from
  the empirical MAD re-normalisation (zeros there shrank the MAD and
  amplified the noise of partly obstructed frames).
* Always honour the configured minimum searched speed. `min_efficiency`
  only triggers a sensitivity warning; it must never raise the lower bound.
  CTA-S1 field 3 contains real 0.13–0.24 arcsec/min asteroids, and silently
  changing a requested 0.10 to 0.37 lost them and falsely labelled the
  boundary detections as stars. Low efficiency is reduced SNR, not proof
  that an object is static.
* Known-object positions are **topocentric astrometric ICRF** (light time,
  no aberration) = MPC convention. N-body with planets, Moon, GR. Never put a
  dummy body into an ephemeris table (see the "old epoch" regression test).
* The search code is written against the NumPy API so that `xp = cupy` runs
  it on the GPU. Do not call SciPy inside `search_tile`; keep GPU-side code
  to `xp` operations.

## Conventions

* Python ≥ 3.10, type hints, dataclasses; no new heavy dependencies without
  a strong reason (current: numpy, scipy, astropy, fastapi, uvicorn,
  requests, tomli-w, pillow, jplephem; optional cupy, astroscrappy).
* Offline-first: any network access must be behind a `settings.online.*`
  switch that defaults to `False`.
* All user-visible strings (GUI, logs via `ctx.log`/`ctx.warn`, errors in
  `PipelineError`) in Czech; explain *why* (e.g. which master was chosen and why).
* Cross-platform: `pathlib`, no shell calls except ASTAP/nvidia-smi, no
  POSIX-only APIs without a Windows branch.
* Threads, not processes, for per-frame/per-tile work (`pipeline._parallel`);
  avoid Python-level loops over pixels/stars in hot paths (they hold the GIL —
  e.g. `np.nanmedian` with NaNs; use `nanmedian_last` / `temporal_stats`).
* Keep functions pure where possible; the pipeline owns I/O and logging.

## Tests and validation

* `tests/test_precision.py` – integrator vs JPL DE421, geometry vs astropy,
  end-to-end vs **JPL Horizons** fixtures (`tests/data/horizons_*.json`, < 1″,
  currently < 0.05″). Do not loosen tolerances to make a change pass.
* `tests/test_wide_field.py` – distortion+dither registration, slow-mover
  detection and the TrackNorm advantage, header quirks, old-epoch regression.
* `tests/test_fields.py` – 5-field interleaved mosaic with dithering, RA 0h wrap and
  sexagesimal mount coordinates without WCS, missing coordinates, per-field jobs,
  pipeline safety net, watcher; pre-sorted sub-folders (GUI API, watcher, folder
  that still mixes fields).
* `tests/test_throughput_ml.py` – throughput maps vs truth, residual removal
  with maps, injection plan/match, classifier training/scoring, frame-strip
  round trip, end-to-end with an obstructed corner + injections.
* `tests/test_pipeline.py` – synthetic end-to-end, known-object
  identification, master selection, MPC80 format, watcher, REST API.
* Reference real dataset (not in the repo, user's folder
  `SyntheticAstronomy/CTA/field_01`, 18 × 30 s, 135 mm, 14.1″/px). Expected
  with default settings: all of (25) Phocaea SNR≈35, (983) Gunila ≈9.4,
  (1660) Wood ≈8.1, (1102) Pepita ≈6.3 detected and passing vetting,
  ~50 candidates in total, O−C 1–9″, ~4 min on 2 CPU cores. Any change to
  `search.py` / `calibration.py` / `registration.py` must be checked against
  this dataset when available; report SNR/rank of the four objects.
* Second real dataset (user folder `SyntheticAstronomy/CTA/sky/field_01`,
  `field_02`, 2026-10-25, 19 × 30 s, airmass 3.3 → 2.0, top-right corner
  blocked in the first ~5 frames, strong extinction gradient). Before the
  throughput maps: field_01 123 passing (117 false), field_02 82 (81 false);
  ~53 / 20 of them were static star residuals in the blocked corner.
  Expected detected: (6) Hebe, (52) Europa, (12) Victoria, (130) Elektra,
  (584) Semiramis, (747) Winchester, (376) Geometria (SNR≈7) in field_01,
  (176) Iduna (SNR≈7.5) in field_02. (357) Ninina has forced SNR≈4.9 –
  below the threshold, not a bug.
* Profiling: `job.json → summary.timings_s` per stage.

## When adding features

* New plate solver → add to `astrometry.SOLVERS`, keep ASTAP default.
* New GPU backend (e.g. AMD) → `gpu.py` returning an object with the NumPy
  API; nothing else should need to change.
* New vetting test → add a flag string (Czech) in `search.vet`, decide if it
  is "hard" (auto-reject), show the metric in the GUI card (`app.js`).
* New output format → `export.py` + an endpoint in `web/app.py`.
* Classifier features live in `classifier.FEATURES`; changing them invalidates
  stored models (`load` returns None → user retrains). Keep it NumPy-only;
  an MLP/CNN (on the `*_frames.png` strips) only once there are hundreds of
  positives – injections provide them.
* GUI animation: the server stores raw frames (8-bit asinh strip); sub-stacks,
  black/white/gamma are computed in `app.js` (`anim*` functions). Do not bake
  a stretch into the strip.
* Planned work (see README "Co dál"): fused CUDA kernel, Gaia DR3 astrometric
  refit + photometry, satellite-trail masking, comet (diffuse) filter,
  MLP/CNN classifier.
