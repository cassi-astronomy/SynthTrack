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
pip install -e ".[dev]"            # + ".[cuda12]" NVIDIA / ".[rocm]" AMD on Linux (CuPy)
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
| `masters.py` | Master library scan + automatic dark/flat/bias selection with human-readable reasons. Dark without EXPTIME: exposure from the file name (`dark_60s`), else used with a warning (never a hard stop). `manual_selection` = user-chosen file (geometry must match, everything else only warns). |
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
2. masters selected for the reference (middle) frame (`_select_masters`: job params `calib_dark` / `calib_flat` = `auto` | `none` | file path; dark `none` still subtracts a master bias when one exists); crop to `DATASEC` (`crop = "auto"`)
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
* Hot pixels with a small/clustered dithering (`defects.chip_fixed`): the
  chip-coordinate median keeps stars when they sit on the same pixel in > n/2
  frames (161P/Hartley-IRAS, Coihueco 6. 11. 2026: 3 dither groups 2–4 px
  apart, a 25 σ star left 9 σ in the median and was masked in every frame).
  A pixel > 6 σ is masked only if it is bright at the same *chip* position in
  more frames than at any fixed *sky* position (registration offsets),
  F_chip − F_sky ≥ 0.2. Real data: 2092 hot / 6775 star pixels in 1024².
* Detector-fixed pattern (`search.subtract_chip_pattern`, `chip_pattern`): the
  median of the filtered frames in *detector* coordinates (registration
  offsets per tile, bilinear) is subtracted from every frame before the
  search. With a small/drift-like dithering flat/dark residuals and warm
  pixels move over the sky like an object at the drift velocity (161P,
  Coihueco: 0.3 σ/px fixed pattern after calibration, split-half r = 0.65;
  ~80 % of 2 308 passing candidates within 0.2 px/min of the drift
  velocity). Emulation on the calibrated frames: track-SNR spread at the
  drift velocity 2.8 → 1.25, movers keep SNR (median ratio 0.97). Objects
  moving exactly with the dither drift are lost – inherent, dither more.
  **OFF by default (6. 11. 23:55):** on the pipeline's registered 161P cube
  the chip-coordinate median of f_raw has split-half r ≈ 0 (no common
  pattern; the 0.3 σ found earlier came from a crude raw calibration), and
  subtracting the noisy median adds a drift-coherent term: tile test
  93 → 135 detections. Do not enable without a split-half check.
* Empirical tile threshold (`search.noise_threshold`, `null_permutations`):
  the same search + vetting on time-shuffled frames estimates the false
  candidates per SNR; the tile threshold is the lowest SNR with expected
  false ≤ `null_false_per_mpx` × area. Detections below it keep a numeric
  record, flag "pod prahem šumu dlaždice", no checks and no previews.
  Caveat (simulation): a shuffle breaks time trends (seeing drift), so the
  null underestimates residuals that follow a trend – it only raises the
  threshold, never lowers it below `snr_threshold`.
* Time-shuffled noise threshold: **OFF by default (7. 11.)**. Real 161P run with
  111 injections: dense tiles 9–16 estimated 0–2 false per tile while
  135–270 passed (shuffle breaks time trends of star residuals), sparse tiles
  raised the threshold to 8–10 and cut 12 of 101 found injections.
* Per-run ranking by injections (`rank.py`, `inject_rank_keep = 0.9`,
  `inject_n = 100` by default): logistic model on 8 physical features,
  positives = passed injections, negatives = passed non-injected (PU),
  out-of-fold scores; unknowns below the cut that keeps 90 % of injections get
  "nízké skóre podle injekcí" (auto-rejected, previews kept). Identified known
  objects are never set aside; those under the cut are reported (the check on
  real objects). 161P: 1 567 → 276, AUC 0.93, comet 0.997. Do not add
  "independent check / SNR" or "raw / SNR": they fit the idealised injections
  but put the real comet under the cut (0.32).
* `chip_ratio` (`search.chip_path_ratio`, computed in `vet_full` for every
  vetted detection, rank feature): signal on the best detector-fixed path
  (sky = detector − registration offset, ±1 px) / signal on the linear track.
  161P had a cluster of 111 of 276 left at PA 270°, 1.2″/min (0.12, 0.13 px/min),
  lower half of the GSENSE4040 only, ratio median 0.85 vs injections 0.2 –
  warm/flickering pixels following the drift-like dither. With the feature:
  276 → 163 left, cluster 111 → 32, AUC 0.954, comet 0.99. Not a hard rule:
  ≥ 1.0 would also take 5 % of injections (those moving with the drift).
* `imaging.local_star_shifts(exclude=track)`: the object's track (disk
  max(6, size/8) px) is left out of the star cross-correlation. Without it a
  bright mover ((7) Iris 11.4 mag, 29P field) dominated it, the "star" shifts
  followed the asteroid, the animation froze it and the correct prediction
  cross seemed to run ahead. Strips also store `nan_on_track` per frame.
* Forced SNR note "data na stopě vynulována (maska; chybí ve k z n snímků)"
  when f_raw is exactly 0 on the track. `keep_work` (search setting) keeps
  work/reg.npy for diagnostics.
* Saturated-core holes must also contain pixels missing in ≥ n − n//6 frames
  (`coverage <= max(1, n // 6)`): saturation is fixed on the sky, its centre is
  never measured. (7) Iris 11.4 mag (29P field, unsaturated, ~140σ) had 3–10 px
  missing per frame within 3 px of its track; its own brightness made them a
  "saturated core" and the track was zeroed (forced SNR 0, not detected).
* rank.py features: object width and `chip_ratio` REMOVED (29P: injections are
  Gaussians, real objects narrower; dither drift ≈ the field's asteroid motion,
  injections random directions → all 3 identified asteroids scored 0.07–0.34).
  Safety: if a passed identified object scores under the cut, the cut is lowered
  below it. Now: 29P 1 130 → 337, knowns 0.88–0.99; 161P 1 567 → 300, comet 0.99.
  To get the features back: inject the empirical PSF and velocities drawn from
  the field's known objects.
* Realistic injections (inject.py): `empirical_psf` = median of isolated,
  unsaturated stars of the reference frame (sub-pixel recentred, sum 1),
  `add_to_frames(psf=…)`, flux from `psf_gain`; half of the injections take the
  motion of the field's known asteroids (±15 % speed, ±10°), predicted before
  injection (`_predict_known`, reused by identification). rank.py tries the full
  feature set (incl. object width and chip_ratio) and falls back to SAFE when an
  identified known object would fall under the cut (summary `feature_set`).
* Frames with ≥ `throughput_drop_frac` (0.5) of the throughput-map cells under
  `throughput_min` are dropped after the maps (reg_keep.npy, metas/stars/regs/mjd/
  maps.grid subset; never the reference). Coihueco 312 (dawn, low, 35 min): 6 of
  19 frames at 19–42 % – search zeroed them but TrackNorm, injection SNRs and
  vetting counted them: every detection "zakryté místo", injections found
  8–20/100, nothing passed. CTA-S1 the same day: 82–85/100 found, few unknowns.
* rank.py judges only detections with SNR ≤ 1.5 × the brightest passed injection
  (`snr_max_judged`); brighter ones are never set aside and are not part of the
  known-object safety check. 29P re-run (realistic injections): full set kept
  Thisbe/Xanthippe/Bertholda at 0.85–1.0, only (7) Iris (SNR 73) scored 0.56 →
  forced the safe set + a lowered cut (1 081 left); with the range limit 122.
  Same day: CTA-S1 3 fields 3–8 unknowns left, Coihueco 312 (frames dropped)
  injections 86–87/100 found, 11–13 unknowns.
* Throughput-weighted sum (`frame_weighting`, `_frame_weights`): per tile
  w_i = median throughput of frame i (maps measured on unit-noise frames, so
  A_i is the relative S/N and w = A is optimal); pt.f / f_raw are multiplied
  and `TrackNorm(weights=w)` uses σ² = Σw² − wᵀρw/n, E = Σ w_i(w_i − (ρw)_i/n).
  Uniform maps → w ≡ 1 (unchanged). Simulation (10 frames at 100 %, 5 at 30 %):
  SNR 2.70 → 3.12. Frames mostly under `throughput_min` are still dropped first
  (a frame missing most of the field cannot subtract the star field reliably).
* Two cuts in rank.py: `inject_rank_keep` (0.9) for the main list and
  `inject_rank_keep_soft` (0.97); between them flag "hraniční skóre podle
  injekcí" (GUI filter "hraniční (injekce)"), below "nízké skóre…". 29P: 122
  main, 153 borderline, 864 low. The 9 of 82 injections under the main cut
  were ambiguous, not only faint: SNR 6–11, chip_ratio 1.3–1.5 (moving with the
  dither drift) or one third of the series at throughput ~0.5.
* `keep_work`: the pipeline writes work/KEEP, `JobStore.cleanup_work` honours it
  (it deleted work/ in its own `finally` before).
* A hole counts as a saturated core only if its connected region touches
  finite static > 0.5 × `bright_star_mask_sigma`; NaN in the template is never
  "bright" by itself and an unsaturated hole is filled with 0 (empty sky).
  Treating NaN as bright zeroed ~200 px around every all-frames hole
  (161P: forced SNR 0.0, "jasná hvězda na stopě", comet not detected).
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
* `transients.py`: transient masking defaults to `mask_transients = true`
  (cosmic must exceed all 8 neighbours, strict area fuses for cosmics and
  trails). Do NOT switch it off by default: on CTA-S1 305/pole 1 (1. 11. 2026)
  saturated satellite trails without trail masking left NaN cores, the
  "holes near bright" rule in `prepare_tile` zeroed whole regions along them
  and Unitas, Tatjana and Atropos (SNR 23–58) were rejected as
  "jednosnímkový artefakt" with median ratio 0. With masking on all pass.
* Previews are written for EVERY detection, including automatic rejects – the
  user must be able to check every automatic decision. Do not skip them.
* Display of masked pixels (`imaging.fill_nan_display`): neighbour median,
  so saturated star cores stay bright and masked sky stays sky. Never fill
  NaN with one global value (median → "holey" bright stars; 0 → black holes).
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
* Lower speed limit = max(configured minimum, speed with `min_efficiency`).
  Do NOT search below it. There the object moves < ~1 FWHM, the static
  template absorbs it, and TrackNorm divides by the tiny white-noise sigma
  of such tracks, so star-subtraction residuals reach SNR 100–600 and bright
  slow asteroids lock onto the slowest grid speed. Removing the floor
  (Codex, 1. 11. 2026) on 305/pole 1–3: Metis 0.35, Libussa 0.49, Paulina,
  Pariana … detected at 0.11″/min and rejected ("astrometrie nejistá"),
  pole 3 got 4296 detections with SNR-600 star residuals falsely identified
  as 18–20 mag asteroids. Slower real objects (Thalia 0.20, Dione 0.14) are
  still found at the floor speed – a sub-pixel difference over the sequence.
* "reziduum hvězdy" is meant for a slow track on a star much brighter than
  the object (CTA-S1 304: ~9×). If `star_ratio` < `slow_star_ratio` the
  static peak is the object itself or a similar star; the mover-vs-static
  test decides (flags "objekt ve statické šabloně", "hvězda podobné
  jasnosti u stopy"). χ² limit for "nekonzistentní snímky" is 4 for
  SNR_raw ≥ 30 (2.5 otherwise).
* `prepare_tile` "holes near bright" (saturated cores) = pixel missing in
  ≥ max(3, n/3) frames. A pixel missing in 1–3 frames is a dithered bad
  column, hot pixel, cosmic or trail; treating it as saturation zeroed
  Bamberga, Dembowska, Nephthys (11–13 mag, not saturated). `vet` skips
  frames without data at the track point (`PreparedTile.valid`); χ² is not
  used for bright (SNR_raw ≥ 20) tracks moving < 1.5 FWHM.
* `mask_cosmics` must stay O(3 frames) in memory (in-place maximum); the
  list-of-8-rolls version needed 1 GB per 4k frame and thread (MemoryError).
* Known objects get a frame strip along the predicted track
  (`known[].images.frames`); stacks and animation are built in the GUI.
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
* GPU backends: CuPy covers NVIDIA (CUDA) and AMD (ROCm/HIP) with one code
  path; `Backend.platform` is "cuda" | "rocm" | "cpu". Keep GPU code to the
  NumPy API (no raw CUDA C, no cupyx-only calls) so both platforms keep
  working; a fused kernel must have a NumPy reference and run under HIP too.
  A non-CuPy backend (e.g. PyTorch for AMD on Windows) → `gpu.py` returning an
  object with the NumPy API; nothing else should need to change.
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

## Object width and mount-position WCS (2. 11. 2026)
- `search.sharpness`: object FWHM from the star-free residual stack, **forward-modelled** through the same
  template subtraction (`residual_stack`, shared with `measure_substacks`) – slow objects partly lost in
  their own template must not look sharp. Reject „ostřejší než hvězdy“ only when width < `sharp_ratio`
  (0.65) × star FWHM **and** Δχ² ≥ `sharp_dchi2` (25); 9–25 = warning „užší než hvězdy?“. Simulation:
  1 of 1250 real objects (FWHM 2–4.2, SNR 5.5–40, 0.4–6 FWHM motion) rejected, 0 of 464 at FWHM ≤ 2.8;
  warm pixels following a pointing drift 97–100 % at FWHM ≥ 3, ~30 % at FWHM 2 (models indistinguishable –
  that is the built-in protection for short lenses). Do not replace it by a plain width cut.
- `astrometry.header_wcs_kind`: INDI-style WCS (CRVAL = mount RA/Dec, CRPIX centre, no CD/SIP) = "pointing".
  Then ASTAP (FOV from the *file's* binning) or `_pointing_offset` – CRPIX shift from bright known objects
  (velocity vector must agree; one object only if unambiguous). Summary `wcs_source`, `wcs_shift`.


## Coihueco / FRAM Test1–4 (8. 11. 2026)
- Weak warm pixels (`defects.chip_fixed_weak`): pixels with chip median between max(1.5, 6·1.2533/√n) σ and 6 σ.
  A frame is *unexplained* when the chip pixel is above 0.5 × its median but the static sky under it
  (median over the other frames of the same sky point) is not; chip-fixed when ≥ 50 % of frames are
  unexplained. `chip_fixed` (3 σ level, 3×3 maximum) cannot be used below 6 σ: any star within the dither
  range satisfies its sky hypothesis. 11P (28 × 60 s, 11 px linear drift): 7 400 of 11 600 passing
  detections moved exactly with the drift on single 2–6 σ pixels; tile emulation 1357 → 371 detections,
  injections and known objects unchanged.
- "pohyb s driftem čipu" (hard): when the per-tile registration offsets are a line within 1 FWHM
  (`_drift_velocity`), a detection within 1 FWHM/series of the drift motion with `chip_ratio` ≥
  `drift_chip_ratio` (0.9) is rejected. 11P: 8 090 of 11 582 unknowns, 0 of 21 known, 0 of 82 injections;
  in the randomly dithered Test1/2 fields (residual ~10 px) the rule never applies. Do not apply chip_ratio
  alone there: with a 20 px random dither known asteroids reach chip_ratio 1.1–1.7 by chance.
- `sharpness` / `residual_stack` use `_nanmedian0` (one sort) instead of `np.nanmedian` (Python loop with
  NaNs) and a separable Gaussian: 320 → 110 ms per detection (was ~80 % of the vetting time).
- χ² ("nekonzistentní snímky") is trimmed (largest max(1, n/10) terms out, rescaled by `_trim_factor` so
  noise still gives 1): 11P SNR 39 had one frame at 23 σ instead of 7 σ → χ² 5.4 → rejected.
- Identification: velocity tolerance ≥ 1.2 FWHM over the series (slow objects partly sit in their own
  template, the sub-stack fit overestimates the rate ~2×: (1558) 0.38 vs 0.18″/min, (2218) 0.50 vs 0.27);
  comets `comet_tolerance_arcsec` = 180″ (11P 84″ off its CometEls prediction), motion vector still checked.
- Forced SNR is divided by the empirical noise of the same sum on 160 random tracks of the same velocity
  (`forced_noise`): for slow tracks the model σ is ~25 % low and the 5×5 offset search adds look-elsewhere
  ((3995)/(1773) showed 7–8 while the search map had 4.8). Slow, faint known objects (≈ 1 FWHM per series,
  16 mag in 12 × 30 s) are at SNR ≈ 5 – below the threshold, not a bug.
- Saturated movers (`_saturated_mover`): (2) Pallas 8.6 mag had its core NaN in 11 of 12 frames, forced
  SNR 0. Centroids of the masked region at the prediction must follow the predicted motion (≥ 60 % of
  frames, ≥ 1 px over the series); a saturated star's hole stays put. `known.saturated`, summary
  `known_saturated`, GUI "✓ přesycený".
- Approximate photometry (`_photometry`): ZP = V_pred + 2.5 log F from identified asteroids (SNR ≥ 8, motion
  ≥ 0.8 FWHM, comets excluded, MAD clipping, ≥ 3 objects); `c.mag`, `known.mag_measured`, summary
  `photometry`. 11P field: 21 asteroids, scatter 0.24 mag. Slow objects come out fainter.
- Comet injections (`inject_comet_fraction` 0.2): nucleus (0–60 % of the light) + 1/ρ coma 4–120″ softly
  truncated, convolved with the empirical PSF; nominal SNR via `profile_gain` (what the stellar matched
  filter sees). They are positives for rank.py and the classifier like any injection; summary
  `injection.comets`.
- Mixed frame groups: preview returns `groups`, the GUI asks which one (`params.frame_group` =
  `fitsmeta.group_key`); default and fallback = the largest group.
- GUI filter "hraniční skóre" (after "injekce"): unknown candidates between the two rank cuts – not injections.
- UCAC4 photometry (`ucac4.py`, setting `ucac4_dir`, GUI Data → Nastavení): zone files u4b/z001…z900,
  78-byte records, RA binary-searched per zone (no index). ZP = m_APASS + 2.5 log F of ≤ 400 isolated
  (no star within 3 FWHM brighter than m + 3), unsaturated (core finite in ≥ 70 % of frames) field stars,
  fitted with the same `_fit_gauss` as the detections on the median of the registered frames. Band from the
  filter (R/clear → r, V → V, B → B). Falls back to the known-asteroid ZP; with the catalogue the known
  asteroids become the check (`photometry.known_offset` = measured − predicted V).
- 9. 11.: the drift rule now uses `search.chip_path` – the detector-fixed path is anchored at the
  track point of every frame (±0.5 px) and `overlap` = share of frames where it is within 1 FWHM of
  the linear track; reject when ratio ≥ `drift_chip_ratio` (0.8) and overlap ≥ `drift_chip_overlap`
  (0.6). A dither drift-like only in part of the series (88P: 13 frames jumping 25 px, then 17
  drifting) made the old mean-position anchor give ratio 0 everywhere. `chip_ratio` (rank feature) is
  the new ratio. 88P field still left ~1 900 SNR 6–7 unknowns at the second-group drift velocity whose
  signal is in *all* frames and whose chip median is < 1.5 σ – not warm pixels; cause open.
- `_saturated_mover` requires a compact hole inside the window with bright wings (median ring ≥ 10 σ);
  holes at the field edge after a large dither made three 15–18 mag asteroids "saturated".
- rank.py: when the cut is lowered below a known object, the borderline band keeps its width below
  the new cut (before: 0 borderline, 03958/11P).
- Per-velocity empirical noise (`velocity_noise_norm`, `search_tile(scales=, noise_mask=)`): a 1/16
  subsample stack of every velocity gives its MAD noise; it is divided by the median of velocities of the
  same speed (±16 %) and the SNR map is scaled by max(1, ratio). Slow tracks are noisier in every
  direction (~1.2–1.3) – left alone, slow known objects keep their SNR ((1558) 6.18, (2218) 7.36 unchanged);
  a directional excess (88P: weak detector pattern seen through 17 drifting frames, σ 1.26 relative at the
  drift velocity) is scaled out: tile 307 → 182 detections before vetting, near-drift ones 3, injections
  and known objects kept. Stars (static > 1.5 σ, dilated 2 FWHM) are left out of the estimate.
  `Candidate.noise_scale`; snr, snr_raw, seg_snr are in units of that noise.
- `_sharp_rescue`: "ostřejší než hvězdy" is calibrated on the run's injections (empirical PSF, same
  resampling): cut = max(0.45, min(sharp_ratio, q02(inj width/star FWHM) − 0.05)); detections rejected only
  for sharpness above it become "užší než hvězdy?" candidates. 88P (FWHM 2.6–2.7, undersampled): 84 restored,
  incl. (1979) Sakharov; (3984) Chacos (0.6) stays below all injections.
- 9. 11. evening (88P/V run): hot/warm pixels are **inpainted** (median of the unmasked 8 neighbours,
  `defects.inpaint_pixels`, setting `inpaint_hot_pixels`) instead of NaN, and only > 20 σ pixels get the
  1-px dilation. With 1.5 % of the chip NaN, `warp` spread every masked pixel to ~4 output pixels and star /
  asteroid cores were missing in ~30 % of the frames (290 bright stars of one tile: 25–42 %); (149) Medusa,
  (419) Aurelia, (494) Virtus, (936) Kunigunde (12–15 mag, forced SNR 34–177) locked onto the slowest grid
  velocity and were rejected as "nekonzistentní segmenty / jednosnímkový artefakt".
- `prepare_tile`: a saturated-star hole is kept only within 3 FWHM + 4 px of its never-measured core. The
  field edge without coverage in part of the frames (large dither) was connected to a nearby saturated
  star and zeroed whole: (4298), (19707), (2664) forced SNR 0, "data na stopě vynulována".
- Cosmic rays are only *candidates* before resampling (`mask_cosmics(apply=False)`); after registration
  `transients.confirm_cosmics` masks a candidate only if nothing ≥ 0.3 × its peak lies within ±2 px in the
  time-adjacent frames. In frames with better seeing an undersampled star (or a slow asteroid) has a peak
  > 4.3× its neighbours and was masked whole: 21 % of the star cores per frame and (149) Medusa, (419)
  Aurelia, (494) Virtus in 3–5 frames each (88P/Coihueco 9. 11.) – segments like [6, 88, 0.7].
- Forced note "hvězda u stopy" only when the bright static pixel is not the object's own light
  (`_own_light`: on the track within 0.7 FWHM and not brighter than the object's median peak).
- vet: segments without any data frame are not judged and the amplitude comes from the frames with data
  ((4298) at the field edge, 13 of 30 frames without coverage: segments [0, 8.8, 7.4] → rejected).
- `_revet`: when the sub-stack PSF fit moves the velocity by > 1 FWHM over the series, the basic vetting
  (segments, χ², median ratio …) is repeated on the fitted track. A bright slow object sits partly in its
  own template and the GLRT prefers a slower velocity (smaller σ): (149) Medusa 12.4 mag was searched at
  0.13″/min, fitted 0.61″/min, and rejected on the search track (segments [56, 116, −5.6]; on the fitted
  track [133, 122, 86], no flag). Hill-climbing on f_raw does not help – the slow track has the higher SNR.
- Bright objects (snr_raw ≥ 30) whose star-free sub-stack fluxes agree (all fits ok, CV ≤ 0.2): "nekonzistentní
  snímky" is replaced by the warning "χ² vysoké, podstacky souhlasí" (`_bright_chi2_rescue`). χ² carries the
  crude matched-filter model of a slow object's self-subtraction: (419) Aurelia 12.5 mag χ² 4.5 with raw
  peaks 64–144 in every frame and sub-stack fluxes 368/359/398. confidence.py: χ² limits 4/3 for SNR ≥ 30,
  ignored when the sub-stack CV ≤ 0.15 (Medusa SNR 204 was "low" for χ² 2.7).
- `synthtrack/__init__.py` sets OPENBLAS/OMP/MKL_NUM_THREADS=1 (setdefault) before NumPy is imported: Python-thread parallelism × OpenBLAS pools crashed the process with "access violation" in numpy (sort, tensordot) on Windows, 8.–10. 11.
- `_substack_rescue` (replaces `_bright_chi2_rescue`): sub-stack fluxes corrected by the throughput of their frames; also turns "nekonzistentní segmenty" into a warning for SNR ≥ 10 and motion ≤ 2.5 FWHM ((441) Bathilde, 314/1 at dawn: segments 7.6/3.3/8.3, corrected fluxes 40/34/41).
- Twilight (`_twilight_frames`, Sun > −18° at the frame times): if every frame that would be dropped for low throughput is a twilight frame, `throughput_min` is lowered to `twilight_throughput_min` (0.3) for the run (drop rule, search mask, forced); throughput weighting does the rest. Night series and night clouds unchanged. 314/1 (10. 11.): Sun −17 … −11°, frames at 31–44 % now kept.
- 12. 11. `static_vs_mover`: both energies leave out the largest max(1, n/10) frame terms – an unmasked 48 σ
  hit 3 px from (3312) Pedersen's track in one of 30 frames gave the static model 2 343 vs 998 (T −1.35,
  "proměnná hvězda"); trimmed T +0.73.
- `_catalog_stars_reject` (needs UCAC4; radius 1.5 FWHM since 12. 11. night, applied to rejected ones too so the classifier rescue cannot return them – 133 did): a slow detection (≤ 3 FWHM over the series), not known, not injected,
  within 1 FWHM of a UCAC4 star at most 0.5 mag fainter than its measured mag (the detection is the star), or within max(1.5 FWHM, 2 px) of one at least 1 mag brighter (a residual; without photometry the detection counts as 15 mag) →
  "hvězda z katalogu UCAC4 (… mag)", rejected. 88P/Coihueco 11. 11.: 254 of 564 passing unknowns (9–13 mag
  stars at 0.26–0.36″/min, many with ML > 0.9), 0 of 64 injections, the one known on a star is exempt.
