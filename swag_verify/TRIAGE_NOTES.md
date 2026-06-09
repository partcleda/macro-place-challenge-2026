# Triage notes — swag minimum-functionality verification

Bar: ibm01 → 0-overlap VALID placement. Harness: air-gapped Docker, challenge
macro_place authoritative + team macro_place.* extras overlaid, PYTHONPATH=/submission,
1 GPU / 16 cores / 28g / 55-min cap. Results: swag_verify/results.tsv.

## Recovery tiers (run after main batch completes; re-runs need free GPUs)

### Tier 1 — bakeable pure-Python deps (one image rebuild + re-run)
Collect all MISSING_DEP modules that are pip-installable pure Python and bake into
the eval image, then re-run those teams. Seen so far:
- denoiseplace → `omegaconf`
- dragonfly → `sklearn` (scikit-learn)
(collect full list from results.tsv after batch)

### Tier 2 — entry-point fixes (re-run with explicit hint)
- hachimi → submission is `gradient_optimizer_submission.zip` (unextracted); unzip, re-detect.
- mlewand → code under `src/`, non-standard layout, ships eval_docker/Dockerfile; find entry.
- macrobioplacement → only `current_experiment.py`; check method signature / branch.

### Tier 3 — compiled-extension teams (build team Dockerfile if shipped, else BLOCKED)
- hoop_dreams → `dreamplace.ops.place_io.place_io_cpp` (uncompiled DREAMPlace)
- no_man_s_sky → `_placer_core` (compiled ext)
- k2hal → self-containment audit requires full macro_place from repo; build their Dockerfile.

### Tier 4 — CLONE_FAIL: retry URL variants, else mark inaccessible
- cloooooo (github.com/sfeirc/macro-place-challenge-2026) — 404
- makercode (github.com/Weiyet/macro-place-challenge-2026) — 404
- eth_zurich_student (github.com/BasilGrande/MacroPlaceChallenge2026) — 404

### Genuine fails (log, not recoverable as-submitted)
- goats → OpenROAD TCL flow, not a place(benchmark) Python submission.
- CRASH (API/version mismatch): figo (Benchmark.congestion_smooth_range), praveen_v
  (torch.Device n/a), combobulating (needs netlist_file metadata), pragnay (plc=None bug), ...
- INVALID: produces overlaps (genuine).
