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

## Pre-release audit (re-verification before sending the list out)

Counts moved 60/23/3 → **57/26/3** (the open PR's "59/24/3" prose was stale; the report is authoritative).

### Issue 1 — PASS-gate substring bug (FIXED in run_one.sh)
The gate `grep -qE 'proxy=[0-9].*VALID'` also matched `INVALID` (substring), so an
overlapping placement printed as `proxy=N ... INVALID (k overlaps)` was scored PASS.
Scanned all 60 prior PASSes for a genuine ` VALID ` line (a space before V, which
`INVALID` cannot satisfy): 57 are real, 3 are not. Reclassified to FAIL/INVALID:
- ilovekiro — DREAMPlace `place_io_cpp` uncompiled → "returning CT initial.plc" → seed, 69 overlaps.
- binghamton — export.py emitted a seed-equivalent placement, 69 overlaps (no Dockerfile).
- snoobquants — EmergencyShelfPlacer ran (0.03s) → 1 overlap.
Fix: check INVALID/DISQUALIFIED before VALID; match `(^|[^N])VALID` for the pass case;
do NOT trigger INVALID on the word "overlaps" (valid runs print "overlaps=0").

### Issue 2 — team Dockerfiles never built (false-negative risk; NOT yet re-run)
README: a submission shipping a `Dockerfile` is built and run in that image (network at
build time, `--network none` at run time); else placer.py is mounted in the standard image
(pytorch 2.5.1-cuda12.4, py3.11). run_one.sh always used the standard image. Failing teams
that ship a real, customized Dockerfile were therefore run in the wrong env:
- dragonfly (MISSING_DEP beartype) — Dockerfile installs beartype/jaxtyping/sklearn + stages
  DREAMPlace; returned placement is the DREAMPlace path (ML/macroformer path is dead code).
- icas_placer (FAIL_DP) — ships a from-source DREAMPlace build Dockerfile (cmake/bison).
- ruslanplace (FAIL_DP "IndexError: 2") — Dockerfile git-clones + cmake-builds DREAMPlace at
  /workspace/DREAMPlace (the path placer.py uses); also a fragile `parents[2]` assumption.
- hoop_dreams (FAIL_DP) — DREAMPlace Dockerfile (vendored .so); was previously verified at 1.2207.
- combobulating (CRASH) — placer requires `benchmark.netlist_file`/`plc_file`; the current
  authoritative Benchmark exposes neither → API drift, not a placement bug.

### Confirmed genuine fails (verified by reading clone + log)
- CLONE_FAIL ×5 (cloooooo, makercode, eth_zurich_student, ut_austin_sl, mr_chonk): all HTTP 404 live.
- NO_ENTRY: goats (OpenROAD TCL), mlewand (only changed challenge internals + a byte-identical
  decoy ShelfPack; no own placer), macrobioplacement (CLI script, no place(self,benchmark)).
- CRASH own-bug: 6ummy (hardcoded WSL path /mnt/c/Users/jshin/…), praveen_v (`torch.Device`),
  barsat_khadka (.view() non-contiguous), figo (own mask-shape mismatch), k2hal (own resolve_plc
  →None), trojanmurugan (eval_bridge.py imports non-existent submissions.sidd.main / bad parents[2]).
- MISSING_DEP: no_man_s_sky (ships placer_core.cpp but eval_docker/Dockerfile is stock — no build),
  besson_plr (macro_packer build actually attempted → FAILED, see besson_b.log).
- FAIL_DP: jaideep_padhi (no Dockerfile; own subprocess tool exits 1). TIMEOUT: solomid (70-min re-run).
