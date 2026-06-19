# Macro-Placement Challenge — Full Swag-Eligibility Report

_Generated 2026-06-19. Leaderboard standings as of ref `origin/main` (Archgen re-instated at #1)._

**Swag bar:** the submission demonstrates minimum functionality — it runs end-to-end and produces a 0-overlap (VALID) placement, judged by the TILOS MacroPlacement scorer. Teams already confirmed by a full judge run qualify automatically.

## Summary

- **Total leaderboard teams:** 132
- **Swag-eligible: 103**  =  37 verified (auto-qualify)  +  57 swag-verified (cohort)  +  9 recovered / newly-verified this pass
- **Ineligible: 25**  =  22 failed the swag check  +  3 disqualified
- **Skipped (could not be run): 4**  (Modal-only / private / unreachable repo)

### ✅ Changed to VALID this pass (recovered — were previously failed / DQ / unlisted)

- **K2HAL** — RECOVERED — harness artifact (uninitialized benchmark submodule made resolve_plc() return None); re-run self-contained w/ submodule populated -> ibm01 VALID (0.8124). Not a code bug.
- **Hoop Dreams** — RECOVERED — built team py3.12 Dockerfile (vendored .so); ibm01 VALID (0.9078)
- **RuslanPlace** — RECOVERED — harness artifact (DREAMPlace had built CPU-only); rebuilt w/ CUDA + configure fix -> ibm01 VALID (0.9350).
- **Macropolis** — NEW form-only entry; ibm01 VALID (1.0276, self-contained)
- **Nikunj Bhatt** — NEW form-only entry; ibm01 VALID (1.0385, legalizes seed)
- **Dragonfly** — RECOVERED — built team Dockerfile; ibm01 VALID (2757s)
- **AxeCap** — NEW form-only entry; ibm01 VALID (1.2391)
- **A-cat-suki** — NEW form-only entry; ibm01 VALID (1.3446)
- **BakaBobo** — RECOVERED — false DQ; ships macro_place.fast_proxy (overlay resolves it); ibm01 VALID (1.0109)

### How eligibility was determined

1. **Already-verified teams** cleared a full multi-benchmark judge run and qualify automatically — not re-run by the swag harness.
2. **Previously-unverified teams** were each run **air-gapped** on `ibm01`, clamped to the eval envelope (1 GPU / 16 cores / ≤40 GB), and scored by the authoritative scorer. Detail: `swag_verify/REPORT.md` and `swag_verify/TRIAGE_NOTES.md`.
3. **Disqualified** entries (failed judge run / superseded / won't run) are excluded.

### Caveats

- **Recovered / newly-verified this pass.** Re-running teams in their declared environment recovered **Dragonfly** & **BakaBobo** (built their Dockerfile / overlaid the module they were wrongly failed for), plus 4 form-only submissions that were never previously ranked (**Macropolis, Nikunj Bhatt, AxeCap, A-cat-suki**) — all ibm01 VALID.
- **Still pending.** Teams that ship a Dockerfile or hit challenge-API drift but aren't yet cleanly reproduced are flagged **⟳ PENDING RE-RUN** — a faithful re-run may move some to ✅: ICAS_placer, Place, Route, Roll. **macrobossesiitp** is unverifiable (repo unreachable).
- **A PASS-gate bug was fixed.** The gate matched `VALID` inside `INVALID`; three submissions the scorer marked INVALID (ilovekiro, Binghamton, SnoobQuants) had been mis-scored eligible and are now correctly ❌.
- **`ibm01` only.** Single-benchmark minimum-functionality check, not the 17-benchmark ranking.

## All teams

| Rank | Team | Status | Basis |
|---|---|---|---|
| 1 | Archgen | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9507) |
| 2 | Shoom | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9842) |
| 3 | Klein-4 | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9846) |
| 4 | tobias-x | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9884) |
| 5 | Vibe | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9939) |
| 6 | Macro Polo | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9965) |
| 7 | thinkorplace | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9974) |
| 8 | JaneRT | ✅ ELIGIBLE | verified — full judge run (avg proxy 0.9978) |
| 9 | Carrotato | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0103) |
| 10 | vmallela | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0109) |
| 11 | DREAMPlaceProMaxUltra | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0121) |
| 12 | QED | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0266) |
| 13 | Two-IIITK-Kids | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0285) |
| 14 | ArzunPD | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0507) |
| 15 | QuantSC | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0513) |
| 16 | Cezar | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0663) |
| 17 | WAVAG | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0689) |
| 18 | Talyxion | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0698) |
| 19 | MacroHard | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.8641 |
| 20 | Kagan Dikmen | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0705) |
| 21 | jrslbenn | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0392 |
| 22 | Shoom | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0808) |
| 23 | Combobulating | ❌ INELIGIBLE | API drift — requires Benchmark.netlist_file/plc_file (not in current API); ships no own macro_place. Confirmed self-contained. |
| 24 | Lawnmower | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0877) |
| 25 | cloooooo | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/sfeirc/macro-place-challenge-2026/' not |
| 26 | K2HAL | ✅ ELIGIBLE | RECOVERED — harness artifact (uninitialized benchmark submodule made resolve_plc() return None); re-run self-contained w/ submodule populated -> ibm01 VALID (0.8124). Not a code bug. |
| 27 | IDK | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.1268) |
| 28 | JonaU | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9862 |
| 29 | solomid | ❌ INELIGIBLE | swag-checked (ibm01): TIMEOUT: exceeded 4200s |
| 30 | Figo | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: runs self-contained (Benchmark+plc OK) but placer hits IndexError: mask shape [5993 |
| 31 | Hoop Dreams | ✅ ELIGIBLE | RECOVERED — built team py3.12 Dockerfile (vendored .so); ibm01 VALID (0.9078) |
| 32 | ilovekiro | ❌ INELIGIBLE | INVALID — DREAMPlace fails even in its own py3.10 env; returns the seed (69 overlaps). Confirmed via re-run. |
| 33 | GOATs | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 34 | Internship pls | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.8828 |
| 35 | mlewand | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 36 | MakerCode | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/Weiyet/macro-place-challenge-2026/' not |
| 37 | Hachimi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.984758 |
| 38 | RuslanPlace | ✅ ELIGIBLE | RECOVERED — harness artifact (DREAMPlace had built CPU-only); rebuilt w/ CUDA + configure fix -> ibm01 VALID (0.9350). |
| 39 | KKPlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=907 |
| 40 | Top 3 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0436 |
| 41 | The Basin Jumpers | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 42 | Adam_A | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9765 |
| 43 | MTK | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2744) |
| 44 | RoRa | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2788) |
| 45 | V5 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9030 |
| 46 | Macropolis | ✅ ELIGIBLE | NEW form-only entry; ibm01 VALID (1.0276, self-contained) |
| 47 | KLA MACH | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2946) |
| 48 | moddedmacro | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.8885 |
| 49 | Vincible | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0357 |
| 50 | MJ97 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1337 |
| 51 | Electric Beatle | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0097 |
| 52 | UToronto Analytical | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.3323) |
| 53 | Jeffrey Chang | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 54 | UT Dallas | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0500 |
| 55 | MacroBioPlacement | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 56 | Jaideep Padhi | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 FileNotFoundError: [Errno 2] No such file or directory: '/challenge/submissi |
| 57 | Barsat Khadka | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 RuntimeError: view size is not compatible with input tensor's size and stride |
| 58 | itried | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0402 |
| 59 | Varun's Parallel Worlds | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4017) |
| 60 | AJAYENDRA KUMAR BANSOD | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9747 |
| 61 | ForageForge | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0374 |
| 62 | UT Austin - AS | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.0463 |
| 63 | ByteDancer | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4151) |
| 64 | Space Monkeys | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0250 |
| 65 | TAISPlAce | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1953 |
| 66 | PinePlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.053650 |
| 67 | Pragnay | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.024876 |
| 68 | Praveen V | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 AttributeError: module 'torch' has no attribute 'Device' |
| 69 | No Man's Sky | ❌ INELIGIBLE | swag-checked (ibm01): MISSING_DEP: _placer_core: ships cpp source (placer_core.cpp) but no build recipe in their |
| 70 | Svyable | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 71 | DenoisePlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1298 |
| 72 | KeepDreaming | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1275 |
| 73 | ECE larpers | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.048676 |
| 74 | Aegir | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0436 |
| 75 | ETH Zurich Student | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/BasilGrande/MacroPlaceChallenge2026/' no |
| 76 | another Waterloo kid | ⏭️ SKIPPED | Modal cloud dispatch — cannot run air-gapped |
| 77 | Nikunj Bhatt | ✅ ELIGIBLE | NEW form-only entry; ibm01 VALID (1.0385, legalizes seed) |
| 78 | yoyoshi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0395 |
| 79 | Dragonfly | ✅ ELIGIBLE | RECOVERED — built team Dockerfile; ibm01 VALID (2757s) |
| 80 | Spectral Convergence | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 81 | macrobossesiitp | ⏭️ SKIPPED | NEW form-only entry; repo unreachable even with judge token — cannot verify |
| 82 | LeetFM | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 83 | W3 Solutions | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1138 |
| 84 | TheViper | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1077 |
| 85 | Team Rocket | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=53.6319 |
| 86 | 1Brown1Yellow | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2018 |
| 87 | Captain.Rhinoceros | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0979 |
| 88 | BZC | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1355 |
| 89 | Jiangban Ya | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4943) |
| 90 | Quantux | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2233 |
| 91 | UTAUSTIN-CT | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5062) |
| 92 | blindfold | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 93 | ZeroLatency | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 94 | oracleX | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5130) |
| 95 | SEVmakers | ⏭️ SKIPPED | private repo — no judge access |
| 96 | WeldonWarriors | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 97 | TROJANMurugan | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 IndexError: 2 |
| 98 | CA | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5247) |
| 99 | #5 ubc cpen student | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5337) |
| 101 | UT Austin - SL | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/samlin-2025/Partcl-Macro-Placement-Chall |
| 102 | 6ummy | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 FileNotFoundError: [Errno 2] No such file or directory: 'wsl' |
| 103 | AxeCap | ✅ ELIGIBLE | NEW form-only entry; ibm01 VALID (1.2391) |
| 104 | RUDY Can't Fail | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5397) |
| 105 | dbzero | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5680) |
| 106 | ICAS_placer | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 ImportError: cannot import name 'Params' from 'dreamplace' (unknown location  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 107 | The Sigma Boys | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1579 |
| 108 | UT Austin - RH | ⏭️ SKIPPED | no resolvable repo URL (profile link only / private) |
| 109 | Besson-PLR | ❌ INELIGIBLE | swag-checked (ibm01): MISSING_DEP: macro_packer C++ ext does not build from their setup.py in eval env |
| 110 | Bocchi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2950 |
| 111 | IITM Placement Cell | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.2640 |
| 112 | sudo optimize --hard | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2583 |
| 113 | Mr_Chonk | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/MasterChonk/Macro-placement-algo/' not f |
| 114 | Team Olmeta | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.3712 |
| 115 | A-cat-suki | ✅ ELIGIBLE | NEW form-only entry; ibm01 VALID (1.3446) |
| 116 | Binghamton | ❌ INELIGIBLE | swag-checked (ibm01): INVALID: ibm01 INVALID (69 overlaps) — export.py emitted a seed-equivalent placement |
| 117 | Can't Place | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2790 |
| 118 | UT Austin - CT | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9146 |
| 119 | rpocevi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.4908 |
| 120 | AS | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.9121) |
| 121 | Adi's Team | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.5720 |
| 122 | Himank Galundia | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.6451 |
| 123 | Sharc #1 | ✅ ELIGIBLE | verified — full judge run (avg proxy 2.0433) |
| 124 | A² | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2920 |
| 125 | Vasu's Snake | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.8361 |
| 126 | Satisficing | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.0463 |
| 127 | Fayaaz | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.7954 |
| 128 | The Sun Also Places Macros | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.2944 |
| 129 | SnoobQuants | ❌ INELIGIBLE | swag-checked (ibm01): INVALID: ibm01 INVALID (1 overlaps) — EmergencyShelfPlacer |
| DQ | Place, Route, Roll | ❌ INELIGIBLE | disqualified — Judge run failed: DREAMPlace produced no usable placements for ibm01 (65 failed   ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| DQ | Mike Gao | ❌ INELIGIBLE | disqualified — 1939 overlaps (old submission). Resubmitted 5/21 as "Place, Route, Roll". |
| DQ | Archgen | ❌ INELIGIBLE | disqualified — 1 overlap on ibm17 (judge run). Judge AVG=1.0203. Resubmitted 5/21 (xplace+CD).  |
| DQ | BakaBobo | ✅ ELIGIBLE | RECOVERED — false DQ; ships macro_place.fast_proxy (overlay resolves it); ibm01 VALID (1.0109) |
