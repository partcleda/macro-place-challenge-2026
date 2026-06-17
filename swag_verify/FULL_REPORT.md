# Macro-Placement Challenge — Full Swag-Eligibility Report

_Generated 2026-06-17. Leaderboard standings as of ref `add-shoom-rescore` (Archgen re-instated at #1)._

**Swag bar:** the submission demonstrates minimum functionality — it runs end-to-end and produces a 0-overlap (VALID) placement, judged by the TILOS MacroPlacement scorer. Teams already confirmed by a full judge run qualify automatically.

## Summary

- **Total leaderboard teams:** 125
- **Swag-eligible: 93**  =  36 already-verified (auto-qualify)  +  57 newly swag-verified
- **Ineligible: 29**  =  26 failed the swag check  +  3 disqualified
- **Skipped (could not be run): 3**  (Modal-only / private / no resolvable repo)

### How eligibility was determined

1. **Already-verified teams** cleared a full multi-benchmark judge run and qualify automatically — not re-run by the swag harness.
2. **Previously-unverified teams** were each run **air-gapped** on `ibm01`, clamped to the eval envelope (1 GPU / 16 cores / ≤40 GB), and scored by the authoritative scorer. Detail: `swag_verify/REPORT.md` and `swag_verify/TRIAGE_NOTES.md`.
3. **Disqualified** entries (failed judge run / superseded / won't run) are excluded.

### Caveats

- **Conservative by construction.** A team is ✅ only if it *demonstrated* a valid placement. Teams whose declared environment (a shipped `Dockerfile`) was not built, or that hit a challenge-API change, are currently ❌ but flagged **⟳ PENDING RE-RUN** — a faithful re-run may move some back to ✅: Combobulating, Dragonfly, Hoop Dreams, ICAS_placer, RuslanPlace, ilovekiro.
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
| 22 | Combobulating | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 RuntimeError: CompetitionPlacer requires loader-provided netlist_file and plc_  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 23 | Lawnmower | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.0877) |
| 24 | cloooooo | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/sfeirc/macro-place-challenge-2026/' not |
| 25 | K2HAL | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: runs self-contained (audit/imports OK) but resolve_plc() returns None for ibm01 |
| 26 | IDK | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.1268) |
| 27 | JonaU | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9862 |
| 28 | solomid | ❌ INELIGIBLE | swag-checked (ibm01): TIMEOUT: exceeded 4200s |
| 29 | Figo | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: runs self-contained (Benchmark+plc OK) but placer hits IndexError: mask shape [5993 |
| 30 | Hoop Dreams | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 ModuleNotFoundError: No module named 'dreamplace.ops.pin_pos.pin_pos_cuda_se  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 31 | ilovekiro | ❌ INELIGIBLE | swag-checked (ibm01): INVALID: ibm01 INVALID (69 overlaps) — DREAMPlace ext uncompiled; placer returned seed ini  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 32 | GOATs | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 33 | Internship pls | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.8828 |
| 34 | mlewand | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 35 | MakerCode | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/Weiyet/macro-place-challenge-2026/' not |
| 36 | Hachimi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.984758 |
| 37 | RuslanPlace | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 IndexError: 2  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 38 | KKPlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=907 |
| 39 | Top 3 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0436 |
| 40 | The Basin Jumpers | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 41 | Adam_A | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9765 |
| 42 | MTK | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2744) |
| 43 | RoRa | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2788) |
| 44 | V5 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9030 |
| 45 | KLA MACH | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.2946) |
| 46 | moddedmacro | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.8885 |
| 47 | Vincible | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0357 |
| 48 | MJ97 | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1337 |
| 49 | Electric Beatle | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0097 |
| 50 | UToronto Analytical | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.3323) |
| 51 | Jeffrey Chang | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 52 | UT Dallas | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0500 |
| 53 | MacroBioPlacement | ❌ INELIGIBLE | swag-checked (ibm01): NO_ENTRY: no class with place(self,benchmark) found |
| 54 | Jaideep Padhi | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 FileNotFoundError: [Errno 2] No such file or directory: '/challenge/submissi |
| 55 | Barsat Khadka | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 RuntimeError: view size is not compatible with input tensor's size and stride |
| 56 | itried | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0402 |
| 57 | Varun's Parallel Worlds | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4017) |
| 58 | AJAYENDRA KUMAR BANSOD | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9747 |
| 59 | ForageForge | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0374 |
| 60 | UT Austin - AS | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.0463 |
| 61 | ByteDancer | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4151) |
| 62 | Space Monkeys | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0250 |
| 63 | TAISPlAce | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1953 |
| 64 | PinePlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.053650 |
| 65 | Pragnay | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.024876 |
| 66 | Praveen V | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 AttributeError: module 'torch' has no attribute 'Device' |
| 67 | No Man's Sky | ❌ INELIGIBLE | swag-checked (ibm01): MISSING_DEP: _placer_core: ships cpp source (placer_core.cpp) but no build recipe in their |
| 68 | Svyable | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 69 | DenoisePlace | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1298 |
| 70 | KeepDreaming | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1275 |
| 71 | ECE larpers | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.048676 |
| 72 | Aegir | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0436 |
| 73 | ETH Zurich Student | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/BasilGrande/MacroPlaceChallenge2026/' no |
| 74 | another Waterloo kid | ⏭️ SKIPPED | Modal cloud dispatch — cannot run air-gapped |
| 75 | yoyoshi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0395 |
| 76 | Dragonfly | ❌ INELIGIBLE | swag-checked (ibm01): MISSING_DEP: No module named 'beartype'  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 77 | Spectral Convergence | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0385 |
| 78 | LeetFM | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 79 | W3 Solutions | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1138 |
| 80 | TheViper | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1077 |
| 81 | Team Rocket | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=53.6319 |
| 82 | 1Brown1Yellow | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2018 |
| 83 | Captain.Rhinoceros | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.0979 |
| 84 | BZC | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1355 |
| 85 | Jiangban Ya | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.4943) |
| 86 | Quantux | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2233 |
| 87 | UTAUSTIN-CT | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5062) |
| 88 | blindfold | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 89 | ZeroLatency | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 90 | oracleX | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5130) |
| 91 | SEVmakers | ⏭️ SKIPPED | private repo — no judge access |
| 92 | WeldonWarriors | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2253 |
| 93 | TROJANMurugan | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 IndexError: 2 |
| 94 | CA | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5247) |
| 95 | #5 ubc cpen student | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5337) |
| 97 | UT Austin - SL | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/samlin-2025/Partcl-Macro-Placement-Chall |
| 98 | 6ummy | ❌ INELIGIBLE | swag-checked (ibm01): CRASH: rc=1 FileNotFoundError: [Errno 2] No such file or directory: 'wsl' |
| 99 | RUDY Can't Fail | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5397) |
| 100 | dbzero | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.5680) |
| 101 | ICAS_placer | ❌ INELIGIBLE | swag-checked (ibm01): FAIL_DP: rc=1 ImportError: cannot import name 'Params' from 'dreamplace' (unknown location  ⟳ PENDING RE-RUN (ships own Dockerfile / API-drift; may recover) |
| 102 | The Sigma Boys | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.1579 |
| 103 | UT Austin - RH | ⏭️ SKIPPED | no resolvable repo URL (profile link only / private) |
| 104 | Besson-PLR | ❌ INELIGIBLE | swag-checked (ibm01): MISSING_DEP: macro_packer C++ ext does not build from their setup.py in eval env |
| 105 | Bocchi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2950 |
| 106 | IITM Placement Cell | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.2640 |
| 107 | sudo optimize --hard | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2583 |
| 108 | Mr_Chonk | ❌ INELIGIBLE | swag-checked (ibm01): CLONE_FAIL: fatal: repository 'https://github.com/MasterChonk/Macro-placement-algo/' not f |
| 109 | Team Olmeta | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.3712 |
| 110 | Binghamton | ❌ INELIGIBLE | swag-checked (ibm01): INVALID: ibm01 INVALID (69 overlaps) — export.py emitted a seed-equivalent placement |
| 111 | Can't Place | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2790 |
| 112 | UT Austin - CT | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=0.9146 |
| 113 | rpocevi | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.4908 |
| 114 | AS | ✅ ELIGIBLE | verified — full judge run (avg proxy 1.9121) |
| 115 | Adi's Team | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.5720 |
| 116 | Himank Galundia | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.6451 |
| 117 | Sharc #1 | ✅ ELIGIBLE | verified — full judge run (avg proxy 2.0433) |
| 118 | A² | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.2920 |
| 119 | Vasu's Snake | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.8361 |
| 120 | Satisficing | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.0463 |
| 121 | Fayaaz | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=1.7954 |
| 122 | The Sun Also Places Macros | ✅ ELIGIBLE | swag-checked (ibm01): PASS: ibm01 valid proxy=2.2944 |
| 123 | SnoobQuants | ❌ INELIGIBLE | swag-checked (ibm01): INVALID: ibm01 INVALID (1 overlaps) — EmergencyShelfPlacer |
| DQ | Place, Route, Roll | ❌ INELIGIBLE | disqualified — Judge run failed: DREAMPlace produced no usable placements for ibm01 (65 failed  |
| DQ | Mike Gao | ❌ INELIGIBLE | disqualified — 1939 overlaps (old submission). Resubmitted 5/21 as "Place, Route, Roll". |
| DQ | BakaBobo | ❌ INELIGIBLE | disqualified — Missing import — code won't run. |
