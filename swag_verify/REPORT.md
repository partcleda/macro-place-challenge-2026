# Swag-eligibility verification — minimum functionality (ibm01 → valid placement)

**Bar:** submission runs end-to-end on ibm01 and produces a 0-overlap (VALID) placement.

**Method:** air-gapped Docker, each submission limited to 1 GPU / 16 cores / ≤40 GB (mirrors the eval machine per-submission). Challenge `macro_place` is authoritative (scorer integrity); a team's *extra* `macro_place.*` modules are overlaid so their imports resolve; bundled decoy placers (`will_seed`/`examples`) excluded by content hash. Main pass capped at 30 min; fixed-budget placers re-run at a 55–70 min cap (the contest's 1 h/bench rule). DREAMPlace-bundled teams attempted in a DREAMPlace-compiled image.

## Summary

- **ELIGIBLE**: 60
- **FAIL**: 23
- **SKIPPED**: 3

Verdict breakdown (run set): {'PASS': 60, 'CRASH': 7, 'CLONE_FAIL': 5, 'TIMEOUT': 1, 'FAIL_DP': 4, 'NO_ENTRY': 3, 'MISSING_DEP': 3}

## Per-team

| Rank | Team | Status | Detail | Entry | Time |
|---|---|---|---|---|---|
| 17 | MacroHard | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.8641 | `submissions/macrohard/placer.py` | 3215s |
| 19 | jrslbenn | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0392 | `submissions/hybrid_analytical_placer.py` | 1115s |
| 26 | JonaU | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.9862 | `our_placer_final.py` | 3139s |
| 30 | ilovekiro | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `submissions/analytical_placer/placer.py` | 7s |
| 32 | Internship pls | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.8828 | `submissions/analytical_placer/placer.py` | 11s |
| 35 | Hachimi | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.984758 | `placer.py` |  |
| 37 | KKPlace | ✅ ELIGIBLE | PASS: ibm01 valid proxy=907 | `kkplace_v16_b_v20_86.py` | 3018s |
| 38 | Top 3 | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0436 | `submissions/our_team/dreamplace/dp_placer.py` | 20s |
| 39 | The Basin Jumpers | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `submissions/dccp_placer.py` | 6s |
| 40 | Adam_A | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.9765 | `submissions/placer.py` | 37s |
| 43 | V5 | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.9030 | `submissions/will_seed/structureplace.py` | 1308s |
| 45 | moddedmacro | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.8885 | `urstrulyvishtan/placer.py` | 1199s |
| 46 | Vincible | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0357 | `submissions/ensemble/placer.py` | 18s |
| 47 | MJ97 | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1337 | `submissions/mj97/placer.py` | 684s |
| 48 | Electric Beatle | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0097 | `submissions/Energy_placer/gpu_energy_placer.py` | 450s |
| 50 | Jeffrey Chang | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `submissions/hrt_winner/placer.py` | 3282s |
| 51 | UT Dallas | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0500 | `submissions/msears/placer.py` | 164s |
| 55 | itried | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0402 | `final_placer.py` | 3598s |
| 57 | AJAYENDRA KUMAR BANSOD | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.9747 | `submissions/ajiit_placer/placer.py` | 85s |
| 58 | ForageForge | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0374 | `placer.py` | 3436s |
| 59 | UT Austin - AS | ✅ ELIGIBLE | PASS: ibm01 valid proxy=2.0463 | `submissions/dreamplace_placer.py` | 6s |
| 61 | Space Monkeys | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0250 | `submissions/analytical_nesterov.py` | 307s |
| 62 | TAISPlAce | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1953 | `placer.py` | 3421s |
| 63 | PinePlace | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.053650 | `submissions/pineplace/placer.py` | 708s |
| 64 | Pragnay | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.024876 | `submissions/gradient_placer.py` |  |
| 67 | Svyable | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `placer.py` | 4s |
| 68 | DenoisePlace | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1298 | `submissions/final_clean_placer.py` | 120s |
| 69 | KeepDreaming | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1275 | `submissions/dreamplace_placer/placer.py` | 3454s |
| 70 | ECE larpers | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.048676 | `submissions/Final_team_submission/placer.py` | 46s |
| 71 | Aegir | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0436 | `submissions/final/placer.py` | 23s |
| 74 | yoyoshi | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0395 | `submissions/yoyoshi/placer.py` | 8s |
| 76 | Spectral Convergence | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `placer.py` | 907s |
| 77 | LeetFM | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2253 | `submissions/sameer_v1/placer.py` | 3302s |
| 78 | W3 Solutions | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1138 | `submissions/placer.py` | 42s |
| 79 | TheViper | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1077 | `placerRot2.py` | 29s |
| 80 | Team Rocket | ✅ ELIGIBLE | PASS: ibm01 valid proxy=53.6319 | `finale.py` | 187s |
| 81 | 1Brown1Yellow | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2018 | `m5_gwtw_sa_placer.py` | 19s |
| 82 | Captain.Rhinoceros | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0979 | `submissions/jack_hybrid_baseline.py` | 51s |
| 83 | BZC | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1355 | `submissions/final/placer.py` | 96s |
| 85 | Quantux | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2233 | `submissions/quantux/quantuxseedplacer_sqa_replace.py` |  |
| 87 | blindfold | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2253 | `sa_solver.py` | 6s |
| 88 | ZeroLatency | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2253 | `placer/placer.py` | 7s |
| 91 | WeldonWarriors | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2253 | `submissions/v12/placer.py` | 18s |
| 101 | The Sigma Boys | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.1579 | `submissions/gnn_placer_submission.py` | 29s |
| 104 | Bocchi | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2950 | `submissions/abplace_placer.py` | 99s |
| 105 | IITM Placement Cell | ✅ ELIGIBLE | PASS: ibm01 valid proxy=2.2640 | `my_placer/placer.py` | 2371s |
| 106 | sudo optimize --hard | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2583 | `placer.py` | 15s |
| 108 | Team Olmeta | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.3712 | `submissions/modified_annealing/main.py` | 1575s |
| 109 | Binghamton | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.0385 | `python/export.py` | 5s |
| 110 | Can't Place | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2790 | `fd_placer.py` | 24s |
| 111 | UT Austin - CT | ✅ ELIGIBLE | PASS: ibm01 valid proxy=0.9146 | `placer.py` | 438s |
| 112 | rpocevi | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.4908 | `submissions/final_submission.py` | 7s |
| 114 | Adi's Team | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.5720 | `submissions/gnn_placer/placer.py` | 3248s |
| 115 | Himank Galundia | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.6451 | `submissions/sa_placer.py` | 60s |
| 117 | A² | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.2920 | `submissions/will_seed/placer.py` | 8s |
| 118 | Vasu's Snake | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.8361 | `submissions/serpentine_shelf/placer.py` | 6s |
| 119 | Satisficing | ✅ ELIGIBLE | PASS: ibm01 valid proxy=2.0463 | `placer.py` | 4s |
| 120 | Fayaaz | ✅ ELIGIBLE | PASS: ibm01 valid proxy=1.7954 | `submissions/fayaaz_placer.py` | 1s |
| 121 | The Sun Also Places Macros | ✅ ELIGIBLE | PASS: ibm01 valid proxy=2.2944 | `submissions/team_plasma/placer.py` | 200s |
| 122 | SnoobQuants | ✅ ELIGIBLE | PASS: ibm01 valid proxy=7.0718 | `submissions/snoob965/hybrid_gnn_placer.py` | 7s |
| 21 | Combobulating | ❌ FAIL | CRASH: rc=1 RuntimeError: CompetitionPlacer requires loader-provided netlist_file and plc_ | `route_aware_pareto_refine.py` | 2s |
| 23 | cloooooo | ❌ FAIL | CLONE_FAIL: fatal: repository 'https://github.com/sfeirc/macro-place-challenge-2026/' not  | `0s` |  |
| 24 | K2HAL | ❌ FAIL | CRASH: runs self-contained (audit/imports OK) but resolve_plc() returns None for ibm01 | `submissions/macro_placer/cd_lns_placer.py` |  |
| 27 | solomid | ❌ FAIL | TIMEOUT: exceeded 4200s | `placer.py` | 4201s |
| 28 | Figo | ❌ FAIL | CRASH: runs self-contained (Benchmark+plc OK) but placer hits IndexError: mask shape [5993 | `submissions/gpu/placer.py` |  |
| 29 | Hoop Dreams | ❌ FAIL | FAIL_DP: rc=1 ModuleNotFoundError: No module named 'dreamplace.ops.pin_pos.pin_pos_cuda_se | `submissions/dreamtuna/main_placer.py` |  |
| 31 | GOATs | ❌ FAIL | NO_ENTRY: no class with place(self,benchmark) found | `` | 1s |
| 33 | mlewand | ❌ FAIL | NO_ENTRY: no class with place(self,benchmark) found | `` | 4s |
| 34 | MakerCode | ❌ FAIL | CLONE_FAIL: fatal: repository 'https://github.com/Weiyet/macro-place-challenge-2026/' not  | `1s` |  |
| 36 | RuslanPlace | ❌ FAIL | FAIL_DP: rc=1 IndexError: 2 | `placer.py` |  |
| 52 | MacroBioPlacement | ❌ FAIL | NO_ENTRY: no class with place(self,benchmark) found | `` | 1s |
| 53 | Jaideep Padhi | ❌ FAIL | FAIL_DP: rc=1 FileNotFoundError: [Errno 2] No such file or directory: '/challenge/submissi | `submissions/final_placer/placer.py` |  |
| 54 | Barsat Khadka | ❌ FAIL | CRASH: rc=1 RuntimeError: view size is not compatible with input tensor's size and stride  | `soln.py` |  |
| 65 | Praveen V | ❌ FAIL | CRASH: rc=1 AttributeError: module 'torch' has no attribute 'Device' | `submissions/praveen/placer.py` |  |
| 66 | No Man's Sky | ❌ FAIL | MISSING_DEP: _placer_core: ships cpp source (placer_core.cpp) but no build recipe in their | `submissions/straple/placer.py` |  |
| 72 | ETH Zurich Student | ❌ FAIL | CLONE_FAIL: fatal: repository 'https://github.com/BasilGrande/MacroPlaceChallenge2026/' no | `0s` |  |
| 73 | another Waterloo kid | ⏭️ SKIPPED | Modal cloud dispatch — cannot run air-gapped | `` |  |
| 75 | Dragonfly | ❌ FAIL | MISSING_DEP: No module named 'beartype' | `submission_eval/macroformer/placer.py` | 3s |
| 90 | SEVmakers | ⏭️ SKIPPED | private repo — no judge access | `` |  |
| 92 | TROJANMurugan | ❌ FAIL | CRASH: rc=1 IndexError: 2 | `eval_bridge.py` | 3s |
| 96 | UT Austin - SL | ❌ FAIL | CLONE_FAIL: fatal: repository 'https://github.com/samlin-2025/Partcl-Macro-Placement-Chall | `1s` |  |
| 97 | 6ummy | ❌ FAIL | CRASH: rc=1 FileNotFoundError: [Errno 2] No such file or directory: 'wsl' | `submissions/dreamplace_placer.py` | 3s |
| 100 | ICAS_placer | ❌ FAIL | FAIL_DP: rc=1 ImportError: cannot import name 'Params' from 'dreamplace' (unknown location | `placer.py` |  |
| 102 | UT Austin - RH | ⏭️ SKIPPED | no resolvable repo URL (profile link only / private) | `` |  |
| 103 | Besson-PLR | ❌ FAIL | MISSING_DEP: macro_packer C++ ext does not build from their setup.py in eval env | `submissions/GeometricLegalizer/placer.py` |  |
| 107 | Mr_Chonk | ❌ FAIL | CLONE_FAIL: fatal: repository 'https://github.com/MasterChonk/Macro-placement-algo/' not f | `0s` |  |

## Caveats (eligible-but-borderline)

- **Seed-legalizer cluster** (proxy 1.0385 / 1.2253 across several teams): the provided ibm01 seed scores 1.0385 but is INVALID (69 overlaps); these teams legalize it to 0 overlaps with minimal displacement (valid + functional, but low-novelty — they pass).
- **Binghamton**: repo ships pre-computed placements + an `export.py` (team's note: source not provided). Passes by emitting stored placements, not a from-scratch placer.
- **Degenerate-but-valid**: KKPlace (proxy≈907), Team Rocket (≈53.6), SnoobQuants (≈7.1) produce 0-overlap placements of very poor quality — they clear the functionality bar only.
- A few entry-points were auto-detected at low confidence; spot-check if a team disputes.
