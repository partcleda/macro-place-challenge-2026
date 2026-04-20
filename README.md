# Partcl/HRT Macro Placement Challenge

<img src="assets/HRT.png" alt="Hudson River Trading" height="80"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="assets/partcl.png" alt="Partcl" height="80">

**Win $20,000 by developing better macro placement algorithms!**

Partcl and Hudson River Trading are excited to co-host a competition to solve the macro placement problem. 

## About Macro Placement

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.

For example, the **ibm01** benchmark has:
- **246 hard macros** of varying sizes (ranging from 0.8 to 27 μm², with 33× size variation)
- **7,269 nets** connecting macros to each other and to 894 pre-placed standard cell clusters
- **A 22.9 × 23.0 μm canvas** with 42.8% area utilization

<p align="center">
  <img src="assets/sa_ibm01.gif" alt="Simulated annealing on ibm01" width="600"><br>
  <img src="assets/fd_ibm01.gif" alt="Force-directed placement on ibm01" width="600">
</p>

## About HRT Hardware

Hudson River Trading (HRT) is a leading quantitative trading firm at the forefront of technical innovation in global financial markets.

HRT’s Hardware team builds the high-performance compute systems at the core of our trading infrastructure. We use FPGAs and ASICs to drive low-latency decision-making and power custom solutions across the trading stack, from bespoke circuits to machine learning accelerators.

We’re proud to sponsor this competition because advancing macro placement and low-level hardware optimization directly aligns with the kinds of hard, performance-critical engineering challenges our teams tackle every day.

Joining Hudson River Trading’s hardware team means working alongside leading engineers in one of the most advanced computing environments in global finance. Learn more about open roles at [hudsonrivertrading.com](https://www.hudsonrivertrading.com/).

## About Partcl

Partcl is rebuilding chip design infrastructure from the ground up for the GPU era.

Modern chip design is slow, fragmented, and fundamentally constrained by tools built decades ago. Critical workflows like timing analysis and placement still take hours to days - limiting how much engineers can explore and optimize.

We’re changing that.

Partcl develops GPU-accelerated systems for physical design that run orders of magnitude faster than legacy tools. Our goal is simple: make iteration cheap enough that design space exploration becomes the default, not the exception.

## Background Papers
[1] [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)

[2] [Assessment of Reinforcement Learning for Macro Placement](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf)

[3] [Reevaluating Google's Reinforcement Learning for IC Macro Placement](https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/)

[4] [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)

## 🏆 Prizes

- **$20,000 — Grand Prize:** The top 7 submissions by proxy score are evaluated through the OpenROAD flow on NG45 designs (including hidden designs). Among those 7, the submission that beats the SA and RePlAce baselines (reported in [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)) by the largest margin on WNS, TNS, and Area wins the Grand Prize. 
- **$20,000 — First Place (Proxy):** Awarded to the #1 submission by proxy score. Only awarded if no submission qualifies for the Grand Prize.
- **$5,000 — Second Place:** Awarded to the runner-up of the Grand Prize. If no submission qualifies for the Grand Prize, awarded to the #2 submission by proxy score.
- **$4,000 — Innovation Award:** Granted to the most creative or technically innovative approach among the top entries, as determined by the judging panel.
- **Swag:** Every valid submission gets HRT swag!
- **Note:** An additional score adjustment will be applied based on human-expert analysis of the resulting placement.

## Submission Format

- All submissions will be via google form. Submissions may be made public or private before the end of judging.
- Private submissions will be required to share repository with judges so they may clone/evaluate the method.
- Teams may be up to 5 individuals.
- The deadline for submissions is May 21, 2026, 11:59 pacific.
- All teams may only submit one algorithm.
- **All winning implementations must be made open-source under Apache 2.0 or GPL**
- All submissions must be registered via this [Submission Link](https://forms.gle/YDRtYV5Vq68SZgKW9).
- All submissions must be under 1 hour end-to-end runtime (per benchmark) for the macro placement algorithm.
- All submissions will be evaluated on a AMD EPYC 9655P with 16 cores + 100GB of memory and an NVIDIA RTX 6000 Ada 48GB.

## Additional Rules

### Allowed

- **Any algorithmic approach**: SA, RL, GNN, analytical methods, hybrid approaches, learning-based, etc.
- **Any framework**: PyTorch, TensorFlow, JAX, or pure Python/C++
- **Any optimization technique**: Gradient descent, evolutionary algorithms, local search, etc.
- **Training on public benchmarks**: You can learn from the IBM benchmark data

### Not Allowed

- Modifying the evaluation functions (must use TILOS MacroPlacement evaluator as-is)
- Hardcoding solutions for specific benchmarks (must be general algorithm)
- Using external/proprietary placement tools (must be open-source submission)
- Exceeding runtime limits (1 hour per benchmark hard timeout)
- Overlaps in resulting placement (strictly zero overlap between hard macros — no tolerance. Participants should add small gaps in their legalization to avoid float-precision edge cases.)

## Evaluation Details

Evaluation is two-tiered:

### Tier 1: Proxy Cost Ranking (All Submissions)

All submissions are ranked by **proxy cost** across the 18 IBM benchmarks. This is the primary qualifying metric. Proxy cost is computed using the TILOS MacroPlacement evaluator:

> **Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion**

Baseline numbers are from: [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)

### Tier 2: OpenROAD Flow Validation (Top Submissions)

The top 7 submissions by proxy score will be evaluated through the full **OpenROAD flow** on NG45 designs to measure real PnR outcomes: **WNS, TNS, and Area**.

- The **Grand Prize ($20K)** is awarded based on best OpenROAD results among these top submissions.
- To qualify, you must surpass the SA and RePlAce baselines for WNS, TNS, and Area.
- To avoid overfitting, we will also evaluate on 1-2 hidden NG45 designs.

## 🚀 Quick Start

### Installation 

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Install the package and all dependencies
uv sync

# Verify the setup
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
```

### Run Your First Example

```bash
# Run the greedy row placer on ibm01
uv run evaluate submissions/examples/greedy_row_placer.py

# Run on all 17 IBM benchmarks
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Run on NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize the result
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

Running on all benchmarks produces a summary like:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    2.0463    1.3166    0.9976    -55.4%     -105.1%         0
   ibm02    2.0431    1.9072    1.8370     -7.1%      -11.2%         0
   ...
     AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

The greedy placer achieves zero overlaps but makes no attempt to optimize wirelength or connectivity — your job is to do better! See [`SETUP.md`](SETUP.md) for the full API reference and [`submissions/examples/`](submissions/examples/) for working examples.

## 🎯 IBM Benchmark Suite (ICCAD04)

We evaluate on the complete ICCAD04 IBM benchmark suite:

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (you can also place these)
- Nets connecting all components
- Initial placement (hand-crafted, serves as reference)

**Baseline Analysis:**
- RePlAce (⭐) consistently outperforms SA across all benchmarks
- RePlAce achieves 15-55% lower proxy cost than SA
- **To qualify for the Grand Prize, your placement must also produce better WNS, TNS, and Area than both baselines when evaluated through the OpenROAD flow on NG45 designs**
- Both baselines achieve zero overlaps (enforced as hard constraint)

## 💡 Why This Is Hard

Despite "only" 246-537 macros, this problem is extremely challenging:

1. **Massive search space**: ~10^800 possible placements (even with constraints)
2. **Conflicting objectives**: Wirelength wants clustering, density wants spreading, congestion wants routing space
3. **Non-convex landscape**: Millions of local minima, discontinuities, plateaus
4. **Long-range dependencies**: Moving one macro affects costs globally through thousands of nets
5. **Hard constraints**: No overlaps between heterogeneous sizes (33× size variation)
6. **Tight packing**: 43-53% area utilization leaves little slack
7. **Runtime matters**: Must be fast enough to be practical (< 5 minutes ideal)

Classical methods (SA, RePlAce) have been refined for decades but still have room for improvement!

## 📖 Documentation

- **Setup & API Reference**: [`SETUP.md`](SETUP.md) - Infrastructure details, benchmark format, cost computation, testing
- **Example Submissions**: [`submissions/examples/`](submissions/examples/) - Working placer examples

## 📚 References

- **TILOS MacroPlacement**: [GitHub Repository](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Source of evaluation infrastructure
  - ICCAD04 benchmarks
  - SA and RePlAce baseline implementations

- **ICCAD04 Benchmarks**: Classic macro placement benchmark suite used in academic research

## 🏅 Leaderboard

Submissions are ranked by **average proxy cost** across all 17 IBM benchmarks (lower is better). Zero overlaps required on all benchmarks. Scores are unverified until confirmed by judges.

| Rank | Team | Avg Proxy Cost | Best | Worst | Overlaps | Runtime | Verified | Notes |
|------|------|---------------|------|-------|----------|---------|----------|-------|
| 1 | "MTK" (DreamPlace++) | **1.2818** | 0.9073 | 1.6529 | 0 | 37s/bench (GPU) | :white_check_mark: | Verified better than self-reported 1.317; beats RePlAce on all 17 benchmarks |
| 2 | "RoRa" (RipPlace) | **1.3241** | — | — | 0 | 694s/bench | | |
| 3 | "Mike Gao" (autoresearch) | **1.3255** | — | — | 0 | 16min/bench | | |
| 4 | "Electric Beatel" (ePlace-Lite) | **1.3913** | 0.9773 | 1.7253 | 0 | 155s/bench (GPU) | :white_check_mark: | |
| 5 | "Varun's Parallel Worlds" (GRPlace) | **1.4017** | 1.0362 | 1.7298 | 0 | 27s/bench | :white_check_mark: | |
| 6 | "BakaBobo" (Global Relocation Sweep) | **1.4044** | — | — | 0 | 282s/bench | | Updated from 1.4403 |
| 7 | "UT Austin" - AS (DREAMPlace Analytical) | **1.4076** | — | — | 0 | 17s/bench | | |
| 8 | "ByteDancer" (Incremental CD) | **1.4151** | 1.0236 | 1.7792 | 0 | 38min/bench | :white_check_mark: | |
| 9 | "Convex Optimization" (UWaterloo Student) | **1.4556** | 1.0432 | 1.7867 | 0 | 11s/bench | :white_check_mark: | Resubmitted 4/13; fixed from DQ (was 846 overlaps) |
| 10 | "another Waterloo kid" (Batched Nesterov GP) | **1.4568** | — | — | 0 | 118s/bench | | |
| — | RePlAce (baseline) | **1.4578** | 0.9976 | 1.8370 | 0 | — | :white_check_mark: | |
| 11 | "Jiangban Ya" (Min-Displacement Legalizer) | **1.4944** | — | — | 0 | 188s/bench | | |
| 12 | "UTAUSTIN-CT" (PLC-Exact Congestion-Aware SA) | **1.5062** | 1.1363 | 1.7941 | 0 | 6s/bench | :white_check_mark: | |
| 13 | "oracleX" (Oracle) | **1.5130** | 1.1340 | 1.7937 | 0 | 11s/bench | :white_check_mark: | |
| 14 | "SEVmakers" (Hybrid Legalization + SA) | **1.5200** | — | — | 0 | 200s/bench | | |
| 15 | "CA" (congestion_aware) | **1.5247** | 1.2226 | 1.7945 | 0 | 2s/bench | :white_check_mark: | Verified 1.5247 vs self-reported 1.5238 |
| 16 | "#5 ubc cpen student" (Gene Pool Shuffle) | **1.5337** | 1.1411 | 1.8084 | 0 | 13s/bench | :white_check_mark: | |
| 17 | Will Seed (Partcl) | **1.5338** | 1.1625 | 1.7965 | 0 | 35s total | :white_check_mark: | |
| 18 | "Cezar" (CRISP) | **1.5781** | 1.1896 | 1.8520 | 0 | 4min/bench | :white_check_mark: | |
| 19 | "UT Austin" - RH (DREAMPlace) | **1.6037** | — | — | 0 | 4.5s/bench | | |
| 20 | "UT Austin" - CT (PROXYCost) | **1.8706** | — | — | 0 | 187s/bench | | |
| 21 | "Sharc #1" (Auction Placer) | **2.0433** | — | — | 0 | 223s/bench | | |
| — | SA (baseline) | 2.1251 | 1.3166 | 3.6726 | 0 | — | :white_check_mark: | |
| — | Greedy Row (demo) | 2.2109 | 1.6728 | 2.7696 | 0 | 0.3s total | :white_check_mark: | |
| — | "Binghamton" (feng shui) | pending | — | — | — | — | | |
| — | "MacroBio" (Two-Opt Swap) | pending | — | — | — | — | | |

*Submit your results via the [Submission Link](https://forms.gle/YDRtYV5Vq68SZgKW9)!*

## 🤔 FAQ

**Q: What benchmarks are used?**
A: Tier 1 (proxy cost) uses 17 IBM ICCAD04 benchmarks — the standard academic suite with well-established baselines. Tier 2 (OpenROAD flow) uses NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla) plus 1-2 hidden designs. You can evaluate on both with `--all` (IBM) and `--ng45` (NG45).

**Q: What if I beat one baseline but not the other?**
A: You must beat BOTH SA and RePlAce baselines on WNS, TNS, and Area to qualify for the Grand Prize. You can still win the Proxy or Innovation prizes regardless.

**Q: Are there hidden test cases?**
A: All 17 IBM benchmarks for proxy cost ranking are public. The 4 NG45 designs are also public. For the OpenROAD flow evaluation (Tier 2), we will additionally test on 1-2 hidden NG45 designs to ensure generalization.

**Q: What counts as "beating" the baseline?**
A: For proxy cost (Tier 1), your aggregate score across all IBM benchmarks must be lower than the baselines. For the Grand Prize (Tier 2), your OpenROAD results for WNS, TNS, and Area must surpass both SA and RePlAce baselines on NG45 designs.

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/partcleda/partcl-macro-place-challenge/issues)
- **Email**: contact@partcl.com

## 📄 License

This project is licensed under the Apache License 2.0 - see [LICENSE.md](LICENSE.md) for details.

## Competition Updates

The organizers may update or clarify rules, evaluation details, timelines, prizes, or infrastructure as needed to ensure fairness, technical accuracy, and smooth operation of the competition. Any updates will be communicated through official channels and will apply going forward.

Participation in the competition constitutes acceptance of the current rules and any subsequent updates. The organizers’ decisions regarding scoring, eligibility, and interpretation of these rules are final.

Submissions & contact information may be shared with sponsors.
