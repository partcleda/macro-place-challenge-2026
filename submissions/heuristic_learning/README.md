# Heuristic Learning Placer

A deterministic portfolio placer for the Macro Place Challenge.

The placer generates legal hard-macro candidates from several heuristic recipes, uses a lightweight benchmark-feature selector to decide which candidates deserve official proxy scoring, and returns the lowest proxy candidate with zero hard-macro overlap.

Current checkpoint behavior:

- preserves the official `place(self, benchmark) -> Tensor` interface
- uses no network or external proprietary tools
- scores candidates with `macro_place.objective.compute_proxy_cost` when the local `PlacementCost` data is available
- adds official density/congestion hotspot-relief candidates derived from PlacementCost maps
- applies a gated soft-macro hotspot pass on low degree-skew, non-high-utilization cases
- supports optional recipe-score logging with `HL_DEBUG=1`

This is not yet top-10 competitive; it is a first heuristic-learning scaffold that improves over `submissions/will_seed/placer.py` on sampled IBM benchmarks and provides a path for offline recipe tuning.

## Offline tuning

Use the sweep helper to collect recipe-level training data for the selector:

```bash
uv run python scripts/sweep_heuristic_learning.py -b ibm02 -b ibm03 -b ibm07 --out /tmp/hl_sweep.jsonl
```

The JSONL records benchmark features, candidate labels, official proxy components, approximate selector cost, and whether the runtime budget would have selected that candidate. The current selector uses this evidence to skip radial-spread recipes on high-utilization IBM cases, protect hotspot-relief candidates, keep radial-mild spreading for lower-utilization cases where it improves congestion, and prune recipe variants that do not win in sweep data.


Sweep notes: safe-base hotspot and radial-plus-hotspot variants were tested and removed because they added runtime without beating the base hotspot or radial-mild winners on sampled IBM cases.


Soft hotspot notes: a cheaper custom soft-macro hotspot pass replaced the built-in `optimize_stdcells()` path, which was too slow in prototype runs. The pass is gated to low degree-skew cases where sampled evidence showed gains (`ibm01`, `ibm03`, `ibm07`, `ibm09`) and skipped for high-utilization or high degree-skew cases where it hurt or only added runtime.
