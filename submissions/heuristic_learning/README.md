# Heuristic Learning Placer

A deterministic portfolio placer for the Macro Place Challenge.

The placer generates legal hard-macro candidates from several heuristic recipes, uses a lightweight benchmark-feature selector to decide which candidates deserve official proxy scoring, and returns the lowest proxy candidate with zero hard-macro overlap.

Current checkpoint behavior:

- preserves the official `place(self, benchmark) -> Tensor` interface
- uses no network or external proprietary tools
- scores candidates with `macro_place.objective.compute_proxy_cost` when the local `PlacementCost` data is available
- adds official density/congestion hotspot-relief candidates derived from PlacementCost maps
- applies gated hard- and soft-macro hotspot passes selected from benchmark features
- runs a bounded official-score hard local search on high-cost small/medium cases
- supports optional recipe-score logging with `HL_DEBUG=1`

This is not yet top-10 competitive; it is a first heuristic-learning scaffold that improves over `submissions/will_seed/placer.py` on sampled IBM benchmarks and provides a path for offline recipe tuning.

## Offline tuning

Use the sweep helper to collect recipe-level training data for the selector:

```bash
uv run python scripts/sweep_heuristic_learning.py -b ibm02 -b ibm03 -b ibm07 --out /tmp/hl_sweep.jsonl
```

The JSONL records benchmark features, candidate labels, official proxy components, approximate selector cost, whether the runtime budget would have selected that candidate, the final soft-hotspot post-pass when enabled, and the bounded official-score local-search candidate when enabled. The current selector uses this evidence to skip radial-spread recipes on high-utilization IBM cases, protect hotspot-relief candidates, keep radial-mild spreading for lower-utilization cases where it improves congestion, and prune recipe variants that do not win in sweep data.

Sweep notes: safe-base hotspot and radial-plus-hotspot variants were tested and removed because they added runtime without beating the base hotspot or radial-mild winners on sampled IBM cases. A two-step medium hotspot candidate is enabled for dense low-degree-skew cases after improving `ibm02`, and a stronger two-step candidate is gated to dense high-degree-skew cases after improving `ibm06`. The official-score local search accepts single hard-macro moves away from top hotspot bins only when the evaluator score improves; current validation moved `ibm02` from `1.5967` to `1.5478` after extending the dense low-skew search to ten rounds, `ibm06` from `1.6975` to `1.6913` after extending the high-skew dense search to six rounds, and a forced `ibm08` probe from `1.4914` to `1.4892`. Very low-utilization cases such as `ibm18` are excluded because the first accepted local-search move only reached `1.7796` from `1.7807` after roughly 30 minutes.

Soft hotspot notes: a cheaper custom soft-macro hotspot pass replaced the built-in `optimize_stdcells()` path, which was too slow in prototype runs. The pass remains officially scored and is only kept if it improves proxy. It is enabled for low degree-skew cases where sampled evidence showed gains (`ibm01`, `ibm03`, `ibm07`, `ibm09`), plus narrower high-utilization and high-degree-skew buckets that improved `ibm02`/`ibm04` experiments without changing hard-macro legality.
