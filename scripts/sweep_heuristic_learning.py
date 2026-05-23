#!/usr/bin/env python3
"""Sweep heuristic-learning placer recipes and record official proxy scores.

This is an offline tuning helper. It intentionally uses private methods from
submissions/heuristic_learning/placer.py so recipe data reflects the shipped
placer implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from macro_place.evaluate import IBM_BENCHMARKS
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


REPO_ROOT = Path(__file__).resolve().parents[1]
PLACER_PATH = REPO_ROOT / "submissions" / "heuristic_learning" / "placer.py"
TESTCASE_ROOT = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"


def _load_hl_module():
    spec = importlib.util.spec_from_file_location("heuristic_learning_placer", PLACER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {PLACER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _benchmarks_from_args(args):
    if args.all:
        return IBM_BENCHMARKS
    if args.benchmark:
        return args.benchmark
    return ["ibm01", "ibm02", "ibm03", "ibm07", "ibm10"]


def _candidate_rows(mod, placer, benchmark, plc):
    n_hard = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes[:n_hard].cpu().numpy().astype("float64")
    initial = benchmark.macro_positions[:n_hard].cpu().numpy().astype("float64")
    fixed_mask = benchmark.macro_fixed[:n_hard].cpu().numpy().astype(bool)
    movable = ~fixed_mask
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    edges, edge_weights, degree = mod._extract_edges(benchmark, plc)
    features = placer._features(benchmark, sizes, degree)
    recipes = placer._select_recipes(features)

    rows = []
    candidates = []
    safe_base = placer._will_seed_legalize(initial, movable, sizes, cw, ch)
    base = placer._legalize(initial, movable, sizes, cw, ch, gap=0.035)
    if not mod._validate_hard(base, sizes, initial, fixed_mask, cw, ch):
        base = safe_base
    candidates.append(("legalized_initial", base, {"seed": "initial", "recipe": "base"}))
    candidates.append(("will_seed_legalized", safe_base, {"seed": "initial", "recipe": "safe"}))
    hotspot_specs = [
        (label, "base", source, strength, steps)
        for label, source, strength, steps in placer._hard_hotspot_specs(features, n_hard, base)
    ]

    for label, source_name, source, strength, steps in hotspot_specs:
        start = time.time()
        relief = placer._hotspot_relief(
            source,
            benchmark,
            plc,
            movable,
            sizes,
            cw,
            ch,
            strength=strength,
            steps=steps,
        )
        candidates.append(
            (
                label,
                relief,
                {
                    "seed": "official_hotspot",
                    "source": source_name,
                    "recipe": label,
                    "strength": strength,
                    "steps": steps,
                    "generation_sec": time.time() - start,
                },
            )
        )

    for recipe in recipes:
        start = time.time()
        seed_pos = base.copy()
        if recipe["seed"] == "area_pack":
            order = sorted(range(n_hard), key=lambda i: (-sizes[i, 0] * sizes[i, 1], initial[i, 1]))
            seed_pos = placer._pack_by_order(order, initial, movable, sizes, cw, ch)
        elif recipe["seed"] == "degree_pack":
            order = sorted(range(n_hard), key=lambda i: (-degree[i], -sizes[i, 0] * sizes[i, 1]))
            seed_pos = placer._pack_by_order(order, initial, movable, sizes, cw, ch)
        elif recipe["seed"] == "x_order_pack":
            order = sorted(range(n_hard), key=lambda i: (initial[i, 0], initial[i, 1]))
            seed_pos = placer._pack_by_order(order, initial, movable, sizes, cw, ch)
        elif recipe["seed"] == "radial_spread":
            seed_pos = placer._radial_spread(base, movable, sizes, cw, ch, recipe.get("spread", 1.06))

        refined = placer._anneal(
            seed_pos,
            initial,
            movable,
            sizes,
            cw,
            ch,
            edges,
            edge_weights,
            degree,
            recipe,
        )
        meta = dict(recipe)
        meta["generation_sec"] = time.time() - start
        candidates.append((recipe["name"], refined, meta))

    limit_candidates = placer._limit_official_scoring(
        candidates, initial, sizes, cw, ch, edges, edge_weights, n_hard
    )
    limited_labels = {label for label, _, _ in limit_candidates}

    full_by_label = {}
    for label, pos, meta in candidates:
        score_start = time.time()
        candidate_pos = pos
        if not mod._validate_hard(candidate_pos, sizes, initial, fixed_mask, cw, ch):
            candidate_pos = placer._legalize(candidate_pos, movable, sizes, cw, ch, gap=0.035)
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(candidate_pos, dtype=torch.float32)
        costs = compute_proxy_cost(full, benchmark, plc)
        valid = costs["overlap_count"] == 0
        full_by_label[label] = full
        approx = None
        if len(edges) > 0:
            approx = placer._approx_cost(
                candidate_pos,
                initial,
                sizes,
                cw,
                ch,
                edges,
                edge_weights,
                {"wl": 0.78, "density": 0.46, "anchor": 0.025},
            )
        rows.append(
            {
                "benchmark": benchmark.name,
                "label": label,
                "selected_by_budget": label in limited_labels,
                "valid": valid,
                "proxy_cost": float(costs["proxy_cost"]),
                "wirelength_cost": float(costs["wirelength_cost"]),
                "density_cost": float(costs["density_cost"]),
                "congestion_cost": float(costs["congestion_cost"]),
                "overlap_count": int(costs["overlap_count"]),
                "approx_cost": None if approx is None else float(approx),
                "score_sec": time.time() - score_start,
                "recipe": meta,
                "features": features,
            }
        )

    soft_strength = placer._soft_strength(features, n_hard, benchmark.num_soft_macros)
    runtime_rows = [row for row in rows if row["selected_by_budget"] and row["valid"]]
    if soft_strength is not None and runtime_rows:
        best_hard = min(runtime_rows, key=lambda row: row["proxy_cost"])
        score_start = time.time()
        soft_full = placer._soft_hotspot_relief(
            full_by_label[best_hard["label"]],
            benchmark,
            plc,
            strength=soft_strength,
            steps=1,
        )
        costs = compute_proxy_cost(soft_full, benchmark, plc)
        full_by_label["soft_hotspot_mild"] = soft_full
        rows.append(
            {
                "benchmark": benchmark.name,
                "label": "soft_hotspot_mild",
                "selected_by_budget": True,
                "valid": costs["overlap_count"] == 0,
                "proxy_cost": float(costs["proxy_cost"]),
                "wirelength_cost": float(costs["wirelength_cost"]),
                "density_cost": float(costs["density_cost"]),
                "congestion_cost": float(costs["congestion_cost"]),
                "overlap_count": int(costs["overlap_count"]),
                "approx_cost": None,
                "score_sec": time.time() - score_start,
                "recipe": {
                    "recipe": "soft_hotspot_mild",
                    "source": best_hard["label"],
                    "strength": soft_strength,
                    "steps": 1,
                },
                "features": features,
            }
        )

    runtime_rows = [row for row in rows if row["selected_by_budget"] and row["valid"]]
    if runtime_rows:
        best_runtime = min(runtime_rows, key=lambda row: row["proxy_cost"])
        if placer._enable_official_hard_search(
            features, n_hard, best_runtime["proxy_cost"]
        ):
            score_start = time.time()
            local_full, local_score = placer._official_hard_local_search(
                full_by_label[best_runtime["label"]],
                benchmark,
                plc,
                best_runtime["proxy_cost"],
                features,
                time.time(),
            )
            costs = compute_proxy_cost(local_full, benchmark, plc)
            rows.append(
                {
                    "benchmark": benchmark.name,
                    "label": "official_hard_local_search",
                    "selected_by_budget": True,
                    "valid": costs["overlap_count"] == 0,
                    "proxy_cost": float(costs["proxy_cost"]),
                    "wirelength_cost": float(costs["wirelength_cost"]),
                    "density_cost": float(costs["density_cost"]),
                    "congestion_cost": float(costs["congestion_cost"]),
                    "overlap_count": int(costs["overlap_count"]),
                    "approx_cost": None,
                    "score_sec": time.time() - score_start,
                    "recipe": {
                        "recipe": "official_hard_local_search",
                        "source": best_runtime["label"],
                        "proxy_from_method": float(local_score),
                    },
                    "features": features,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", "-b", action="append", help="IBM benchmark name; repeatable")
    parser.add_argument("--all", action="store_true", help="sweep all IBM benchmarks")
    parser.add_argument("--out", type=Path, default=Path("runs/heuristic_learning/recipe_sweep.jsonl"))
    args = parser.parse_args()

    if not TESTCASE_ROOT.exists():
        raise SystemExit("external/MacroPlacement submodule is not initialized")

    mod = _load_hl_module()
    placer = mod.HeuristicLearningPlacer()
    benchmarks = _benchmarks_from_args(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    with args.out.open("w") as f:
        for name in benchmarks:
            benchmark, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / name))
            start = time.time()
            rows = _candidate_rows(mod, placer, benchmark, plc)
            rows.sort(key=lambda row: (not row["valid"], row["proxy_cost"]))
            all_rows.extend(rows)
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            best = rows[0]
            print(
                f"{name}: best={best['label']} proxy={best['proxy_cost']:.4f} "
                f"valid={best['valid']} candidates={len(rows)} elapsed={time.time() - start:.1f}s"
            )

    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
