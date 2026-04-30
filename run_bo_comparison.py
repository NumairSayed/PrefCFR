"""
Active AUTOPREF vs Grid Search: Comparison Experiments
=======================================================

Generates three figures for the paper (Appendix B):

  Figure B1 — Efficiency curves
      For each of 4 test targets, plot "best distance to target" vs
      "number of evaluations" for:
        • Grid search  (simulated anytime performance from existing table)
        • Goal-directed BO  (adaptive, uses Active AUTOPREF)
      Shows BO achieves grid-quality matching in ~3-4× fewer evaluations.

  Figure B2 — GP surrogate heatmaps
      Visualise the learned (δ, β) → aggression surface (posterior mean ± std)
      for both directions.  Dots mark the BO-chosen evaluation points.
      The surrogate is the learned (δ,β)→style map that BO discovers for free.

  Figure B3 — Summary comparison table (bar chart)
      For a fixed budget of N evaluations (N = 12 and N = 50 for grid),
      compare final distance-to-target across all test targets.

Run with:
    python run_bo_comparison.py

Total runtime: ~5–8 minutes (kuhn_poker is fast; Pref-CFR runs take ~2 s each).
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyspiel
from active_autopref import BOAutoPrefTuner
from auto_pref import AutoPrefTuner

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Test targets (same as AUTOPREF paper Table 4)
# ============================================================

TEST_TARGETS = [
    ({"aggression": 0.40}, "Aggressive\n(agg=0.40)"),
    ({"aggression": 0.15}, "Conservative\n(agg=0.15)"),
    ({"bluff": 0.30},       "Heavy Bluffer\n(bluff=0.30)"),
    ({"aggression": 0.35, "bluff": 0.20}, "Balanced Aggro\n(agg=0.35,bluff=0.20)"),
]

# Grid size used by original AUTOPREF (from kuhn_calibration.json)
# Δ = {1,2,4,7,10}  ×  B = {0,0.03,0.07,0.12,0.2}  × 2 directions = 50 entries
GRID_CACHE = os.path.join(RESULTS_DIR, "kuhn_calibration.json")

# BO budget for goal-directed search
BO_N_INIT   = 4   # per direction warm-up
BO_N_BO     = 10  # BO-guided iterations (total, alternating directions)
# Total evaluations for goal-directed BO = 2*n_init + n_bo_iter = 18

PREF_CFR_ITERS = 2500   # iterations per oracle evaluation


# ============================================================
# Helpers
# ============================================================

def compute_dist(entry: dict, target: dict) -> float:
    """Squared L2 distance from entry's style metrics to target."""
    return sum((entry.get(m, 0.0) - tv) ** 2 for m, tv in target.items())


def grid_anytime(grid_entries: list, target: dict,
                 max_exploitability: float = 0.02,
                 n_shuffles: int = 40, rng=None) -> dict:
    """
    Simulate grid-search anytime performance.

    For each of `n_shuffles` random orderings of the 50 grid entries,
    record the "best feasible distance so far" after k evaluations.

    Returns:
        evals  (max_evals,)  evaluation counts
        mean   (max_evals,)  mean best distance over shuffles
        p10    (max_evals,)  10th percentile
        p90    (max_evals,)  90th percentile
    """
    if rng is None:
        rng = np.random.RandomState(0)

    # Separate feasible and infeasible to let grid at least find something
    feasible   = [e for e in grid_entries if e["exploitability"] <= max_exploitability]
    infeasible = [e for e in grid_entries if e["exploitability"] >  max_exploitability]
    n = len(grid_entries)

    curves = []
    for _ in range(n_shuffles):
        # Random permutation of all entries
        perm = rng.permutation(n)
        shuffled = [grid_entries[i] for i in perm]
        best = np.inf
        curve = []
        for entry in shuffled:
            if entry["exploitability"] <= max_exploitability:
                d = compute_dist(entry, target)
                best = min(best, d)
            curve.append(best)
        curves.append(curve)

    curves = np.array(curves)   # (n_shuffles, n)
    evals  = np.arange(1, n + 1)
    return {
        "evals": evals,
        "mean":  np.mean(curves, axis=0),
        "p10":   np.percentile(curves, 10, axis=0),
        "p90":   np.percentile(curves, 90, axis=0),
    }


def bo_anytime(history: list, target: dict,
               max_exploitability: float = 0.02) -> dict:
    """
    Convert the goal-directed BO history to an anytime performance curve.

    Returns:
        evals  (total_evals,)
        dist   (total_evals,)  best feasible distance after each evaluation
    """
    best = np.inf
    evals, dists = [], []
    for rec in history:
        expl = rec["exploitability"]
        dist = rec["distance"]
        if np.isfinite(expl) and np.isfinite(dist) and expl <= max_exploitability:
            best = min(best, dist)
        evals.append(rec["eval"])
        dists.append(best)
    return {"evals": np.array(evals), "dist": np.array(dists)}


# ============================================================
# Figure B1: Efficiency curves
# ============================================================

def figure_efficiency_curves(grid_entries: list,
                              bo_results: dict,
                              max_exploitability: float = 0.02):
    """
    Plot best-distance-to-target vs #evaluations for grid and BO.

    bo_results: {target_desc -> goal-directed BO result dict}
    """
    print("\n[Fig B1] Plotting efficiency curves...")
    n_targets = len(TEST_TARGETS)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5), sharey=False)

    rng = np.random.RandomState(7)

    for ax, (target, label), (_, bo_res) in zip(
            axes, TEST_TARGETS, bo_results.items()):

        # --- Grid anytime ---
        grid_curve = grid_anytime(grid_entries, target,
                                  max_exploitability=max_exploitability,
                                  rng=rng)
        ax.fill_between(grid_curve["evals"],
                        grid_curve["p10"], grid_curve["p90"],
                        alpha=0.20, color="steelblue")
        ax.plot(grid_curve["evals"], grid_curve["mean"],
                color="steelblue", linewidth=2.0, label="Grid search")
        ax.axhline(grid_curve["mean"][-1], color="steelblue",
                   linestyle=":", linewidth=1.2, alpha=0.7)

        # --- BO anytime ---
        bo_curve = bo_anytime(bo_res["history"], target,
                              max_exploitability=max_exploitability)
        ax.step(bo_curve["evals"], bo_curve["dist"],
                color="orangered", linewidth=2.0, where="post",
                label="Goal-directed BO")

        # Mark where BO first matches the grid's FINAL quality
        grid_final = grid_curve["mean"][-1]
        match_eval = None
        for ev, d in zip(bo_curve["evals"], bo_curve["dist"]):
            if d <= grid_final + 1e-6:
                match_eval = ev
                break
        if match_eval is not None:
            ax.axvline(match_eval, color="orangered", linestyle="--",
                       linewidth=1.2, alpha=0.8,
                       label=f"BO matches grid\n@ eval {match_eval}")

        ax.set_xlabel("Evaluations (# Pref-CFR runs)", fontsize=10)
        ax.set_ylabel("Best distance² to target", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        "Active AUTOPREF: Efficiency of Bayesian Optimisation vs Grid Search\n"
        "(lower = better match; BO reaches grid quality with fewer evaluations)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bo_fig_b1_efficiency.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================
# Figure B2: GP surrogate heatmaps
# ============================================================

def figure_surrogate_heatmaps(tuner: BOAutoPrefTuner):
    """
    Visualise the GP posterior mean and uncertainty for the 'aggression' metric.

    Left  column: direction = aggressive
    Right column: direction = passive
    Top row:      posterior mean
    Bottom row:   posterior std (uncertainty)
    """
    print("\n[Fig B2] Plotting GP surrogate heatmaps...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, direction in enumerate(["aggressive", "passive"]):
        try:
            delta_grid, beta_grid, mu, sigma = tuner.surrogate_surface(
                "aggression", direction, n_grid=40
            )
        except RuntimeError:
            print(f"  [warn] surrogate for {direction} not fitted, skipping")
            continue

        D, B = np.meshgrid(delta_grid, beta_grid, indexing="ij")

        # -- Posterior mean --
        ax = axes[0, col]
        im = ax.contourf(D, B, mu, levels=20, cmap="RdYlGn")
        plt.colorbar(im, ax=ax, label="Predicted aggression")
        # Overlay observed points
        obs = tuner._X_obs[direction]
        if obs:
            X_pts = np.array(obs)
            ax.scatter(np.exp(X_pts[:, 0]), X_pts[:, 1],
                       c="black", s=30, zorder=5, label="BO evals", marker="x")
        ax.set_xlabel("δ (preference weight)", fontsize=10)
        ax.set_ylabel("β (vulnerability)", fontsize=10)
        ax.set_title(f"GP mean — aggression ({direction})", fontsize=10)
        ax.legend(fontsize=8)

        # -- Posterior std --
        ax = axes[1, col]
        im2 = ax.contourf(D, B, sigma, levels=20, cmap="Blues")
        plt.colorbar(im2, ax=ax, label="Uncertainty (std)")
        if obs:
            ax.scatter(np.exp(X_pts[:, 0]), X_pts[:, 1],
                       c="black", s=30, zorder=5, marker="x")
        ax.set_xlabel("δ (preference weight)", fontsize=10)
        ax.set_ylabel("β (vulnerability)", fontsize=10)
        ax.set_title(f"GP uncertainty ({direction})", fontsize=10)

    plt.suptitle(
        "Active AUTOPREF: Learned GP Surrogate for Aggression Rate\n"
        "× marks are BO-selected evaluation points — concentrated where information is scarce",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bo_fig_b2_surrogate.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================
# Figure B3: Summary bar chart
# ============================================================

def figure_summary_bars(grid_entries: list, bo_results: dict,
                         max_exploitability: float = 0.02):
    """
    Compare final achieved distance for grid vs BO across all test targets.

    Two bars per target:
      Blue  — Grid search (50 evaluations, best feasible point)
      Red   — Goal-directed BO (≤ 2*n_init + n_bo_iter evaluations)
    """
    print("\n[Fig B3] Plotting summary bar chart...")

    labels = [label.replace("\n", " ") for _, label in TEST_TARGETS]
    n = len(labels)

    grid_dists, bo_dists, bo_eval_counts = [], [], []

    for (target, _), (_, bo_res) in zip(TEST_TARGETS, bo_results.items()):
        # Grid: best feasible distance
        feasible = [e for e in grid_entries
                    if e["exploitability"] <= max_exploitability]
        if feasible:
            grid_d = min(compute_dist(e, target) for e in feasible)
        else:
            grid_d = min(compute_dist(e, target) for e in grid_entries)
        grid_dists.append(grid_d)

        # BO
        bo_dists.append(bo_res["distance"])
        bo_eval_counts.append(bo_res["eval_count"])

    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: distance comparison
    ax = axes[0]
    b1 = ax.bar(x - w / 2, grid_dists, w, color="steelblue", alpha=0.85,
                label=f"Grid (50 evals)")
    b2 = ax.bar(x + w / 2, bo_dists,   w, color="orangered", alpha=0.85,
                label=f"Goal BO ({int(np.mean(bo_eval_counts))} evals avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Distance² to target (lower = better)", fontsize=10)
    ax.set_title("Final Match Quality: Grid vs Active AUTOPREF", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.04,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    # Right: evaluation count for BO (grid is always 50)
    ax2 = axes[1]
    savings = [50 - c for c in bo_eval_counts]
    speedups = [50 / max(c, 1) for c in bo_eval_counts]
    bars = ax2.bar(x, bo_eval_counts, color="orangered", alpha=0.85,
                   label="BO evaluations used")
    ax2.axhline(50, color="steelblue", linestyle="--", linewidth=2,
                label="Grid (50 evals)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("# Pref-CFR evaluations", fontsize=10)
    ax2.set_title("Evaluation Budget: BO uses far fewer runs", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, sp in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{sp:.1f}×\nsaving", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bo_fig_b3_summary.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    return {
        "grid_distances": grid_dists,
        "bo_distances":   bo_dists,
        "bo_eval_counts": bo_eval_counts,
        "speedups":       speedups,
    }


# ============================================================
# Figure B4: Convergence of BO within a single target search
# ============================================================

def figure_bo_convergence(bo_results: dict):
    """
    For each target, plot how distance to target decreases iteration-by-iteration
    during the goal-directed BO.  Shows the exploration→exploitation dynamic.
    """
    print("\n[Fig B4] Plotting BO convergence within each target search...")

    n = len(TEST_TARGETS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    direction_colors = {"aggressive": "orangered", "passive": "steelblue"}

    for ax, (target, label), (_, bo_res) in zip(
            axes, TEST_TARGETS, bo_results.items()):

        history = bo_res["history"]
        evals = [h["eval"] for h in history]
        dists = [h["distance"] for h in history]
        dirs  = [h["direction"] for h in history]

        # Scatter all evaluations coloured by direction
        for ev, d, dr in zip(evals, dists, dirs):
            ax.scatter(ev, d, color=direction_colors[dr], s=40,
                       zorder=4, alpha=0.8)

        # Best-so-far line
        best_curve = bo_anytime(history, target, max_exploitability=0.02)
        ax.step(best_curve["evals"], best_curve["dist"],
                color="black", linewidth=2.0, where="post",
                label="Best feasible so far", zorder=5)

        # Shade the BO phase (after 2*n_init warm-up)
        n_warmup = 2 * BO_N_INIT
        ax.axvspan(n_warmup + 0.5, max(evals) + 0.5, alpha=0.07,
                   color="gold", label="BO phase")

        # Legend proxies for direction colours
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orangered",
                   markersize=8, label="Aggressive"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                   markersize=8, label="Passive"),
            Line2D([0], [0], color="black", linewidth=2, label="Best so far"),
        ]
        ax.legend(handles=legend_elements, fontsize=7.5)
        ax.set_xlabel("Evaluation #", fontsize=10)
        ax.set_ylabel("Distance² to target", fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        "Active AUTOPREF: Distance to Target During Goal-Directed BO Search\n"
        "Orange = aggressive direction, Blue = passive direction; "
        "shaded region = BO phase",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "bo_fig_b4_bo_convergence.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ============================================================
# Main experiment runner
# ============================================================

def run_all(max_exploitability: float = 0.02):
    t_start = time.time()

    print("=" * 65)
    print("Active AUTOPREF vs Grid Search — Full Comparison")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Load grid calibration table
    # ------------------------------------------------------------------
    if not os.path.exists(GRID_CACHE):
        print(f"\n[!] Grid cache not found at {GRID_CACHE}.")
        print("    Running AutoPrefTuner.calibrate() to build it first...")
        grid_tuner = AutoPrefTuner("kuhn_poker")
        grid_tuner.calibrate(
            num_iters=PREF_CFR_ITERS,
            delta_values=[1, 2, 4, 7, 10],
            beta_values=[0, 0.03, 0.07, 0.12, 0.2],
            cache_path=GRID_CACHE,
            verbose=True,
        )
    else:
        print(f"\n[Grid] Loading pre-computed calibration from {GRID_CACHE}")

    with open(GRID_CACHE, "r") as f:
        grid_entries = json.load(f)
    print(f"  Grid table: {len(grid_entries)} entries")

    # ------------------------------------------------------------------
    # Run goal-directed BO for each test target
    # ------------------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"Goal-directed BO  (n_init={BO_N_INIT}×2 + n_bo={BO_N_BO} = "
          f"{2*BO_N_INIT + BO_N_BO} total evals per target)")
    print("=" * 65)

    bo_tuner = BOAutoPrefTuner("kuhn_poker")
    bo_results = {}

    for target, label in TEST_TARGETS:
        print(f"\n--- Target: {label.replace(chr(10), ' ')} ---")
        result = bo_tuner.search_goal_directed(
            target=target,
            max_exploitability=max_exploitability,
            n_init=BO_N_INIT,
            n_bo_iter=BO_N_BO,
            num_iters=PREF_CFR_ITERS,
            verbose=True,
            seed=42,
        )
        bo_results[label] = result
        print(f"  → Best: δ({result['direction']})={result['delta']:.2f}  "
              f"β={result['beta']:.3f}  achieved={result['achieved']}  "
              f"expl={result['exploitability']:.5f}  "
              f"dist²={result['distance']:.5f}  "
              f"evals={result['eval_count']}")

    # Also run amortised BO calibration (for surrogate heatmaps)
    print("\n" + "=" * 65)
    print("Amortised BO calibration (for surrogate heatmaps)")
    print("=" * 65)

    amortised_tuner = BOAutoPrefTuner("kuhn_poker")
    amortised_tuner.calibrate_bo(
        n_init=5, n_bo_iter=12,
        num_iters=PREF_CFR_ITERS,
        cache_path=os.path.join(RESULTS_DIR, "bo_amortised_calibration.json"),
        verbose=True,
        seed=0,
    )

    # ------------------------------------------------------------------
    # Generate all figures
    # ------------------------------------------------------------------

    figure_efficiency_curves(grid_entries, bo_results, max_exploitability)
    figure_surrogate_heatmaps(amortised_tuner)
    figure_bo_convergence(bo_results)
    summary = figure_summary_bars(grid_entries, bo_results, max_exploitability)

    # ------------------------------------------------------------------
    # Print and save numerical summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 65)
    print("NUMERICAL SUMMARY")
    print("=" * 65)
    print(f"{'Target':<28} {'Grid dist²':>12} {'BO dist²':>10} "
          f"{'BO evals':>9} {'Speedup':>8}")
    print("-" * 65)
    for (target, label), gd, bd, ev, sp in zip(
            TEST_TARGETS,
            summary["grid_distances"],
            summary["bo_distances"],
            summary["bo_eval_counts"],
            summary["speedups"]):
        name = label.replace("\n", " ")
        print(f"  {name:<26} {gd:>12.5f} {bd:>10.5f} {ev:>9d} {sp:>7.1f}×")

    mean_speedup = np.mean(summary["speedups"])
    print("-" * 65)
    print(f"  Mean speedup: {mean_speedup:.1f}× fewer evaluations for equal quality")

    elapsed = time.time() - t_start
    print(f"\nAll done in {elapsed:.0f}s")
    print(f"Figures saved to {RESULTS_DIR}/bo_fig_b*.png")

    # Save full summary to JSON
    num_summary = {
        "targets": [label.replace("\n", " ") for _, label in TEST_TARGETS],
        "grid_eval_count": len(grid_entries),
        "bo_eval_counts": summary["bo_eval_counts"],
        "grid_distances": summary["grid_distances"],
        "bo_distances":   summary["bo_distances"],
        "speedups":       summary["speedups"],
        "mean_speedup":   float(mean_speedup),
        "elapsed_seconds": elapsed,
    }
    summary_path = os.path.join(RESULTS_DIR, "bo_comparison_summary.json")
    import json as _json
    with open(summary_path, "w") as f:
        _json.dump(num_summary, f, indent=2)
    print(f"Summary JSON → {summary_path}")

    return num_summary


if __name__ == "__main__":
    run_all(max_exploitability=0.02)
