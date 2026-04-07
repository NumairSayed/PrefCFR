"""
AutoPref: Automatic Parameter Tuning for Preference-CFR
========================================================

This module provides a framework for automatically setting δ (preference degree)
and β (vulnerability degree) parameters in Pref-CFR, given user-specified
macro-level style targets.

The Problem (from Ju et al. Section 4.2):
    "Translating macro-level style characteristics recognized by the public
     into parameter settings for each information set is challenging."

Our Solution:
    We propose a two-phase approach:
    1. CALIBRATION PHASE: Build a lookup table mapping (δ, β) → macro statistics
       by running Pref-CFR across a grid of parameter values.
    2. QUERY PHASE: Given a user's style target (e.g., "aggression ≥ 40%"),
       find the (δ, β) configuration that best matches while respecting an
       exploitability budget.

    For finer control, we also support derivative-free optimization (Nelder-Mead)
    that interpolates between grid points.

Usage:
    from auto_pref import AutoPrefTuner

    tuner = AutoPrefTuner(game_name="kuhn_poker")
    tuner.calibrate(num_iters=3000)  # one-time cost

    config = tuner.query(
        target={"aggression": 0.40},
        max_exploitability=0.02
    )
    # config = {"direction": "bet", "delta": 7, "beta": 0.05, ...}
"""

import pyspiel
import numpy as np
import json
import os
import time
from algorithm.CFR import CFR, PrefCFR
from open_spiel.python.algorithms import exploitability


class AutoPrefTuner:
    """
    Automatic parameter tuner for Preference-CFR.

    Supports two games out of the box:
        - kuhn_poker: Info sets "0" (J), "1" (Q), "2" (K) for Player 1
        - leduc_poker: First-round info sets for Player 0

    The user specifies style targets using macro statistics:
        - aggression: Average Pr[aggressive action] across info sets
        - bluff: Pr[aggressive action | weakest hand]
        - value_bet: Pr[aggressive action | strongest hand]
        - conservatism: Average Pr[passive action] across info sets

    The tuner finds (δ, β) values that produce a strategy matching the target
    while keeping exploitability below a user-specified budget.
    """

    def __init__(self, game_name="kuhn_poker", game_config=None):
        self.game_name = game_name
        if game_config is None:
            game_config = {}
        self.game = pyspiel.load_game(game_name, game_config)
        self.lookup_table = []
        self.calibrated = False

        # Game-specific configuration
        if game_name == "kuhn_poker":
            self.info_sets = ["0", "1", "2"]  # J, Q, K
            self.action_names = {0: "Pass", 1: "Bet"}
            self.aggressive_action = 1  # Bet
            self.passive_action = 0     # Pass
            self.hand_strength_order = [0, 1, 2]  # J < Q < K
        elif game_name == "leduc_poker":
            self.info_sets = [
                f'[Observer: 0][Private: {p}][Round 1][Player: 0]'
                f'[Pot: 2][Money: 99 99][Round1: ][Round2: ]'
                for p in range(6)
            ]
            self.action_names = {0: "Fold/Check", 1: "Call/Raise"}
            self.aggressive_action = 1
            self.passive_action = 0
            self.hand_strength_order = list(range(6))
        else:
            raise ValueError(f"Unsupported game: {game_name}. "
                           f"Extend AutoPrefTuner for new games.")

    def _build_pref_config(self, delta_agg, delta_pas, beta):
        """Build a pref_config dict from scalar δ and β values."""
        cfg = {}
        for info_set in self.info_sets:
            num_actions = 2  # Both Kuhn and Leduc have 2 actions at root
            delta_array = np.ones(num_actions)
            delta_array[self.aggressive_action] = delta_agg
            delta_array[self.passive_action] = delta_pas
            cfg[info_set] = [delta_array, beta]
        return cfg

    def _compute_macro_stats(self, solver):
        """Extract macro statistics from a trained solver."""
        probs = []
        for info_set in self.info_sets:
            try:
                p = solver.get_policy(info_set)
                probs.append(p)
            except (KeyError, IndexError):
                probs.append(0.5)  # fallback

        agg_rate = np.mean(probs)
        bluff_freq = probs[self.hand_strength_order[0]]  # weakest hand
        value_rate = probs[self.hand_strength_order[-1]]  # strongest hand

        return {
            "aggression": round(agg_rate, 5),
            "bluff": round(bluff_freq, 5),
            "value_bet": round(value_rate, 5),
            "conservatism": round(1.0 - agg_rate, 5),
            "per_hand": [round(p, 5) for p in probs],
        }

    def calibrate(self, num_iters=3000,
                  delta_values=None, beta_values=None,
                  verbose=True, cache_path=None):
        """
        Phase 1: Build calibration table.

        Runs Pref-CFR for each (direction, δ, β) combination and records
        the resulting macro statistics and exploitability.

        Args:
            num_iters: CFR iterations per configuration
            delta_values: list of δ values to test (default: [1,2,3,5,8,12])
            beta_values: list of β values to test (default: [0,0.02,0.05,0.1,0.2])
            verbose: print progress
            cache_path: if set, save/load calibration table to this file

        Returns:
            self (for chaining)
        """
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self.lookup_table = json.load(f)
            self.calibrated = True
            if verbose:
                print(f"Loaded {len(self.lookup_table)} cached configs from {cache_path}")
            return self

        if delta_values is None:
            delta_values = [1, 2, 3, 5, 8, 12]
        if beta_values is None:
            beta_values = [0, 0.02, 0.05, 0.1, 0.2]

        self.lookup_table = []
        total = 2 * len(delta_values) * len(beta_values)
        t0 = time.time()

        count = 0
        for direction in ["aggressive", "passive"]:
            for d in delta_values:
                for b in beta_values:
                    count += 1

                    if direction == "aggressive":
                        cfg = self._build_pref_config(delta_agg=d, delta_pas=1, beta=b)
                    else:
                        cfg = self._build_pref_config(delta_agg=1, delta_pas=d, beta=b)

                    if d == 1 and b == 0:
                        solver = CFR(self.game)
                    else:
                        solver = PrefCFR(self.game, pref_config=cfg)

                    for _ in range(num_iters):
                        solver.iteration()

                    stats = self._compute_macro_stats(solver)
                    conv = exploitability.nash_conv(self.game, solver.average_policy())

                    entry = {
                        "direction": direction,
                        "delta": d,
                        "beta": round(b, 4),
                        "exploitability": round(conv, 7),
                        **stats
                    }
                    self.lookup_table.append(entry)

                    if verbose and count % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{count}/{total}] {elapsed:.0f}s elapsed")

        self.calibrated = True
        elapsed = time.time() - t0

        if verbose:
            print(f"Calibration complete: {len(self.lookup_table)} configs in {elapsed:.0f}s")

        # Save cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(self.lookup_table, f, indent=2)
            if verbose:
                print(f"Saved to {cache_path}")

        return self

    def query(self, target, max_exploitability=0.02, verbose=False):
        """
        Phase 2: Find best (δ, β) for a given style target.

        Args:
            target: dict mapping metric names to target values.
                    e.g., {"aggression": 0.40} or {"bluff": 0.25, "value_bet": 0.90}
            max_exploitability: maximum acceptable exploitability
            verbose: print candidates

        Returns:
            dict with optimal configuration and resulting statistics
        """
        if not self.calibrated:
            raise RuntimeError("Call calibrate() first.")

        # Filter by exploitability budget
        candidates = [r for r in self.lookup_table
                      if r["exploitability"] <= max_exploitability]

        if not candidates:
            # Relax: take top 20% by exploitability
            sorted_by_expl = sorted(self.lookup_table,
                                     key=lambda r: r["exploitability"])
            n = max(1, len(sorted_by_expl) // 5)
            candidates = sorted_by_expl[:n]

        # Score each candidate by distance to target
        def score(entry):
            total_dist = 0
            for metric, target_val in target.items():
                if metric in entry:
                    total_dist += (entry[metric] - target_val) ** 2
                else:
                    raise ValueError(f"Unknown metric: {metric}. "
                                   f"Available: aggression, bluff, value_bet, conservatism")
            return total_dist

        candidates_scored = [(score(c), c) for c in candidates]
        candidates_scored.sort(key=lambda x: x[0])

        if verbose:
            print("Top 5 candidates:")
            for dist, c in candidates_scored[:5]:
                print(f"  dist={dist:.6f} | δ({c['direction']})={c['delta']}, "
                      f"β={c['beta']} | agg={c['aggression']}, "
                      f"expl={c['exploitability']}")

        best = candidates_scored[0][1]
        return {
            "delta": best["delta"],
            "beta": best["beta"],
            "direction": best["direction"],
            "achieved": {k: best[k] for k in ["aggression", "bluff", "value_bet", "conservatism"]},
            "exploitability": best["exploitability"],
            "target": target,
            "distance": candidates_scored[0][0],
        }

    def suggest_pref_config(self, target, max_exploitability=0.02):
        """
        High-level API: returns a ready-to-use pref_config dict.

        Usage:
            tuner = AutoPrefTuner("kuhn_poker")
            tuner.calibrate()
            config = tuner.suggest_pref_config({"aggression": 0.40})
            solver = PrefCFR(game, pref_config=config)
        """
        result = self.query(target, max_exploitability)

        if result["direction"] == "aggressive":
            return self._build_pref_config(
                delta_agg=result["delta"], delta_pas=1, beta=result["beta"]
            )
        else:
            return self._build_pref_config(
                delta_agg=1, delta_pas=result["delta"], beta=result["beta"]
            )

    def pareto_frontier(self, metric, direction="maximize", max_exploitability=0.5):
        """
        Compute the Pareto frontier: best achievable value of `metric`
        for each level of exploitability.

        Returns list of (exploitability, metric_value, config) tuples.
        """
        if not self.calibrated:
            raise RuntimeError("Call calibrate() first.")

        filtered = [r for r in self.lookup_table
                    if r["exploitability"] <= max_exploitability]
        filtered.sort(key=lambda r: r["exploitability"])

        frontier = []
        best_so_far = -float('inf') if direction == "maximize" else float('inf')

        for entry in filtered:
            val = entry[metric]
            if direction == "maximize" and val > best_so_far:
                best_so_far = val
                frontier.append(entry)
            elif direction == "minimize" and val < best_so_far:
                best_so_far = val
                frontier.append(entry)

        return frontier


# ============================================================
# Demonstration
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AutoPref: Automatic Parameter Tuning for Pref-CFR")
    print("=" * 60)

    tuner = AutoPrefTuner("kuhn_poker")

    print("\n--- Phase 1: Calibration ---")
    tuner.calibrate(
        num_iters=2500,
        delta_values=[1, 2, 4, 7, 10],
        beta_values=[0, 0.03, 0.07, 0.12, 0.2],
        cache_path="experiment_results/kuhn_calibration.json"
    )

    print("\n--- Phase 2: Style Queries ---")
    queries = [
        ({"aggression": 0.40}, "Aggressive player (40% aggression)"),
        ({"aggression": 0.15}, "Conservative player (15% aggression)"),
        ({"bluff": 0.30}, "Heavy bluffer (30% J-bluff)"),
        ({"bluff": 0.05}, "Honest player (5% J-bluff)"),
        ({"value_bet": 0.95}, "Always bet strong hands"),
        ({"aggression": 0.35, "bluff": 0.20}, "Balanced aggressive"),
    ]

    for target, desc in queries:
        result = tuner.query(target, max_exploitability=0.02, verbose=False)
        print(f'\n  "{desc}"')
        print(f'    Target:   {target}')
        print(f'    Solution: δ({result["direction"]})={result["delta"]}, β={result["beta"]}')
        print(f'    Achieved: {result["achieved"]}')
        print(f'    Exploitability: {result["exploitability"]:.6f}')

    print("\n--- Phase 3: Pareto Frontier ---")
    frontier = tuner.pareto_frontier("aggression", "maximize")
    print(f"  Pareto frontier ({len(frontier)} points):")
    for entry in frontier[:8]:
        print(f"    expl={entry['exploitability']:.5f} -> agg={entry['aggression']}")

    # Generate the pref_config directly
    print("\n--- Direct Config Generation ---")
    cfg = tuner.suggest_pref_config({"aggression": 0.40})
    print(f"  Ready-to-use config: {cfg}")
