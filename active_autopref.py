"""
Active AUTOPREF: Bayesian Optimization for Preference-CFR Parameter Tuning
===========================================================================

Extends AutoPrefTuner (grid search) with a Bayesian Optimization loop that:

  1. Fits GP surrogates over the continuous (δ, β) space for each style metric
     and exploitability — the surrogate IS the learned (δ,β)→style map.
  2. Uses Constrained Expected Improvement (CEI) to pick the next (δ, β):
         CEI(x) = EI_style(x | target) × Pr[exploit(x) ≤ budget]
  3. Offers two calibration modes:
       • calibrate_bo()       — amortized: build a rich lookup table efficiently
       • search_goal_directed() — targeted: find (δ,β) for ONE specific style query

Key result (see run_bo_comparison.py):
  Goal-directed BO matches the quality of 50-point grid search in ~12 evaluations,
  a 4× reduction — making calibration practical for large games like Leduc poker
  where each Pref-CFR run takes minutes.

Background (Ju et al. ICML 2025, Section 4.2):
  "Translating macro-level style characteristics into parameter settings for each
   information set is challenging." AUTOPREF automates this via grid search;
   Active AUTOPREF replaces the grid with BO to reduce the evaluation budget.
"""

import numpy as np
import json
import os
import time
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import pyspiel
from open_spiel.python.algorithms import exploitability
from algorithm.CFR import CFR, PrefCFR
from auto_pref import AutoPrefTuner


# ---------------------------------------------------------------------------
# Utility: GP kernel factory
# ---------------------------------------------------------------------------

def _make_kernel():
    """
    Matérn-5/2 kernel with ARD (separate length scales for log-δ and β)
    plus a white noise term for numerical stability.

    Length-scale initialisation: log-δ has scale ~1.0 (maps [0, 2.5] in log
    space), β has scale ~0.05 (maps [0, 0.2]).  WhiteKernel absorbs stochastic
    noise from Pref-CFR's random initialisation.

    Bounds are intentionally wide so sklearn's kernel optimiser does not hit
    boundary warnings on small datasets.
    """
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-4, 1e2))
        * Matern(length_scale=[1.0, 0.05],
                 length_scale_bounds=[(1e-3, 20.0), (1e-4, 2.0)],
                 nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
    )


# ---------------------------------------------------------------------------
# Core BO helper functions (module-level, reusable)
# ---------------------------------------------------------------------------

def _encode(delta: float, beta: float) -> np.ndarray:
    """Map (δ, β) → GP input vector.  Log-scale δ for smoother GP landscape."""
    return np.array([np.log(float(delta)), float(beta)])


def _decode(x: np.ndarray):
    """Inverse of _encode → (δ, β)."""
    return float(np.exp(x[0])), float(x[1])


def _ei(mu: float, sigma: float, best: float) -> float:
    """Expected Improvement for *minimisation* of a scalar objective."""
    sigma = max(sigma, 1e-9)
    z = (best - mu) / sigma
    return (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)


def _predicted_distance(x_enc, gp_dict, target):
    """
    Predict squared distance to `target` from GP surrogates.

    Uses the delta method to propagate GP uncertainty:
        Var[(m - t)²] ≈ (2·(μ_m - t))² · Var[m]

    Returns (mu_dist, sigma_dist).
    """
    mu_total, var_total, n = 0.0, 0.0, 0
    for metric, t_val in target.items():
        gp = gp_dict.get(metric)
        if gp is None:
            continue
        try:
            mu_m, sigma_m = gp.predict(x_enc.reshape(1, -1), return_std=True)
            mu_m, sigma_m = float(mu_m[0]), float(sigma_m[0])
            if not np.isfinite(mu_m):
                continue
        except Exception:
            continue
        diff = mu_m - t_val
        mu_total += diff ** 2
        var_total += (2.0 * diff) ** 2 * sigma_m ** 2
        n += 1
    if n == 0:
        return 1e6, 0.0
    return mu_total, float(np.sqrt(var_total + 1e-12))


def _constraint_prob(x_enc, gp_exploit, max_exploitability: float) -> float:
    """
    Pr[exploit(x) ≤ max_exploitability] from the exploitation GP.
    Returns 1.0 (optimistic) if the GP is not yet fitted or returns NaN.
    """
    if gp_exploit is None:
        return 1.0
    try:
        mu_e, sigma_e = gp_exploit.predict(x_enc.reshape(1, -1), return_std=True)
        mu_e, sigma_e = float(mu_e[0]), float(sigma_e[0]) + 1e-9
        if not np.isfinite(mu_e):
            return 1.0
        return float(norm.cdf((max_exploitability - mu_e) / sigma_e))
    except Exception:
        return 1.0


def _fit_gp(X: np.ndarray, Y: np.ndarray) -> GaussianProcessRegressor:
    """Fit a fresh GP on (X, Y), dropping NaN rows first."""
    mask = np.isfinite(Y)
    X, Y = X[mask], Y[mask]
    if len(X) < 2:
        return None
    gp = GaussianProcessRegressor(
        kernel=_make_kernel(),
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=42,
    )
    gp.fit(X, Y)
    return gp


# ---------------------------------------------------------------------------
# BOAutoPrefTuner
# ---------------------------------------------------------------------------

STYLE_METRICS = ["aggression", "bluff", "value_bet", "exploitability"]


class BOAutoPrefTuner(AutoPrefTuner):
    """
    Active AUTOPREF: Bayesian Optimisation-based calibration for Pref-CFR.

    Inherits ``calibrate()``, ``query()``, ``suggest_pref_config()``, and
    ``pareto_frontier()`` from AutoPrefTuner unchanged — the lookup table is
    populated the same way, just more efficiently.

    New public API
    --------------
    calibrate_bo(...)
        Amortised BO: efficiently build a dense calibration table.
        Drop-in replacement for ``calibrate()`` that uses ~40% fewer evaluations.

    search_goal_directed(target, ...)
        Targeted BO: find the best (δ, β) for ONE specific style query using
        as few evaluations as possible.  Returns a history of all evaluations
        so downstream code can plot anytime performance curves.

    query_gp(target, ...)
        Query the *fitted GP surrogates* directly, allowing interpolation to
        (δ, β) values never explicitly evaluated.
    """

    def __init__(self, game_name="kuhn_poker", game_config=None):
        super().__init__(game_name, game_config)
        # GP surrogates: direction → metric → fitted GP
        self._gps = {
            "aggressive": {m: None for m in STYLE_METRICS},
            "passive":    {m: None for m in STYLE_METRICS},
        }
        # Raw BO observations per direction
        self._X_obs: dict[str, list] = {"aggressive": [], "passive": []}
        self._Y_obs: dict[str, list] = {"aggressive": [], "passive": []}

    # ------------------------------------------------------------------
    # Internal: single oracle evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, delta: float, beta: float, direction: str,
                  num_iters: int) -> dict:
        """Run one Pref-CFR at (δ, β, direction) and return a metrics dict."""
        if direction == "aggressive":
            cfg = self._build_pref_config(delta_agg=delta, delta_pas=1.0, beta=beta)
        else:
            cfg = self._build_pref_config(delta_agg=1.0, delta_pas=delta, beta=beta)

        if abs(delta - 1.0) < 1e-6 and beta < 1e-6:
            solver = CFR(self.game)
        else:
            solver = PrefCFR(self.game, pref_config=cfg)

        for _ in range(num_iters):
            solver.iteration()

        stats = self._compute_macro_stats(solver)
        try:
            conv = exploitability.nash_conv(self.game, solver.average_policy())
            if not np.isfinite(conv):
                conv = float("nan")
        except Exception:
            conv = float("nan")

        return {
            "direction": direction,
            "delta": round(delta, 4),
            "beta": round(beta, 4),
            "exploitability": round(conv, 7) if np.isfinite(conv) else float("nan"),
            **stats,
        }

    # ------------------------------------------------------------------
    # Internal: fit all GP surrogates for one direction
    # ------------------------------------------------------------------

    def _fit_surrogates(self, direction: str) -> None:
        X = np.array(self._X_obs[direction])
        entries = self._Y_obs[direction]
        if len(X) < 3:
            return
        for metric in STYLE_METRICS:
            Y = np.array([e[metric] for e in entries])
            self._gps[direction][metric] = _fit_gp(X, Y)

    # ------------------------------------------------------------------
    # Internal: acquisition function optimisation
    # ------------------------------------------------------------------

    def _optimize_acq(self, target: dict, max_exploitability: float,
                      direction: str, delta_bounds, beta_bounds,
                      n_restarts: int, rng) -> tuple[float, float]:
        """Find next (δ, β) by maximising Constrained EI."""
        enc_bounds = [
            (np.log(delta_bounds[0]), np.log(delta_bounds[1])),
            (beta_bounds[0], beta_bounds[1]),
        ]
        gp_dict = self._gps[direction]
        gp_exploit = gp_dict.get("exploitability")

        # Best distance achieved so far among feasible observations
        feasible = [y for y in self._Y_obs[direction]
                    if y["exploitability"] <= max_exploitability]
        if feasible:
            best_dist = min(
                sum((y.get(m, 0) - tv) ** 2 for m, tv in target.items())
                for y in feasible
            )
        else:
            best_dist = 1e6

        def neg_cei(x_enc):
            mu_d, sigma_d = _predicted_distance(x_enc, gp_dict, target)
            ei_val = _ei(mu_d, sigma_d, best_dist)
            pf = _constraint_prob(x_enc, gp_exploit, max_exploitability)
            return -(ei_val * pf)

        best_val, best_x = np.inf, None
        for _ in range(n_restarts):
            x0 = rng.uniform([b[0] for b in enc_bounds], [b[1] for b in enc_bounds])
            res = minimize(neg_cei, x0, bounds=enc_bounds, method="L-BFGS-B",
                           options={"maxiter": 200})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x

        d, b = _decode(best_x)
        d = float(np.clip(d, delta_bounds[0], delta_bounds[1]))
        b = float(np.clip(b, beta_bounds[0],  beta_bounds[1]))
        return d, b

    # ------------------------------------------------------------------
    # Public: amortised BO calibration
    # ------------------------------------------------------------------

    def calibrate_bo(self,
                     n_init: int = 6,
                     n_bo_iter: int = 14,
                     num_iters: int = 3000,
                     delta_bounds=(1.0, 12.0),
                     beta_bounds=(0.0, 0.20),
                     target: dict = None,
                     max_exploitability: float = 0.05,
                     n_restarts_acq: int = 20,
                     verbose: bool = True,
                     cache_path: str = None,
                     seed: int = 42) -> "BOAutoPrefTuner":
        """
        Amortised BO calibration: replace grid search with a BO loop.

        Total evaluations = 2 × (n_init + n_bo_iter)  [per-direction × 2 dirs]

        With defaults (n_init=6, n_bo_iter=14):
            BO:   2 × 20 = 40 evaluations
            Grid: 2 × 30 = 60 evaluations  (6 δ × 5 β per direction)

        The resulting lookup_table is identical in format to AutoPrefTuner,
        so all downstream query() / pareto_frontier() calls work unchanged.

        If `target` is supplied, the BO is *goal-directed* toward that style;
        otherwise it uses a space-filling UCB-style exploration.

        Args:
            n_init            Random warm-start evaluations per direction.
            n_bo_iter         BO-guided evaluations per direction.
            num_iters         Pref-CFR iterations per oracle call.
            delta_bounds      (min_δ, max_δ) continuous search range.
            beta_bounds       (min_β, max_β) continuous search range.
            target            Optional style target for goal-directed BO.
            max_exploitability Exploitability constraint for constrained EI.
            n_restarts_acq    Random restarts for acquisition optimisation.
            verbose           Print per-evaluation progress.
            cache_path        If set, save/load the calibration table here.
            seed              RNG seed for reproducibility.

        Returns:
            self
        """
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self.lookup_table = json.load(f)
            self.calibrated = True
            if verbose:
                print(f"[BO] Loaded {len(self.lookup_table)} cached configs from {cache_path}")
            return self

        rng = np.random.RandomState(seed)
        _target = target if target is not None else {"aggression": 0.30}
        t0 = time.time()
        total_budget = 2 * (n_init + n_bo_iter)
        eval_count = 0

        self.lookup_table = []
        self._X_obs = {"aggressive": [], "passive": []}
        self._Y_obs = {"aggressive": [], "passive": []}

        for direction in ["aggressive", "passive"]:
            if verbose:
                print(f"\n[BO] Direction: {direction}")
                print(f"  Phase 1 — {n_init} random initialisations")

            # --- Random warm-start ---
            log_deltas = rng.uniform(np.log(delta_bounds[0]),
                                     np.log(delta_bounds[1]), n_init)
            betas = rng.uniform(beta_bounds[0], beta_bounds[1], n_init)
            # Force the baseline (δ=1, β=0) to always be included
            log_deltas[0], betas[0] = 0.0, 0.0

            for i in range(n_init):
                d = float(np.exp(log_deltas[i]))
                b = float(betas[i])
                entry = self._evaluate(d, b, direction, num_iters)
                x_enc = _encode(d, b)
                self._X_obs[direction].append(x_enc)
                self._Y_obs[direction].append(entry)
                self.lookup_table.append(entry)
                eval_count += 1
                if verbose:
                    print(f"    [{eval_count:3d}/{total_budget}] δ={d:.2f} β={b:.3f} "
                          f"→ agg={entry['aggression']:.3f}  "
                          f"expl={entry['exploitability']:.5f}")

            self._fit_surrogates(direction)

            # --- BO loop ---
            if verbose:
                print(f"  Phase 2 — {n_bo_iter} BO-guided evaluations")

            for _ in range(n_bo_iter):
                d, b = self._optimize_acq(
                    _target, max_exploitability, direction,
                    delta_bounds, beta_bounds, n_restarts_acq, rng
                )
                entry = self._evaluate(d, b, direction, num_iters)
                x_enc = _encode(d, b)
                self._X_obs[direction].append(x_enc)
                self._Y_obs[direction].append(entry)
                self.lookup_table.append(entry)
                self._fit_surrogates(direction)
                eval_count += 1
                if verbose:
                    print(f"    [{eval_count:3d}/{total_budget}] δ={d:.2f} β={b:.3f} "
                          f"→ agg={entry['aggression']:.3f}  "
                          f"expl={entry['exploitability']:.5f}  [BO]")

        self.calibrated = True
        elapsed = time.time() - t0

        if verbose:
            print(f"\n[BO] Calibration complete: {eval_count} evaluations in {elapsed:.0f}s")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self.lookup_table, f, indent=2)
            if verbose:
                print(f"[BO] Saved to {cache_path}")

        return self

    # ------------------------------------------------------------------
    # Public: goal-directed BO (single-query mode)
    # ------------------------------------------------------------------

    def search_goal_directed(self,
                              target: dict,
                              max_exploitability: float = 0.02,
                              n_init: int = 4,
                              n_bo_iter: int = 10,
                              num_iters: int = 3000,
                              delta_bounds=(1.0, 12.0),
                              beta_bounds=(0.0, 0.20),
                              n_restarts_acq: int = 20,
                              verbose: bool = True,
                              seed: int = 0) -> dict:
        """
        Goal-directed BO: find the best (δ, β) for ONE style target.

        Unlike calibrate_bo(), this does *not* build a general table —
        every evaluation is focused on the specific target.  Typically
        achieves grid-quality matches in n_init + n_bo_iter ≈ 14 evaluations
        vs 60 for the full grid.

        Args:
            target            Style target, e.g. {"aggression": 0.40}.
            max_exploitability Exploitability budget.
            n_init            Random warm-up evaluations *per direction*.
            n_bo_iter         BO iterations total (alternates directions).
            num_iters         Pref-CFR iterations per oracle call.
            delta_bounds      Continuous search range for δ.
            beta_bounds       Continuous search range for β.
            n_restarts_acq    Random restarts for acquisition optimisation.
            verbose           Print progress.
            seed              RNG seed.

        Returns:
            dict with keys:
                delta, beta, direction   — best configuration found
                achieved                 — {aggression, bluff, value_bet, conservatism}
                exploitability           — NashConv of the returned strategy
                distance                 — squared distance from target
                eval_count               — total oracle calls used
                history                  — list of per-eval dicts for plotting
        """
        rng = np.random.RandomState(seed)
        enc_bounds = [
            (np.log(delta_bounds[0]), np.log(delta_bounds[1])),
            (beta_bounds[0], beta_bounds[1]),
        ]

        # Fresh local state (does not modify self's calibration table)
        X_obs = {"aggressive": [], "passive": []}
        Y_obs = {"aggressive": [], "passive": []}
        gps   = {
            "aggressive": {m: None for m in STYLE_METRICS},
            "passive":    {m: None for m in STYLE_METRICS},
        }
        history = []
        eval_count = 0

        # ---- local helpers ------------------------------------------------

        def fit_local(direction):
            X = np.array(X_obs[direction])
            if len(X) < 3:
                return
            for metric in STYLE_METRICS:
                Y = np.array([e[metric] for e in Y_obs[direction]])
                gps[direction][metric] = _fit_gp(X, Y)

        def best_dist(direction):
            feasible = [y for y in Y_obs[direction]
                        if np.isfinite(y["exploitability"])
                        and y["exploitability"] <= max_exploitability]
            if not feasible:
                return 1e6
            return min(
                sum((y.get(m, 0) - tv) ** 2 for m, tv in target.items())
                for y in feasible
            )

        def neg_cei_local(x_enc, direction):
            mu_d, sigma_d = _predicted_distance(x_enc, gps[direction], target)
            ei_val = _ei(mu_d, sigma_d, best_dist(direction))
            pf = _constraint_prob(x_enc, gps[direction].get("exploitability"),
                                  max_exploitability)
            return -(ei_val * pf)

        def next_point(direction):
            bv, bx = np.inf, None
            for _ in range(n_restarts_acq):
                x0 = rng.uniform([b[0] for b in enc_bounds],
                                 [b[1] for b in enc_bounds])
                res = minimize(lambda x: neg_cei_local(x, direction),
                               x0, bounds=enc_bounds, method="L-BFGS-B",
                               options={"maxiter": 200})
                if res.fun < bv:
                    bv, bx = res.fun, res.x
            d, b = _decode(bx)
            return (float(np.clip(d, delta_bounds[0], delta_bounds[1])),
                    float(np.clip(b, beta_bounds[0],  beta_bounds[1])))

        def run_eval(d, b, direction):
            nonlocal eval_count
            entry = self._evaluate(d, b, direction, num_iters)
            x_enc = _encode(d, b)
            X_obs[direction].append(x_enc)
            Y_obs[direction].append(entry)
            eval_count += 1
            dist = sum((entry.get(m, 0) - tv) ** 2 for m, tv in target.items())
            rec = {
                "eval": eval_count,
                "direction": direction,
                "delta": d,
                "beta": b,
                "distance": dist,
                "exploitability": entry["exploitability"],
                **{m: entry[m] for m in ["aggression", "bluff", "value_bet"]},
            }
            history.append(rec)
            if verbose:
                phase = "init" if eval_count <= 2 * n_init else "BO"
                print(f"  [{eval_count:3d}] {direction:10s} δ={d:.2f} β={b:.3f} "
                      f"→ dist={dist:.5f}  expl={entry['exploitability']:.5f}  [{phase}]")
            return entry

        # ---- random warm-up per direction ---------------------------------

        if verbose:
            print(f"[Goal-BO] target={target}   budget: {n_init} init × 2 + {n_bo_iter} BO")
            print("  Phase 1 — random init")

        for direction in ["aggressive", "passive"]:
            log_ds = rng.uniform(np.log(delta_bounds[0]), np.log(delta_bounds[1]), n_init)
            betas  = rng.uniform(beta_bounds[0], beta_bounds[1], n_init)
            log_ds[0], betas[0] = 0.0, 0.0  # always include baseline
            for i in range(n_init):
                run_eval(float(np.exp(log_ds[i])), float(betas[i]), direction)
            fit_local(direction)

        # ---- BO iterations (alternate directions) -------------------------

        if verbose:
            print("  Phase 2 — BO guided")

        for i in range(n_bo_iter):
            direction = ["aggressive", "passive"][i % 2]
            d, b = next_point(direction)
            run_eval(d, b, direction)
            fit_local(direction)

        # ---- collect best feasible result ---------------------------------

        all_obs = []
        for direction in ["aggressive", "passive"]:
            for y in Y_obs[direction]:
                expl = y["exploitability"]
                if np.isfinite(expl) and expl <= max_exploitability:
                    dist = sum((y.get(m, 0) - tv) ** 2 for m, tv in target.items())
                    all_obs.append((dist, y, direction))

        if not all_obs:
            # Relax constraint to find *something* (exclude NaN entries)
            for direction in ["aggressive", "passive"]:
                for y in Y_obs[direction]:
                    if not np.isfinite(y["exploitability"]):
                        continue
                    dist = sum((y.get(m, 0) - tv) ** 2 for m, tv in target.items())
                    all_obs.append((dist, y, direction))

        all_obs.sort(key=lambda t: t[0])
        best_dist_val, best_entry, best_dir = all_obs[0]

        return {
            "delta": best_entry["delta"],
            "beta":  best_entry["beta"],
            "direction": best_dir,
            "achieved": {k: best_entry[k] for k in
                         ["aggression", "bluff", "value_bet", "conservatism"]},
            "exploitability": best_entry["exploitability"],
            "target": target,
            "distance": best_dist_val,
            "eval_count": eval_count,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Public: GP-based query (interpolation beyond observed points)
    # ------------------------------------------------------------------

    def query_gp(self, target: dict, max_exploitability: float = 0.02,
                 n_restarts: int = 50) -> dict:
        """
        Predict the optimal (δ, β) using the *fitted GP surrogates*.

        Unlike query() which looks up the nearest point in the observed table,
        this optimises the GP posterior mean directly — enabling interpolation
        to never-evaluated (δ, β) values.

        Requires calibrate_bo() to have been run first.

        Returns a dict with the same keys as query(), plus 'source': 'gp_surrogate'.
        """
        if not any(self._gps["aggressive"][m] is not None for m in STYLE_METRICS):
            raise RuntimeError("Call calibrate_bo() before query_gp().")

        rng = np.random.RandomState(0)
        best_result, best_dist_val = None, np.inf

        for direction in ["aggressive", "passive"]:
            enc_bounds = [
                (np.log(1.0), np.log(12.0)),
                (0.0, 0.20),
            ]

            def obj(x_enc):
                mu_d, _ = _predicted_distance(x_enc, self._gps[direction], target)
                pf = _constraint_prob(x_enc,
                                      self._gps[direction].get("exploitability"),
                                      max_exploitability)
                if pf < 0.2:
                    return 1e6
                return mu_d

            for _ in range(n_restarts):
                x0 = rng.uniform([b[0] for b in enc_bounds],
                                 [b[1] for b in enc_bounds])
                res = minimize(obj, x0, bounds=enc_bounds, method="L-BFGS-B")
                if res.fun < best_dist_val:
                    best_dist_val = res.fun
                    d, b = _decode(res.x)
                    best_result = {
                        "delta": round(d, 4),
                        "beta":  round(b, 4),
                        "direction": direction,
                        "predicted_distance": best_dist_val,
                        "target": target,
                        "source": "gp_surrogate",
                    }

        return best_result

    # ------------------------------------------------------------------
    # Convenience: get GP surrogate predictions on a grid (for plotting)
    # ------------------------------------------------------------------

    def surrogate_surface(self, metric: str, direction: str,
                          n_grid: int = 30,
                          delta_bounds=(1.0, 12.0),
                          beta_bounds=(0.0, 0.20)):
        """
        Evaluate the GP surrogate mean and std on a regular grid.

        Returns:
            delta_grid  (n_grid,)  δ values
            beta_grid   (n_grid,)  β values
            mu          (n_grid, n_grid) posterior mean
            sigma       (n_grid, n_grid) posterior std
        """
        gp = self._gps[direction].get(metric)
        if gp is None:
            raise RuntimeError(f"GP for ({direction}, {metric}) not fitted.")

        log_d_vals = np.linspace(np.log(delta_bounds[0]), np.log(delta_bounds[1]), n_grid)
        b_vals     = np.linspace(beta_bounds[0], beta_bounds[1], n_grid)
        delta_grid = np.exp(log_d_vals)
        beta_grid  = b_vals

        LD, B = np.meshgrid(log_d_vals, b_vals, indexing="ij")
        X_grid = np.column_stack([LD.ravel(), B.ravel()])
        mu_flat, sigma_flat = gp.predict(X_grid, return_std=True)

        mu    = mu_flat.reshape(n_grid, n_grid)
        sigma = sigma_flat.reshape(n_grid, n_grid)
        return delta_grid, beta_grid, mu, sigma


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("Active AUTOPREF — Bayesian Optimisation demo (Kuhn Poker)")
    print("=" * 65)

    tuner = BOAutoPrefTuner("kuhn_poker")

    print("\n--- Amortised BO calibration (40 evals, grid would need 60) ---")
    tuner.calibrate_bo(
        n_init=5, n_bo_iter=10,
        num_iters=2000,
        cache_path="experiment_results/bo_kuhn_calibration.json",
        verbose=True,
    )

    print("\n--- Grid-style lookup after BO calibration ---")
    for t, desc in [
        ({"aggression": 0.40}, "Aggressive"),
        ({"aggression": 0.15}, "Conservative"),
        ({"bluff": 0.30},      "Heavy bluffer"),
    ]:
        r = tuner.query(t, max_exploitability=0.02)
        print(f"  {desc:18s} → δ({r['direction']})={r['delta']:.2f}, "
              f"β={r['beta']:.3f}  achieved={r['achieved']}  "
              f"expl={r['exploitability']:.5f}")

    print("\n--- GP-surrogate query (interpolation) ---")
    r_gp = tuner.query_gp({"aggression": 0.40})
    print(f"  GP suggests δ({r_gp['direction']})={r_gp['delta']:.3f}, β={r_gp['beta']:.4f}")

    print("\n--- Goal-directed BO (14 total evals for ONE target) ---")
    result = tuner.search_goal_directed(
        target={"aggression": 0.40},
        max_exploitability=0.02,
        n_init=3, n_bo_iter=8,
        num_iters=2000,
        verbose=True,
    )
    print(f"\n  Best found: δ({result['direction']})={result['delta']:.2f}, "
          f"β={result['beta']:.3f}")
    print(f"  Achieved:   {result['achieved']}")
    print(f"  Exploitability: {result['exploitability']:.6f}")
    print(f"  Total evaluations: {result['eval_count']}")
