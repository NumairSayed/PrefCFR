# Preference-CFR: Empirical Study + AutoPref

This repo accompanies two reports:

1. **Main report** — an empirical study of Preference-CFR (Pref-CFR), covering how the parameters δ and β affect playing style and exploitability in Kuhn and Leduc poker.
2. **Active AutoPref report** — an extension that replaces the AutoPref calibration grid with a Bayesian Optimisation loop, cutting calibration cost from 50 to 18 solver runs (2.8× speedup).

---

## Setup

```bash
pip install -r requirements.txt
pip install scikit-learn   # needed for Active AutoPref only
```

The experiments use [OpenSpiel](https://github.com/google-deepmind/open_spiel). Make sure it is installed and importable (`import pyspiel` should work).

---

## Report 1: Reproducing the Main Empirical Study

Run all experiments (RQ1, RQ2, RQ3) with one command:

```bash
python run_experiments.py
```

This takes **15–20 minutes** and saves all figures to `experiment_results/`:

| Output file | What it shows |
|---|---|
| `rq1_conjecture_verification.png` | RQ1: Does vanilla CFR always converge to α = 1/3 in Kuhn poker? (30 trials) |
| `rq2_delta_influence.png` | RQ2: How δ smoothly shifts the equilibrium selected (5 δ values × 10 trials) |
| `rq3_beta_exploitability.png` | RQ3: Exploitability vs β in Leduc poker (5 β values × 5 trials) |
| `rq3_tradeoff_summary.png` | RQ3: Bar chart of the style–exploitability trade-off |
| `bonus_action_frequencies.png` | Per-card bet probability over training |
| `summary.json` | All numerical results |

**Timing per experiment:**
- RQ1 (Kuhn, 30 trials): ~80 s
- RQ2 (Kuhn, 50 trials): ~150 s
- RQ3 (Leduc, 25 trials): ~10–15 min
- Bonus: ~30 s

If RQ3 is too slow, open `run_experiments.py` and reduce the trial count:
```python
rq3_results = experiment_rq3(num_trials=3, num_nodes=200000)
```

To run the original paper's Kuhn poker script directly:
```bash
python PrefCFRMain.py
```

---

## Report 2: Reproducing Active AutoPref (Bayesian Optimisation)

Active AutoPref is implemented in `active_autopref.py` (the `BOAutoPrefTuner` class). The full comparison experiment is in `run_bo_comparison.py`.

```bash
python run_bo_comparison.py
```

This takes **~10 minutes** and produces four figures in `experiment_results/`:

| Output file | What it shows |
|---|---|
| `bo_fig_b1_efficiency.png` | Match quality vs. number of evaluations: BO vs. grid |
| `bo_fig_b2_surrogate.png` | GP surrogate surface vs. ground truth after 18 evaluations |
| `bo_fig_b3_summary.png` | Side-by-side bar chart: match quality and evaluation count |
| `bo_fig_b4_bo_convergence.png` | Which (δ, β) points BO chose at each step, per target |

**To compile the Active AutoPref report PDF:**
```bash
pdflatex active_autopref_report.tex && pdflatex active_autopref_report.tex
```
(Run twice to resolve cross-references.)

**Using AutoPref or Active AutoPref in your own code:**

```python
# Classic grid-based AutoPref
from auto_pref import AutoPrefTuner
tuner = AutoPrefTuner("kuhn_poker")
tuner.calibrate(num_iters=2500)
result = tuner.query({"aggression": 0.40}, max_exploitability=0.02)

# Bayesian Optimisation version (fewer runs)
from active_autopref import BOAutoPrefTuner
tuner = BOAutoPrefTuner("kuhn_poker")
tuner.calibrate_bo(target={"aggression": 0.40}, n_init=4, n_bo=10)
result = tuner.query_gp({"aggression": 0.40})
```

---

## Project Structure

```
├── algorithm/
│   ├── CFR.py              # Vanilla CFR + PrefCFR (Kuhn poker)
│   ├── MCCFR.py            # External Sampling MCCFR
│   ├── MCCFVFP.py          # MCCFVFP variant
│   └── PrefCFR.py          # ES-MCPrefCFR (Leduc poker)
├── experiment_results/     # All generated figures go here
├── active_autopref.py      # BOAutoPrefTuner: BO-based calibration
├── auto_pref.py            # AutoPrefTuner: grid-based calibration
├── active_autopref_report.tex  # LaTeX source for Active AutoPref report
├── Game_config.py          # δ, β parameter configurations
├── PrefCFRMain.py          # Original paper's main script
├── run_bo_comparison.py    # Reproduces all Active AutoPref figures
├── run_experiments.py      # Reproduces all main report figures
└── requirements.txt
```
