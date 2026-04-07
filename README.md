# Preference-CFR: Empirical Study

## Setup
```bash
pip install -r requirements.txt
```

## Run All Experiments
```bash
python run_experiments.py
```

This runs 4 experiments and saves plots to `experiment_results/`:

| File | What it tests |
|------|---------------|
| `rq1_conjecture_verification.png` | RQ1: Does CFR always converge to α=0.2? (30 trials) |
| `rq2_delta_influence.png` | RQ2: How δ shifts equilibrium selection (5 settings × 10 trials) |
| `rq3_beta_exploitability.png` | RQ3: Exploitability vs β in Leduc poker (5 β values × 5 trials) |
| `rq3_tradeoff_summary.png` | RQ3: Bar chart of β vs exploitability trade-off |
| `bonus_action_frequencies.png` | Per-card Bet probability during training |
| `summary.json` | Numerical results for the report |

## Run Original Paper Code (Kuhn)
```bash
python PrefCFRMain.py
```

## Project Structure
```
├── algorithm/
│   ├── CFR.py          # Vanilla CFR + PrefCFR for Kuhn
│   ├── MCCFR.py        # External Sampling MCCFR
│   ├── MCCFVFP.py      # MCCFVFP variant
│   └── PrefCFR.py      # ES-MCPrefCFR for Leduc
├── draw/
│   ├── convergence_rate.py
│   └── draw_martix.py
├── Game_config.py       # All δ, β configs
├── PrefCFRMain.py       # Original paper's main script
├── run_experiments.py   # ← YOUR EXPERIMENT SCRIPT (run this)
└── requirements.txt
```

## Timing
- RQ1 (Kuhn, 30 trials): ~80 seconds
- RQ2 (Kuhn, 50 trials): ~150 seconds
- RQ3 (Leduc, 25 trials): ~10-15 minutes
- Bonus: ~30 seconds
- **Total: ~15-20 minutes**

## Tuning
If RQ3 is too slow, reduce in `run_experiments.py`:
```python
rq3_results = experiment_rq3(num_trials=3, num_nodes=200000)  # faster
```
