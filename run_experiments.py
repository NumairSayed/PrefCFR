"""
Preference-CFR: Empirical Study
================================
Phase 2 experiments for the mini-project report.

Research Questions:
  RQ1: Does standard CFR always converge to the same equilibrium? (Conjecture 3.1)
  RQ2: How strongly does δ influence equilibrium selection?
  RQ3: How does empirical exploitability grow as β increases?

Also tracks action frequencies during training (RQ bonus).
"""

import pyspiel
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from open_spiel.python.algorithms import exploitability
from algorithm.CFR import CFR, PrefCFR
from algorithm.MCCFR import ES_MCCFR
from algorithm.PrefCFR import ES_MCPrefCFR
import os, json, time, csv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Utility: run CFR/PrefCFR on Kuhn poker and collect data
# ============================================================

def run_kuhn_cfr(num_iterations, pref_config=None, seed=None):
    """Run CFR or PrefCFR on Kuhn poker and return trajectory data."""
    if seed is not None:
        np.random.seed(seed)
    
    game = pyspiel.load_game("kuhn_poker")
    if pref_config is not None and len(pref_config) > 0:
        solver = PrefCFR(game, pref_config=pref_config)
    else:
        solver = CFR(game)
    
    data = {"iteration": [], "node_touched": [], "nash_conv": [], "alpha": []}
    
    checkpoints = set()
    # Log-spaced checkpoints
    for exp in np.linspace(1, np.log10(num_iterations), 80):
        checkpoints.add(int(10**exp))
    checkpoints.add(num_iterations)
    checkpoints = sorted(checkpoints)
    
    for it in range(1, num_iterations + 1):
        solver.iteration()
        if it in checkpoints:
            conv = exploitability.nash_conv(game, solver.average_policy())
            alpha = solver.get_policy("0")  # sigma(J_, Bet)
            data["iteration"].append(it)
            data["node_touched"].append(solver.node_touched)
            data["nash_conv"].append(conv)
            data["alpha"].append(alpha)
    
    return data


def run_leduc_mccfr(num_node_target, pref_config=None, seed=None):
    """Run ES-MCCFR or ES-MCPrefCFR on Leduc poker."""
    if seed is not None:
        np.random.seed(seed)
    
    game = pyspiel.load_game("leduc_poker")
    if pref_config is not None and len(pref_config) > 0:
        solver = ES_MCPrefCFR(game, pref_config=pref_config)
    else:
        solver = ES_MCCFR(game)
    
    info_key = "[Observer: 0][Private: 5][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]"
    
    data = {"node_touched": [], "nash_conv": [], "call_prob": []}
    
    print_node = 1000
    while solver.node_touched < num_node_target:
        solver.iteration()
        if solver.node_touched >= print_node:
            print_node *= 1.5
            try:
                conv = exploitability.nash_conv(game, solver.average_policy())
                call_prob = solver.get_policy(info_key)
            except:
                continue
            data["node_touched"].append(solver.node_touched)
            data["nash_conv"].append(conv)
            data["call_prob"].append(call_prob)
    
    return data


# ============================================================
# RQ1: Does CFR always converge to the same α? (Conjecture 3.1)
# ============================================================

def experiment_rq1(num_trials=30, num_iterations=5000):
    """Run standard CFR from random initializations, track α convergence."""
    print("=" * 60)
    print("RQ1: Testing Conjecture 3.1 - CFR convergence uniqueness")
    print(f"     {num_trials} trials, {num_iterations} iterations each")
    print("=" * 60)
    
    all_data = []
    final_alphas = []
    
    for trial in range(num_trials):
        seed = trial * 137 + 42
        data = run_kuhn_cfr(num_iterations, pref_config=None, seed=seed)
        all_data.append(data)
        final_alphas.append(data["alpha"][-1])
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{num_trials}: final α = {data['alpha'][-1]:.4f}, NashConv = {data['nash_conv'][-1]:.6f}")
    
    # Statistics
    mean_alpha = np.mean(final_alphas)
    std_alpha = np.std(final_alphas)
    min_alpha = np.min(final_alphas)
    max_alpha = np.max(final_alphas)
    
    print(f"\n  Results: α = {mean_alpha:.4f} ± {std_alpha:.4f} (range: [{min_alpha:.4f}, {max_alpha:.4f}])")
    print(f"  Paper claims α → 0.2 regardless of initialization")
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Convergence of NashConv
    for i, data in enumerate(all_data):
        x = np.log10(data["node_touched"])
        y = np.log10(data["nash_conv"])
        axes[0].plot(x, y, alpha=0.3, color='steelblue', linewidth=0.8)
    axes[0].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[0].set_ylabel("log10(Exploitability)", fontsize=12)
    axes[0].set_title("RQ1: CFR Convergence Rate (Kuhn Poker)", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Right: α trajectory
    for i, data in enumerate(all_data):
        x = np.log10(data["node_touched"])
        axes[1].plot(x, data["alpha"], alpha=0.3, color='steelblue', linewidth=0.8)
    axes[1].axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='α = 0.2 (paper)')
    axes[1].axhline(y=mean_alpha, color='green', linestyle=':', linewidth=2, label=f'α = {mean_alpha:.3f} (empirical)')
    axes[1].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[1].set_ylabel("α = σ̄(J_, Bet)", fontsize=12)
    axes[1].set_title("RQ1: α Convergence Across Random Inits", fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rq1_conjecture_verification.png", dpi=200)
    plt.close()
    print(f"  Plot saved to {RESULTS_DIR}/rq1_conjecture_verification.png")
    
    return {
        "mean_alpha": mean_alpha,
        "std_alpha": std_alpha,
        "min_alpha": min_alpha,
        "max_alpha": max_alpha,
        "final_alphas": final_alphas,
        "all_data": all_data
    }


# ============================================================
# RQ2: How strongly does δ influence equilibrium selection?
# ============================================================

def experiment_rq2(num_trials=10, num_iterations=5000):
    """Test Pref-CFR with different δ values on Kuhn poker."""
    print("\n" + "=" * 60)
    print("RQ2: Effect of δ on equilibrium selection")
    print(f"     {num_trials} trials per setting, {num_iterations} iterations each")
    print("=" * 60)
    
    # Settings matching the paper's Appendix C
    settings = {
        "CFR (baseline)": None,
        "Pref-CFR(BR) δ(Bet)=5": {
            "0": [np.array([1, 5]), 0],
            "1": [np.array([1, 5]), 0],
            "2": [np.array([1, 5]), 0],
        },
        "Pref-CFR(BR) δ(Bet)=10": {
            "0": [np.array([1, 10]), 0],
            "1": [np.array([1, 10]), 0],
            "2": [np.array([1, 10]), 0],
        },
        "Pref-CFR(BR) δ(Pass)=5": {
            "0": [np.array([5, 1]), 0],
            "1": [np.array([5, 1]), 0],
            "2": [np.array([5, 1]), 0],
        },
        "Pref-CFR(BR) δ(Pass)=10": {
            "0": [np.array([10, 1]), 0],
            "1": [np.array([10, 1]), 0],
            "2": [np.array([10, 1]), 0],
        },
    }
    
    colors = ['steelblue', 'orangered', 'darkred', 'forestgreen', 'darkgreen']
    results = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (name, pref_cfg) in enumerate(settings.items()):
        print(f"\n  Setting: {name}")
        all_data = []
        final_alphas = []
        
        for trial in range(num_trials):
            seed = trial * 137 + idx * 1000 + 42
            data = run_kuhn_cfr(num_iterations, pref_config=pref_cfg, seed=seed)
            all_data.append(data)
            final_alphas.append(data["alpha"][-1])
        
        mean_alpha = np.mean(final_alphas)
        std_alpha = np.std(final_alphas)
        print(f"    → α = {mean_alpha:.4f} ± {std_alpha:.4f}")
        
        results[name] = {
            "mean_alpha": mean_alpha,
            "std_alpha": std_alpha,
            "final_alphas": final_alphas,
        }
        
        # Plot convergence
        for data in all_data:
            x = np.log10(data["node_touched"])
            axes[0].plot(x, np.log10(data["nash_conv"]), alpha=0.15, color=colors[idx], linewidth=0.5)
        
        # Plot mean α
        mean_alpha_traj = np.mean([d["alpha"] for d in all_data], axis=0)
        x_mean = np.log10(all_data[0]["node_touched"])
        axes[1].plot(x_mean, mean_alpha_traj, color=colors[idx], linewidth=2, label=f"{name} (α→{mean_alpha:.3f})")
        
        # Confidence band
        all_alpha_arr = np.array([d["alpha"] for d in all_data])
        alpha_low = np.percentile(all_alpha_arr, 5, axis=0)
        alpha_high = np.percentile(all_alpha_arr, 95, axis=0)
        axes[1].fill_between(x_mean, alpha_low, alpha_high, color=colors[idx], alpha=0.1)
    
    axes[0].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[0].set_ylabel("log10(Exploitability)", fontsize=12)
    axes[0].set_title("RQ2: Convergence Rate Comparison", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[1].set_ylabel("α = σ̄(J_, Bet)", fontsize=12)
    axes[1].set_title("RQ2: δ Influence on Equilibrium Selection", fontsize=13)
    axes[1].legend(fontsize=8, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rq2_delta_influence.png", dpi=200)
    plt.close()
    print(f"\n  Plot saved to {RESULTS_DIR}/rq2_delta_influence.png")
    
    return results


# ============================================================
# RQ3: How does exploitability grow as β increases? (Leduc)
# ============================================================

def experiment_rq3(num_trials=5, num_nodes=500000):
    """Test Vulnerability CFR with different β values on Leduc poker."""
    print("\n" + "=" * 60)
    print("RQ3: Exploitability vs β (Leduc Poker)")
    print(f"     {num_trials} trials per setting, {num_nodes} nodes target")
    print("=" * 60)
    
    leduc_info_keys = [
        f'[Observer: 0][Private: {p}][Round 1][Player: 0][Pot: 2][Money: 99 99][Round1: ][Round2: ]'
        for p in range(6)
    ]
    
    beta_values = [0.0, 0.05, 0.10, 0.20, 0.50]
    colors_beta = ['steelblue', 'orange', 'orangered', 'red', 'darkred']
    
    results = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, beta in enumerate(beta_values):
        name = f"δ(raise)=5, β={beta}"
        print(f"\n  Setting: {name}")
        
        pref_cfg = {}
        if beta > 0 or True:  # always set config for raise preference
            for key in leduc_info_keys:
                pref_cfg[key] = [np.array([1, 5]), beta]
        
        if beta == 0.0 and idx == 0:
            # baseline: no pref config
            pref_cfg = {}
        
        all_data = []
        final_exploitabilities = []
        final_call_probs = []
        
        for trial in range(num_trials):
            seed = trial * 137 + idx * 1000 + 99
            data = run_leduc_mccfr(num_nodes, pref_config=pref_cfg, seed=seed)
            all_data.append(data)
            if len(data["nash_conv"]) > 0:
                final_exploitabilities.append(data["nash_conv"][-1])
                final_call_probs.append(data["call_prob"][-1])
            print(f"    Trial {trial+1}: NashConv = {data['nash_conv'][-1]:.4f}, Call prob = {data['call_prob'][-1]:.4f}" if len(data['nash_conv']) > 0 else "    Trial failed")
        
        mean_exploit = np.mean(final_exploitabilities) if final_exploitabilities else 0
        mean_call = np.mean(final_call_probs) if final_call_probs else 0
        
        results[name] = {
            "beta": beta,
            "mean_exploitability": mean_exploit,
            "mean_call_prob": mean_call,
        }
        
        print(f"    → Mean exploitability: {mean_exploit:.4f}, Mean call prob: {mean_call:.4f}")
        
        # Plot
        for data in all_data:
            if len(data["node_touched"]) > 0:
                x = np.log10(data["node_touched"])
                axes[0].plot(x, np.log10(data["nash_conv"]), alpha=0.3, color=colors_beta[idx], linewidth=0.8)
                axes[1].plot(x, data["call_prob"], alpha=0.3, color=colors_beta[idx], linewidth=0.8)
        
        # Mean line for call prob
        if all_data and len(all_data[0]["node_touched"]) > 0:
            min_len = min(len(d["call_prob"]) for d in all_data)
            if min_len > 0:
                mean_call_traj = np.mean([d["call_prob"][:min_len] for d in all_data], axis=0)
                x_mean = np.log10(all_data[0]["node_touched"][:min_len])
                axes[1].plot(x_mean, mean_call_traj, color=colors_beta[idx], linewidth=2.5,
                            label=f"β={beta} (call→{mean_call:.2f})")
    
    axes[0].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[0].set_ylabel("log10(Exploitability)", fontsize=12)
    axes[0].set_title("RQ3: Exploitability vs β (Leduc Poker)", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("log10(Nodes touched)", fontsize=12)
    axes[1].set_ylabel("Pr[Call] at info set", fontsize=12)
    axes[1].set_title("RQ3: Call Probability vs β", fontsize=13)
    axes[1].legend(fontsize=9, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rq3_beta_exploitability.png", dpi=200)
    plt.close()
    print(f"\n  Plot saved to {RESULTS_DIR}/rq3_beta_exploitability.png")
    
    # Summary bar chart: β vs final exploitability
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4))
    betas = [r["beta"] for r in results.values()]
    exploits = [r["mean_exploitability"] for r in results.values()]
    calls = [r["mean_call_prob"] for r in results.values()]
    
    ax2_twin = ax2.twinx()
    bars = ax2.bar(range(len(betas)), exploits, color='steelblue', alpha=0.7, label='Exploitability')
    ax2_twin.plot(range(len(betas)), calls, 'ro-', linewidth=2, markersize=8, label='Call Probability')
    
    ax2.set_xticks(range(len(betas)))
    ax2.set_xticklabels([f"β={b}" for b in betas], fontsize=10)
    ax2.set_ylabel("Final Exploitability", fontsize=12, color='steelblue')
    ax2_twin.set_ylabel("Final Call Probability", fontsize=12, color='red')
    ax2.set_title("RQ3: Trade-off Between β, Exploitability, and Style", fontsize=13)
    
    ax2.legend(loc='upper left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/rq3_tradeoff_summary.png", dpi=200)
    plt.close()
    print(f"  Summary plot saved to {RESULTS_DIR}/rq3_tradeoff_summary.png")
    
    return results


# ============================================================
# RQ Bonus: Action frequency tracking during training
# ============================================================

def experiment_action_frequencies(num_iterations=5000):
    """Track how action frequencies evolve during Pref-CFR training."""
    print("\n" + "=" * 60)
    print("BONUS: Action frequency tracking during training")
    print("=" * 60)
    
    settings = {
        "CFR (baseline)": None,
        "Aggressive (δ(Bet)=5)": {
            "0": [np.array([1, 5]), 0],
            "1": [np.array([1, 5]), 0],
            "2": [np.array([1, 5]), 0],
        },
        "Defensive (δ(Pass)=5)": {
            "0": [np.array([5, 1]), 0],
            "1": [np.array([5, 1]), 0],
            "2": [np.array([5, 1]), 0],
        },
    }
    
    colors = ['steelblue', 'orangered', 'forestgreen']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # For each card (J=0, Q=1, K=2), track Bet probability
    card_names = ["J (info '0')", "Q (info '1')", "K (info '2')"]
    info_keys = ["0", "1", "2"]
    
    for idx, (name, pref_cfg) in enumerate(settings.items()):
        print(f"  Running: {name}")
        game = pyspiel.load_game("kuhn_poker")
        if pref_cfg:
            solver = PrefCFR(game, pref_config=pref_cfg)
        else:
            solver = CFR(game)
        
        iters = []
        bet_probs = {k: [] for k in info_keys}
        
        checkpoints = sorted(set([int(10**e) for e in np.linspace(1, np.log10(num_iterations), 60)] + [num_iterations]))
        
        for it in range(1, num_iterations + 1):
            solver.iteration()
            if it in checkpoints:
                iters.append(it)
                for k in info_keys:
                    bet_probs[k].append(solver.get_policy(k))
        
        for card_idx, key in enumerate(info_keys):
            axes[card_idx].plot(np.log10(iters), bet_probs[key], 
                              color=colors[idx], linewidth=2, label=name)
    
    for card_idx in range(3):
        axes[card_idx].set_xlabel("log10(Iterations)", fontsize=12)
        axes[card_idx].set_ylabel("Pr[Bet]", fontsize=12)
        axes[card_idx].set_title(f"Card: {card_names[card_idx]}", fontsize=13)
        axes[card_idx].legend(fontsize=8)
        axes[card_idx].grid(True, alpha=0.3)
        axes[card_idx].set_ylim(-0.05, 1.05)
    
    plt.suptitle("Action Frequencies During Training (Kuhn Poker)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/bonus_action_frequencies.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved to {RESULTS_DIR}/bonus_action_frequencies.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start = time.time()
    
    print("\n" + "#" * 60)
    print("# Preference-CFR Empirical Study")
    print("# Running all experiments...")
    print("#" * 60)
    
    # RQ1
    rq1_results = experiment_rq1(num_trials=30, num_iterations=5000)
    
    # RQ2
    rq2_results = experiment_rq2(num_trials=10, num_iterations=5000)
    
    # RQ3
    rq3_results = experiment_rq3(num_trials=5, num_nodes=500000)
    
    # Bonus
    experiment_action_frequencies(num_iterations=5000)
    
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"All experiments completed in {elapsed:.1f} seconds")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"{'=' * 60}")
    
    # Save summary
    summary = {
        "rq1": {
            "mean_alpha": rq1_results["mean_alpha"],
            "std_alpha": rq1_results["std_alpha"],
            "conclusion": "CFR converges to unique α" if rq1_results["std_alpha"] < 0.02 else "Some variance in α"
        },
        "rq2": {k: {"mean_alpha": v["mean_alpha"], "std_alpha": v["std_alpha"]} for k, v in rq2_results.items()},
        "rq3": {k: v for k, v in rq3_results.items()},
        "elapsed_seconds": elapsed
    }
    
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to {RESULTS_DIR}/summary.json")
