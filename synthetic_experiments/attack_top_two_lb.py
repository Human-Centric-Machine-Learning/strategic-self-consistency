import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import hashlib
import pandas as pd
from matplotlib.patches import Patch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils

LOWER_BOUND_CSV = "../results/attack_lower_bound.csv"
MAX_CLASSES = 6
PROB_COLS = [f"p{i}" for i in range(1, MAX_CLASSES + 1)]


def pad_probs(probs, max_classes=MAX_CLASSES):
    """
    Pad a probability vector with zeros up to max_classes.
    """
    probs = np.asarray(probs, dtype=float)

    if len(probs) > max_classes:
        raise ValueError(f"At most {max_classes} response classes are supported.")

    padded = np.zeros(max_classes)
    padded[:len(probs)] = probs
    return padded

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{textcomp}
    """,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,

    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    "axes.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.edgecolor": "black",

    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
})



def load_attack_lower_bound_results(csv_path=LOWER_BOUND_CSV):
    """Load cached simulation results."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return pd.read_csv(csv_path)

    return pd.DataFrame(columns=["method", "K", *PROB_COLS, "n_trials", "T_max", "asc_confidence",
        "delta", "epsilon", "seed", "include_final_wasteful", "d_hat",
        "lower_bound", "empirical_mean", "empirical_sem", "n_valid",
        "n_hit_cap", "tilde_gap"])

def result_already_exists(df, method, probs, n_trials, T_max, asc_confidence, delta, epsilon,seed,include_final_wasteful):
    if df.empty:
        return False

    p = pad_probs(probs)

    mask = (
        (df["method"] == method)
        & (df["n_trials"] == n_trials)
        & (df["T_max"] == T_max)
        & np.isclose(df["asc_confidence"], asc_confidence)
        & np.isclose(df["delta"], delta)
        & np.isclose(df["epsilon"], epsilon)
        & (df["seed"] == seed)
        & (df["include_final_wasteful"] == include_final_wasteful)
    )

    for i, col in enumerate(PROB_COLS):
        mask &= np.isclose(df[col], p[i])

    return mask.any()

def append_attack_result(row, csv_path=LOWER_BOUND_CSV):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def probability_seed(probs, base_seed=0):
    """
    Give each probability configuration a deterministic RNG seed.

    This means p=[p1, p2, ..., pK] gets the same sample reservoir
    every time the script is run.
    """
    key = ",".join(f"{p:.12f}" for p in probs)
    digest = hashlib.sha256(key.encode()).hexdigest()
    offset = int(digest[:8], 16)
    return (base_seed + offset) % (2**32)


def _continuation_lower_bound(d, p1, p2):
    num = (p1 * p2 * (p1 ** d - p2 ** d)) - (d * (p1 - p2) * p2 ** (d+1))
    denom = (p1 - p2) * (p1 ** (d + 2) - p2 ** (d + 2))
    return num / denom

def make_attack_lower_bound_figure(df, save_dir="figures"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    methods = ["PPR-1v1", "ASC"]

    df = df.copy()
    df["prob_gap"] = df["p1"] - df["p2"]

    Ks = [2]#sorted(df["K"].astype(int).unique())
    colors = plt.get_cmap("tab10").colors
    k_colors = {K: colors[i] for i, K in enumerate(Ks)}

    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method].copy()

        for K in Ks:
            sub_k = sub[sub["K"].astype(int) == K].sort_values("prob_gap")
            color = k_colors[K]
            x = sub_k["prob_gap"].to_numpy()
            y_emp = sub_k["empirical_mean"].to_numpy()
            y_sem = sub_k["empirical_sem"].to_numpy()
            ax.fill_between(x, y_emp - 2 * y_sem, y_emp + 2 * y_sem, color=color, alpha=0.20, linewidth=0, zorder=1)

        # Draw curves on top
        for K in Ks:
            sub_k =  sub[sub["K"].astype(int) == K].sort_values("prob_gap")
            color = k_colors[K]
            x = sub_k["prob_gap"].to_numpy()
            y_emp = sub_k["empirical_mean"].to_numpy()
            y_lb = sub_k["lower_bound"].to_numpy()

            # Empirical
            ax.plot(x, y_emp, color=color, lw=1.2, alpha=0.9, zorder=3, ls=':')

            # Lower bound
            ax.plot(x, y_lb, color=color,lw=1.3, ls='-', alpha=0.85, zorder=2)

        x = sub_k["prob_gap"].to_numpy()
        y_emp = sub_k["empirical_mean"].to_numpy()
        y_sem = sub_k["empirical_sem"].to_numpy()
        ax.fill_between(x, y_emp - y_sem, y_emp + y_sem, color=color, alpha=0.12, linewidth=1.2, zorder=1)
       
        if method == "ASC":
            method = "(b) ASC stopping rule"
        else:
            method = "(a) PPR-1v1 stopping rule"
        ax.set_title(method, fontsize=14, y=-0.3)
        ax.set_xlabel(r"$p_{(1)}-p_{(2)}$", fontsize=14)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.3")
        ax.set_ylim(0, 50)

    axes[0].set_ylabel(r"$\mathbb{E}[N'-N]$",fontsize=14)

    # Shared legend: class count color (if plotting multiple K)
    # class_handles = [ Patch(facecolor=k_colors[K], edgecolor="none", label=fr"$K={K}$") for K in Ks]
    # fig.legend(handles=class_handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol= len(class_handles), fontsize=12, framealpha=0.0)

    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "attack_lower_bound.pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_attack_lower_bound(prob_configs, n_trials=1000,T_max=4096,asc_confidence=0.95,delta=0.1,epsilon=0.0,seed=0,include_final_wasteful=True,csv_path=LOWER_BOUND_CSV,save_dir="../figures"):
    prob_configs = [np.asarray(p, dtype=float) for p in prob_configs]

    methods = [
        ("ASC", lambda K: utils.make_asc_stop_rule(confidence=asc_confidence)),
        ("PPR-1v1", lambda K: utils.make_ppr_stop_rule(K=K, delta=delta, epsilon=epsilon)),
    ]

    cached = load_attack_lower_bound_results(csv_path)

    # Compute only missing probability/method combinations
    for probs in prob_configs:
        probs = probs / probs.sum()
        K = len(probs)

        # Make this distribution reproducible across script runs
        rng = np.random.default_rng(probability_seed(probs, seed))

        trial_samples = [rng.choice(np.arange(K), size=T_max, p=probs) for _ in range(n_trials)]

        for method_name, make_rule in methods:
            if result_already_exists(cached, method=method_name, probs=probs, n_trials=n_trials, T_max=T_max,
                asc_confidence=asc_confidence,  delta=delta, epsilon=epsilon, seed=seed, include_final_wasteful=include_final_wasteful):
                continue

            print(f"Running: {method_name}, p={probs}")

            stop_rule = make_rule(K)
            ds = []
            realized_extra = []
            n_hit_cap = 0

            for samples in trial_samples:
                out = utils.simulate_provider(samples,T_max=T_max, alpha=None, stop_rule=stop_rule, rng=rng)

                if not out["faithful_stopped"]:
                    continue
                n = out["faithful_n"]

                # Counts at faithful stopping.
                faithful_counts = np.bincount(samples[:n], minlength=K)

                leader_idx = int(np.argmax(faithful_counts))
                prefix_counts = faithful_counts.copy()
                prefix_counts[leader_idx] -= 1

                ds.append(out["faithful_d"])
                extra = out["adversarial_n"] - out["faithful_n"]
            
                if include_final_wasteful and not out["hit_cap"]:
                    extra += 1

                realized_extra.append(extra)

                if out["hit_cap"]:
                    n_hit_cap += 1

            if len(ds) == 0:
                print("No faithful stopping events for " + f"{method_name}, p={probs}")
                continue

            ds = np.asarray(ds, dtype=float)
            realized_extra = np.asarray(realized_extra, dtype=float)

            d_hat = ds.mean()
            lower_bounds = [ _continuation_lower_bound(d=d-1, p1=probs[0], p2=probs[1]) for d in ds]
            lower_bound = np.mean(lower_bounds)
            empirical_mean = realized_extra.mean()

            if len(realized_extra) > 1:
                empirical_sem = realized_extra.std(ddof=1) / np.sqrt(len(realized_extra))
            else:
                empirical_sem = 0.0

            p1_tilde = probs[0] / (probs[0] + probs[1])
            p2_tilde = probs[1] / (probs[0] + probs[1])
            
            # Pad probabilities to 10 columns.
            p = pad_probs(probs)

            row = {
                "method": method_name,
                "K": int(np.sum(p > 1e-12)),
                **{f"p{i + 1}": p[i] for i in range(MAX_CLASSES)},
                "n_trials": n_trials,
                "T_max": T_max,
                "asc_confidence": asc_confidence,
                "delta": delta,
                "epsilon": epsilon,
                "seed": seed,
                "include_final_wasteful": include_final_wasteful,
                "d_hat": d_hat,
                "lower_bound": lower_bound,
                "empirical_mean": empirical_mean,
                "empirical_sem": empirical_sem,
                "n_valid": len(ds),
                "n_hit_cap": n_hit_cap,
                "tilde_gap": p1_tilde - p2_tilde,
            }
            append_attack_result(row, csv_path=csv_path)
            cached = pd.concat([cached, pd.DataFrame([row])], ignore_index=True)
            print(f"Saved: {method_name}, p={probs} | d={d_hat:.3f}, LB={lower_bound:.3f}, actual={empirical_mean:.3f}")

    # Reload CSV and select exactly the experiments to plot   
    df = load_attack_lower_bound_results(csv_path)

    df = df[
        (df["n_trials"] == n_trials)
        & (df["T_max"] == T_max)
        & np.isclose(
            df["asc_confidence"],
            asc_confidence,
        )
        & np.isclose(df["delta"], delta)
        & np.isclose(df["epsilon"], epsilon)
        & (df["seed"] == seed)
        & (
            df["include_final_wasteful"]
            == include_final_wasteful
        )
    ].copy()

    wanted = {tuple(np.round(pad_probs(p / np.sum(p)), 12)) for p in prob_configs}
    df = df[df.apply(lambda row: tuple(np.round([row[col] for col in PROB_COLS],12)) in wanted, axis=1)]
    make_attack_lower_bound_figure(df,save_dir=save_dir)
    return df


def make_prob_configs(min_gap=0.1, max_gap=0.90,n_gaps=20, Ks=range(2, 7),):
    """
    Generate probability configurations for K=2,...,6 over a range of top-two probability gaps.
    """

    gaps = np.linspace(min_gap, max_gap, n_gaps)
    prob_configs = []

    for K in Ks:
        for gap in gaps:
            p_other = (1.0 - gap) / K
            p1 = p_other + gap
            probs = np.array([p1] + [p_other] * (K - 1), dtype=float)
            prob_configs.append(probs)

    return prob_configs


if __name__ == "__main__":
    prob_configs = make_prob_configs()
    plot_attack_lower_bound(prob_configs)


 