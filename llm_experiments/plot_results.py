import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

PRICE_PER_M_OUTPUT_TOKEN = {
    "Llama-3.2-3B": 0.1,
    "Qwen2.5-7B": 0.2,
    "reason-R1-D-Qwen-7B": 0.2,
}

mpl.rcdefaults()
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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",

    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
})


def _difficulty_spec(method):
    """
    Return per-query difficulty column and colorbar label.
    """
    if method == "threshold_bon":
        return ("audit_trigger_prob", r"Mean trigger probability $\hat q$")

    return ("gap", r"Mean top-two gap ($\hat p_{(1)}-\hat p_{(2)}$)")


def _display_model_name(model):
    if "reason" in model:
        parts = ["DeepSeek-R1-Distill"] + model.split("-")[-2:]
        return "-".join(parts)
    return model


def bottom_of_top_fraction(x, fraction):
    """
    Return the smallest observed value among the top `fraction`
    of observations. Unlike .quantile(), this never interpolates.
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    x = np.sort(x.to_numpy())
    k = int(np.ceil(fraction * len(x)))

    return x[-k]


def bottom_of_top_10(x):
    return bottom_of_top_fraction(x, 0.10)

def plot_by_query_difficulty(df, datasets, method, model_temps, alpha):
    """
    Plot per-query audit-safe excess samples against faithful stopping time for multiple models and datasets.
    """
    df = df.copy()
    n_rows = len(datasets)
    n_models = len(model_temps)

    fig = plt.figure(figsize=(12.5, 3 * n_rows))

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_models + 1,
        width_ratios=[1] * n_models + [0.07],
        wspace=0.15,
        hspace=0.15,
    )

    axes = [[fig.add_subplot(gs[row, col]) for col in range(n_models)] for row in range(n_rows)]
    cax = fig.add_subplot(gs[:, n_models])
    scatter = None

    for row, dataset in enumerate(datasets):
        for i, (ax, (model, temperature, t_max)) in enumerate(zip(axes[row], model_temps)):

            s = df[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & np.isclose(df["temperature"], temperature)
                & np.isclose(df["alpha"], alpha)
            ].copy()

            if len(s) == 0:
                print(f"No observations found for dataset={dataset}, model={model}, temperature={temperature}, alpha={alpha}")
                continue

            if method == "threshold_bon":
                color_values = s["audit_trigger_prob"]
                colorbar_label = r"Trigger probability $\hat q$"
            else:
                color_values = s["gap"]
                colorbar_label = (r"$\hat p_{(1)}-\hat p_{(2)}$")

            scatter = ax.scatter(
                s["faithful_stop"],
                s["extra_samples_audit"],
                c=color_values,
                cmap="viridis",
                alpha=0.5,
                s=30,
                vmin=0,
                vmax=1,
                edgecolors="none",
            )

            # Distribution summaries
            extra_paths = pd.to_numeric(s["extra_samples_audit"], errors="coerce")
            extra_tokens = pd.to_numeric(s["extra_output_tokens_audit"], errors="coerce")

            valid = (
                extra_paths.notna()
                & extra_tokens.notna()
                & np.isfinite(extra_paths)
                & np.isfinite(extra_tokens)
            )

            extra_paths = extra_paths.loc[valid]
            extra_tokens = extra_tokens.loc[valid]

            if len(extra_paths) > 0:
                # Smallest observed value among the top 25% / top 10%
                paths_top25 = bottom_of_top_fraction(extra_paths, 0.25)
                paths_top10 = bottom_of_top_fraction(extra_paths, 0.10)

                paths_tail_top25 = (extra_paths >= paths_top25).mean()
                paths_tail_top10 = (extra_paths >= paths_top10).mean()

                tokens_mean = extra_tokens.mean()
                tokens_median = extra_tokens.median()

                # Smallest observed value among the top 25% / top 10%
                tokens_top25 = bottom_of_top_fraction(extra_tokens, 0.25)
                tokens_top10 = bottom_of_top_fraction(extra_tokens, 0.10)

                tokens_tail_top25 = (extra_tokens >= tokens_top25).mean()
                tokens_tail_top10 = (extra_tokens >= tokens_top10).mean()
    
                print(f"\n{dataset} | {model} | T={temperature} | alpha={alpha} | n={len(extra_paths)}")
                print(f"  Top 25% threshold: {paths_top25:.1f} (P[N'-N >= {paths_top25:.1f}] = {paths_tail_top25:.1%})")
                print(f"  Top 10% threshold: {paths_top10:.1f} (P[N'-N >= {paths_top10:.1f}] = {paths_tail_top10:.1%})")

                print("Token-level overcharge:")
                print(f"  mean: {tokens_mean:.1f} tokens")
                print(f"  median: {tokens_median:.1f} tokens")
                print(f"  Top 25% threshold: {tokens_top25:.1f} tokens (P[extra tokens >= {tokens_top25:.1f}] = {tokens_tail_top25:.1%})")
                print(f"  Top 10% threshold: {tokens_top10:.1f} tokens (P[extra tokens >= {tokens_top10:.1f}] = {tokens_tail_top10:.1%})")

            ax.grid(True, alpha=0.3)

            if row == 0:
                display_model = model

                if "reason" in display_model:
                    display_model = ["DeepSeek-R1-Distill"] + display_model.split("-")[-2:]
                    display_model = "-".join(display_model)

                ax.set_title(display_model)

        axes[row][0].set_ylabel(rf"$\mathbf{{{dataset}}}$",labelpad=25)

    fig.supylabel(r"Additional reasoning paths ($N'-N$)", x=0.038)
    fig.supxlabel(r"Faithful reasoning paths ($N$)", y=-0.015)

    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(colorbar_label, fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12, width=0.5)
        cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.08, top=0.92)

    figure_root = Path(f"../figures/{method}")
    figure_root.mkdir(parents=True, exist_ok=True)

    output_dir = figure_root / "overcharge_by_difficulty"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_alpha={alpha}_excess_samples.pdf"
    fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)


def aggregate_results(df):
    """Aggregate over queries for which the faithful procedure stopped."""
    stopped = df[df["faithful_stopped"].fillna(False).astype(bool)].copy()
    grouped = (stopped.groupby(["dataset", "model", "alpha"], as_index=False)
        .agg(
            n_questions=("qid", "count"),
            mean_faithful_n=("faithful_stop", "mean"),
            median_faithful_n=("faithful_stop", "median"),
            mean_extra_audit=("extra_samples_audit", "mean"),
            median_extra_audit=("extra_samples_audit", "median"),
            mean_relative_overcharge=("relative_overcharge_audit", "mean"),
            median_relative_overcharge=("relative_overcharge_audit", "median"),
            hit_cap_rate=("hit_cap", "mean")
        )
    )
    return grouped


def plot_audit_evasion_summary(df, alpha_lo=0.01, alpha_hi=0.1, value_col="extra_samples_audit", only_stopped=True, save_path=None):
    """
    Plot 90th-percentile overcharge with no audit and at two audit levels
    (low vs. high alpha), with a separate y-axis scale for each dataset.
    """

    data = df.copy()

    if only_stopped and "faithful_stopped" in data.columns:
        data = data[data["faithful_stopped"].fillna(False).astype(bool)].copy()

    data = data[data["alpha"].isin([alpha_lo, alpha_hi])].copy()

    # 90th percentile under each audit level
    summary = data.groupby(["dataset", "model", "alpha"])[value_col].apply(bottom_of_top_10).reset_index(name="q90_overcharge")
    
    wide = (summary.pivot_table(
        index=["dataset", "model"],
        columns="alpha",
        values="q90_overcharge",
    ).reset_index().rename_axis(None, axis=1))

    if alpha_lo not in wide.columns or alpha_hi not in wide.columns:
        raise ValueError(f"Could not find both alpha={alpha_lo} and alpha={alpha_hi} in the data.")

    wide = wide.rename(columns={alpha_lo: "q90_lo", alpha_hi: "q90_hi"})

    # 90th percentile with no audit
    # Use one alpha slice so repeated query rows are not counted twice.
    no_audit = (
        data[np.isclose(data["alpha"], alpha_hi)].groupby(["dataset", "model"])["extra_samples_reg"]
        .apply(bottom_of_top_10)
        .reset_index(name="q90_no_audit")
    )

    wide = wide.merge(
        no_audit,
        on=["dataset", "model"],
        how="left",
    )

    wide["abs_reduction"] = wide["q90_lo"] - wide["q90_hi"]
    wide["pct_reduction"] = np.where(wide["q90_lo"] > 0, 100 * (wide["q90_lo"] - wide["q90_hi"]) / wide["q90_lo"], np.nan)
    wide["fraction_remaining"] = np.where(wide["q90_lo"] > 0, wide["q90_hi"] / wide["q90_lo"], np.nan)

    dataset_order = list(wide["dataset"].drop_duplicates())

    # Sort models independently within each dataset
    pieces = []
    for d in dataset_order:
        sub = wide[wide["dataset"] == d].sort_values("q90_lo", ascending=True)
        pieces.append(sub)

    wide = pd.concat(pieces, ignore_index=True)

    def format_model_name(model):
        if "reason" in model:
            model = ["DS"] + model.split("-")[-2:]
            return "-".join(model)
        return model

    wide["model"] = wide["model"].apply(format_model_name)
    color_no_audit = "#555555"
    color_lo = "#9C2DF7"
    color_hi = "#6CC82E"

    n_datasets = len(dataset_order)
    fig, axes = plt.subplots(
        1,
        n_datasets,
        figsize=(12, 4),
        sharey=False, 
        gridspec_kw={"wspace": 0.22},
    )

    # Handle single-dataset case
    if n_datasets == 1:
        axes = [axes]

    for ax, d in zip(axes, dataset_order):
        sub = wide[ wide["dataset"] == d].reset_index(drop=True)

        x = np.arange(len(sub))
        for i, row in sub.iterrows():
            # Connector through all three audit conditions
            ax.plot([i, i], [ row["q90_hi"], row["q90_no_audit"]],
                linewidth=1.0,
                color="0.45",
                zorder=1,
            )

            # No audit
            ax.scatter(i, row["q90_no_audit"], s=22, color=color_no_audit, zorder=2)

            # alpha_lo
            ax.scatter(i, row["q90_lo"], s=22, color=color_lo, zorder=2)

            # alpha_hi
            ax.scatter(i, row["q90_hi"], s=22, color=color_hi, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [fr"\texttt{{{model}}}" for model in sub["model"]],
            rotation=50,
            ha="right",
            fontsize=11,
            rotation_mode="anchor",
        )

        subcaptions = {'AIME': '(a)', 'GPQA': '(b)', 'GSM8K': '(c)'}

        ax.set_title(
            subcaptions[d] + ' ' + fr"\texttt{{{d}}}",
            fontsize=14,
            fontweight="bold",
            y=-0.6
        )

        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)

        y_max = np.nanmax(sub[["q90_no_audit", "q90_lo", "q90_hi"]].to_numpy())
        ax.set_ylim(0, y_max * 1.08)

    for ax in axes:
        ax.set_ylabel(r"90th percentile of $N'-N$")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_no_audit, label=rf"$\alpha=0$ (no audit)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_lo, label=rf"$\alpha={alpha_lo}$"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_hi, label=rf"$\alpha={alpha_hi}$")
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.82, wspace=0.30)

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, wide

def print_bon_audit_binding_rates(df, alphas=(0.01, 0.1), only_stopped=True):
    data = df.copy()

    if only_stopped and "faithful_stopped" in data.columns:
        data = data[data["faithful_stopped"].fillna(False).astype(bool)].copy()

    data["extra_samples_reg"] = pd.to_numeric(data["extra_samples_reg"], errors="coerce")
    data["extra_samples_audit"] = pd.to_numeric(data["extra_samples_audit"], errors="coerce")
    data = data[
        data["alpha"].isin(alphas)
        & data["extra_samples_reg"].notna()
        & data["extra_samples_audit"].notna()
    ].copy()

    data["audit_binds"] = data["extra_samples_audit"] < data["extra_samples_reg"]
    data["samples_removed_by_audit"] =  data["extra_samples_reg"] - data["extra_samples_audit"]

    summary = (data.groupby(["dataset", "model", "temperature", "alpha"], as_index=False).agg(
            n_queries=("audit_binds", "size"),
            n_audit_binds=("audit_binds", "sum"),
            fraction_audit_binds=("audit_binds", "mean"),
            mean_samples_removed=("samples_removed_by_audit", "mean"),
        )
    )

    summary["fraction_audit_binds"] *= 100

    for _, row in summary.iterrows():
        print(f"{row['dataset']:6s} | {row['model']:30s} | "
            f"alpha={row['alpha']:.2f} | audit binds: {int(row['n_audit_binds'])}/"
            f"{int(row['n_queries'])} ({row['fraction_audit_binds']:.1f}%) | "
            f"mean samples removed: {row['mean_samples_removed']:.2f}")

    return summary

def plot_bon_audit_evasion_summary(df, alpha_lo=0.01, alpha_hi=0.1, value_col="extra_samples_audit", only_stopped=True, save_path=None):
    """
    Plot the top-10% threshold of audit-safe overcharge for threshold
    best-of-N, with no audit and at two audit significance levels.

    Assumes `extra_samples_audit` was computed using the BoN-specific
    threshold-crossing audit.
    """

    data = df.copy()

    if only_stopped and "faithful_stopped" in data.columns:
        data = data[data["faithful_stopped"].fillna(False).astype(bool)].copy()

    data = data[data["alpha"].isin([alpha_lo, alpha_hi])].copy()

    # Top-10% threshold under each audit level
    summary = data.groupby(["dataset", "model", "alpha"])[value_col].apply(bottom_of_top_10).reset_index(name="q90_overcharge")

    wide = (summary.pivot_table(
        index=["dataset", "model"],
        columns="alpha",
        values="q90_overcharge",
    ).reset_index().rename_axis(None, axis=1))

    if alpha_lo not in wide.columns or alpha_hi not in wide.columns:
        raise ValueError(f"Could not find both alpha={alpha_lo} and alpha={alpha_hi} in the data.")

    wide = wide.rename(columns={alpha_lo: "q90_lo", alpha_hi: "q90_hi"})

    # Top-10% threshold with no audit
    # Use one alpha slice so each query is counted only once.
    no_audit = data[np.isclose(data["alpha"], alpha_hi)].groupby(["dataset", "model"])["extra_samples_reg"].apply(bottom_of_top_10).reset_index(name="q90_no_audit")
    wide = wide.merge(no_audit, on=["dataset", "model"], how="left")

    # Summarize trigger probabilities 
    if "audit_trigger_prob" in data.columns:

        q_summary = (
            data[np.isclose(data["alpha"], alpha_hi)]
            .groupby(["dataset", "model"])["audit_trigger_prob"]
            .median()
            .reset_index(name="median_trigger_prob")
        )

        wide = wide.merge(q_summary, on=["dataset", "model"], how="left")

    # Audit effect
    wide["abs_reduction"] =  wide["q90_lo"]  - wide["q90_hi"]
    wide["pct_reduction"] = np.where(wide["q90_lo"] > 0, 100 * (wide["q90_lo"] - wide["q90_hi"]) / wide["q90_lo"], np.nan)
    wide["fraction_remaining"] = np.where(wide["q90_lo"] > 0, wide["q90_hi"] / wide["q90_lo"], np.nan)

    dataset_order = list(wide["dataset"].drop_duplicates())

    # Sort models independently within each dataset
    pieces = []
    for d in dataset_order:
        sub =  wide[wide["dataset"] == d].sort_values("q90_lo", ascending=True)
        pieces.append(sub)

    wide = pd.concat(pieces,  ignore_index=True)

    def format_model_name(model):
        model = str(model)
        if "reason" in model.lower():
            model = ["DS"] + model.split("-")[-2:]
            return "-".join(model)

        return model

    wide["model"] = wide["model"].apply(format_model_name)

    color_no_audit = "#555555"
    color_lo = "#9C2DF7"
    color_hi = "#6CC82E"

    n_datasets = len(dataset_order)

    fig, axes = plt.subplots(1, n_datasets,
        figsize=(12, 4),
        sharey=False,
        gridspec_kw={"wspace": 0.22}
    )

    if n_datasets == 1:
        axes = [axes]

    for ax, d in zip(axes, dataset_order):
        sub = wide[wide["dataset"] == d].reset_index(drop=True)
        x = np.arange(len(sub))

        for i, row in sub.iterrows():

            # Connector through all three audit conditions
            ax.plot([i, i], [row["q90_hi"], row["q90_no_audit"]],
                linewidth=1.0,
                color="0.45",
                zorder=1,
            )

            # No audit
            ax.scatter(i, row["q90_no_audit"], s=22, color=color_no_audit, zorder=2)

            # Lower alpha
            ax.scatter(i, row["q90_lo"], s=22, color=color_lo, zorder=2)

            # Higher alpha
            ax.scatter(i, row["q90_hi"], s=22, color=color_hi, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=50, ha="right", fontsize=11, rotation_mode="anchor")
        ax.set_title(d, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)

        # Independent y-axis scale for each dataset
        y_max = np.nanmax(sub[["q90_no_audit", "q90_lo", "q90_hi"]].to_numpy())

        if np.isfinite(y_max) and y_max > 0:
            ax.set_ylim(0, y_max * 1.08)

    for ax in axes:
        ax.set_ylabel(r"Top-10\% threshold for $N'-N$")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_no_audit, label=r"$\alpha=0$ (no audit)"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_lo, label=rf"$\alpha={alpha_lo}$"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=color_hi, label=rf"$\alpha={alpha_hi}$"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=12,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.82, wspace=0.30)

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, wide

def plot_billing_increase_by_query_difficulty(df, method, datasets, model_temps, alpha):
    """Plot audit-safe absolute monetary billing overcharge by query, in cents."""

    df = df.copy()
    n_rows = len(datasets)
    n_models = len(model_temps)

    fig = plt.figure(figsize=(12.5, 3 * n_rows))

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_models + 1,
        width_ratios=[1] * n_models + [0.07],
        wspace=0.15,
        hspace=0.15,
    )

    axes = [[fig.add_subplot(gs[row, col]) for col in range(n_models)] for row in range(n_rows)]
    cax = fig.add_subplot(gs[:, n_models])
    scatter = None

    difficulty_col, _ = _difficulty_spec(method)

    colorbar_label = r"Trigger probability $\hat q$" if method == "threshold_bon" else r"$\hat p_{(1)}-\hat p_{(2)}$"

    for row, dataset in enumerate(datasets):
        for i, (ax, (model, temperature, _)) in enumerate(zip(axes[row], model_temps)):
            s = df[
                (df["dataset"] == dataset)
                & (df["model"] == model)
                & np.isclose(
                    df["temperature"],
                    temperature,
                )
                & np.isclose(
                    df["alpha"],
                    alpha,
                )
                & df["faithful_stopped"]
                .fillna(False)
                .astype(bool)
            ].copy()

            if len(s) == 0:
                print(f"No observations for {dataset}, {model}, T={temperature}, alpha={alpha}")
                continue

            if model not in PRICE_PER_M_OUTPUT_TOKEN:
                raise KeyError(f"No output-token price specified for model={model}")

            extra_output_tokens = pd.to_numeric(s["extra_output_tokens_audit"], errors="coerce")
            billing_overcharge_cents = extra_output_tokens * PRICE_PER_M_OUTPUT_TOKEN[model] / 1_000_000 * 100

            valid = (
                billing_overcharge_cents.notna()
                & np.isfinite(billing_overcharge_cents)
                & np.isfinite(s[difficulty_col])
            )

            s = s.loc[valid].copy()
            billing_overcharge_cents = billing_overcharge_cents.loc[valid]

            if len(s) == 0:
                print(f"No valid billing observations for {dataset}, {model}, T={temperature}, alpha={alpha}")
                continue

            scatter = ax.scatter(s["faithful_stop"], billing_overcharge_cents,
                c=s[difficulty_col],
                cmap="magma",
                vmin=0,
                vmax=1,
                alpha=0.5,
                s=30,
                edgecolors="none",
            )
            ax.set_xscale("symlog")
            ax.set_yscale("symlog")
            xticks = [10, 100, 1000]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])

            yticks = [0, 1, 5, 10, 60]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y) for y in yticks])

            ax.grid(True, alpha=0.3)

            if row == 0:
                model_name = _display_model_name(model)
                ax.set_title(fr"\texttt{{{model_name}}}", fontsize=15)

        axes[row][0].set_ylabel(fr"$\texttt{{{dataset}}}$", fontsize=16, labelpad=25)


    fig.supylabel("Billing overcharge in cents", x=0.04)
    fig.supxlabel(r"Faithful reasoning paths ($N$)",  y=0.02)

    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(colorbar_label, fontsize=16, labelpad=10)
        cbar.ax.tick_params(labelsize=12, width=0.5)

        cbar.outline.set_linewidth(0.5)

    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.08, top=0.92)
    figure_root = Path(f"../figures/{method}")

    figure_root.mkdir(parents=True, exist_ok=True)
    output_dir = figure_root / "billing_increase"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_alpha={alpha}_billing_overcharge.pdf"

    fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)

def plot_overcharge_distribution(df, method, datasets, alpha=0.1):

    """
    Plot empirical CCDFs of audit-safe additional reasoning paths N' - N.
    """

    df = df.copy()

    def get_family(model):
        m = str(model).lower()
        # Many reasoning models also contain "Llama" or "Qwen".
        if ("reason" in m or "deepseek-r1" in m or "r1-distill" in m):
            return "DeepSeek-R1-Distill"

        if "llama" in m:
            return "Llama"

        if "qwen" in m:
            return "Qwen"

        return None


    def display_model_name(model):
        name = str(model)
        if "reason-r1-d" in name.lower():
            tags = name.split("-")
            return "R1-Distill-" + tags[-2] + '-' + tags[-1]

        name = name.replace("-Instruct", "")
        return name


    available = (
        df[np.isclose(df["alpha"], alpha) & (df["faithful_stopped"] == True)][["model", "temperature"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["model", "temperature"])
    )

    model_temps = [(row.model, float(row.temperature), None) for row in available.itertuples(index=False)]

    # Group models by family
    model_groups = {
        "Llama": [],
        "Qwen": [],
        "DeepSeek-R1-Distill": [],
    }

    for model, temperature, t_max in model_temps:
        family = get_family(model)

        if family is not None:
            model_groups[family].append((model, temperature, t_max))

    # Plot 3 per family
    model_groups = {family: models[:3] for family, models in model_groups.items() if len(models) > 0}
    families = list(model_groups.keys())

    if len(families) == 0:
        raise ValueError("No Llama, Qwen, or reasoning models found.")

    n_panels = len(families)

    fig, axes = plt.subplots(1, n_panels,
        figsize=(4.5 * n_panels, 4),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    axes = axes[0]
    cmap = plt.get_cmap("Set2")
    dataset_colors = {dataset: cmap(i) for i, dataset in enumerate(datasets)}

    all_values = []
    max_n = 0

    for models in model_groups.values():
        for model, temperature, _ in models:
            for dataset in datasets:
                s = df[
                    (df["dataset"] == dataset)
                    & (df["model"] == model)
                    & np.isclose(
                        df["temperature"],
                        temperature,
                    )
                    & np.isclose(
                        df["alpha"],
                        alpha,
                    )
                    & (df["faithful_stopped"] == True)
                ]

                vals = s["extra_samples_audit"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                vals = vals[vals >= 0]

                if len(vals):
                    all_values.extend(vals)
                    max_n = max(max_n, len(vals))

    global_xmax = max(all_values)  if all_values else 100

    for ax, family in zip(axes, families):
        models = model_groups[family]

        if len(models) == 1:
            widths = [1.5]
        else:
            widths = [0.5, 1.5, 3.0]

        model_widths = {(model, temperature): width for width, (model, temperature, _) in zip(widths, models)}

        # Check whether the same model appears at multiple temperatures.
        model_counts = {}

        for model, temperature, _ in models:
            model_counts[model] = model_counts.get(model, 0) + 1
            
        for model, temperature, _ in models:
            linewidth = model_widths[(model, temperature)]
            for dataset in datasets:
                s = df[
                    (df["dataset"] == dataset)
                    & (df["model"] == model)
                    & np.isclose(
                        df["temperature"],
                        temperature,
                    )
                    & np.isclose(
                        df["alpha"],
                        alpha,
                    )
                    & (
                        df["faithful_stopped"]
                        == True
                    )
                ].copy()

                values = s["extra_samples_audit"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
                values = values[values >= 0]

                if len(values) == 0:
                    continue

                x, counts = np.unique(values, return_counts=True)
                y =  np.cumsum(counts[::-1])[::-1] / len(values)
              
                ax.step(x, y,
                    where="post",
                    linewidth=linewidth,
                    linestyle="-",
                    color=dataset_colors[dataset],
                    alpha=0.9,
                )

        # Linear from 0--10, log thereafter.
        ax.set_xscale("symlog", linthresh=10, linscale=1, base=10)
        xticks = [x for x in [0, 1, 2, 5, 10, 100, 1000, 5000] if x <= global_xmax * 1.05]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=12)
        ax.set_yscale("log")
        ax.set_ylim(0.001,1.05)
        ax.set_xlim(0.0,)
        yticks = [0.001, 0.01, 0.1, 0.5, 1]
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(y) for y in yticks])

        ax.grid(True, which="major", alpha=0.2)

        models_label = "(a)"
        if family == "Qwen":
            models_label = "(b)"
        elif family == "Reasoning":
            models_label = "(c)"
        models_label += fr" \texttt{{{family}}} models"
        ax.set_title(models_label, fontsize=16, y=-0.3)

   
        model_handles = []

        for model, temperature, _ in models:
            label = display_model_name(model)
            # # If the same model occurs at multiple temperatures,
            # # make temperature explicit.
            # if model_counts[model] > 1:
            #     label += rf" ($T={temperature:g}$)"

            model_handles.append(Line2D([0], [0], color="0.25", linestyle="-", linewidth=model_widths[(model, temperature)], label=fr"\texttt{{{label}}}"))

        legend_cols = 2 if len(model_handles) >= 6 else 1

        ax.legend(
            handles=model_handles,
            loc="lower left",
            frameon=False,
            fontsize=11,
            title_fontsize=8.5,
            ncol=legend_cols,
            handlelength=2.0,
            columnspacing=2.5,
            labelspacing=0.35,
        )
        ax.set_xlabel(r"$N'-N$", y=0.0, fontsize=16)

    fig.supylabel(r"$\Pr(N'-N \geq x)$", x=0.025, fontsize=16)

    dataset_handles = [Line2D([0],[0], color=dataset_colors[dataset], lw=2.2, linestyle="-", label=fr"\texttt{{{dataset}}}") for dataset in datasets]
    fig.legend(
        handles=dataset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(datasets),
        frameon=False,
        fontsize=14
    )

    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.16, top=0.85, wspace=0.15)

    figure_root = Path(f"../figures/{method}")
    output_dir = figure_root / "overcharge_distribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_alpha={alpha}_overcharge_ccdf.pdf"
    fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["asc", "esc", "threshold_bon"], required=True)
    args = parser.parse_args()

    method = args.method
    results_path = f"../results/{method}_all_results.csv"

    df = pd.read_csv(results_path)
    summary = aggregate_results(df)

    alpha = 0.1
    model_temps = [
        ("Llama-3.2-3B", 0.6, 128),
        ("Qwen2.5-7B", 0.7, 128),
        ("reason-R1-D-Qwen-7B", 0.6, 32),
    ]

    plot_by_query_difficulty(
        df=df,
        datasets=["GSM8K", "AIME", "GPQA"],
        method=method,
        model_temps=model_temps,
        alpha=alpha,
    )

    plot_billing_increase_by_query_difficulty(
        df,
        method,
        ["GSM8K", "AIME", "GPQA"],
        model_temps,
        alpha=alpha,
    )

    plot_overcharge_distribution(
        df=df,
        method=method,
        datasets=["GSM8K", "AIME", "GPQA"],
        alpha=0.1,
    )

    if method == "threshold_bon":
        binding_summary = print_bon_audit_binding_rates(
            df,
            alphas=(0.01, 0.1),
        )
        fig, bon_summary = plot_bon_audit_evasion_summary(
            df,
            alpha_lo=0.01,
            alpha_hi=0.1,
            value_col="extra_samples_audit",
            save_path="../figures/threshold_bon/threshold_bon_audit_evasion_summary.pdf",
        )
    else:
        fig, summary_df = plot_audit_evasion_summary(
            df,
            alpha_lo=0.01,
            alpha_hi=0.1,
            value_col="extra_samples_audit",
            save_path=f"../figures/{method}/{method}_audit_evasion_summary.pdf",
        )
        
