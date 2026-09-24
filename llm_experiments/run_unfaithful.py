"""
Simulate an unfaithful LLM inference provider using a stopping rule tau

Stopping rules tested:
    - Adaptive self-consistency (ASC) with Beta criterion 
    - Early-stopping self-consistency (ESC)
    - Adaptive best-N with a threshold
"""
import argparse
import pandas as pd
from collections import Counter
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import csv
import numpy as np
from pathlib import Path
import json
from functools import lru_cache


T_MAX = 128
DATA_ROOT = Path("../hf_data")
ALPHAS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1]


#  (qid, prompt, answers, correct, explanations, num_tokens, rewards, ground_truths, question_meta, model_name, reward_model_name)
def get_boxed(explanations):
    boxed = explanations.split(';')[0]
    ans = boxed.split('=')
    if len(ans) < 2:
        ans = ''
    else:
        ans = ans[1].strip('\'')
    return ans

def get_answer_and_token_sequences(explanations, num_tokens, skip_unparsed=True):
    """
    Return parsed answers and their output-token counts.
    skip_unparsed=True drop samples with no boxed answer ('') from the sequence
    """
    answers = []
    token_counts = []

    for explanation, tokens in zip(explanations, num_tokens):
        answer = get_boxed(explanation)

        if answer == "" and skip_unparsed:
            continue

        answers.append(answer)
        token_counts.append(tokens)

    return answers, np.asarray(token_counts, dtype=float)


def parse_file_metadata(path):
    path = Path(path)
    dataset = path.parent.name
    config = path.stem
    parts = config.split("--")
    model = parts[0]

    metadata = {
        "dataset": dataset,
        "model": model,
        "config": config,
        "path": str(path),
    }

    for part in parts[1:]:
        if part.startswith("temp-"):
            metadata["temperature"] = float(part.removeprefix("temp-")) 
        elif part.startswith("samples-"):
            metadata["num_samples"] = int(part.removeprefix("samples-")) 
        elif part.startswith("max-"):
            metadata["max_tokens"] = int(part.removeprefix("max-"))

    return metadata



def get_boxed(explanation):
    boxed = explanation.split(";")[0]
    ans = boxed.split("=")
    return ans[1].strip() if len(ans) >= 2 else ""


def get_answer_sequence(explanations, skip_unparsed=True):
    seq = []

    for e in explanations:
        a = get_boxed(e)
        if a == "" and skip_unparsed:
            continue
        seq.append(a)

    return seq


def empirical_top_two(seq):
    counts = Counter(seq)
    probs = sorted([c / len(seq) for c in counts.values()], reverse=True)
    p1 = probs[0]
    p2 = probs[1] if len(probs) > 1 else 0.0
    return p1, p2


def run_file(path, alpha, tau, writer, output_file, completed=None):
    """
    Run one model/dataset file at an alpha-level audit.

    completed:
        Optional set of (path, alpha, qid) tuples that have
        already been written, allowing the experiment to resume.
    """
    meta = parse_file_metadata(path)
    data = pd.read_json(path, lines=True)

    path_str = str(path)

    for row_idx, row in data.iterrows():
        qid = row["qid"]
        key = (path_str, float(alpha), str(qid))

        if completed is not None and key in completed:
            continue

        # Filter answers and token counts together so their indices align.
        ans_seq, token_counts = get_answer_and_token_sequences(row["explanations"], row["num_tokens"])

        cap = min(T_MAX, len(ans_seq))

        ans_seq = ans_seq[:cap]
        token_counts = token_counts[:cap]

        if cap == 0:
            continue

        K = len(set(ans_seq))
        p1, p2 = empirical_top_two(ans_seq)

        rng = np.random.default_rng(utils.stable_seed(path_str, qid))
        result = utils.simulate_provider(ans_seq, T_max=cap, alpha=alpha, stop_rule=tau, rng=rng)

        faithful_n = result["faithful_n"]
        adversarial_n = result["adversarial_n"]
        audit_safe_n = result["audit_safe_n"]

        extra_reg = adversarial_n - faithful_n
        extra_audit = audit_safe_n - faithful_n

        # Token-level billing
        if result["faithful_stopped"]:
            source_indices = np.asarray(result["generation_source_indices"], dtype=int)
            faithful_output_tokens = (token_counts[source_indices[:faithful_n]].sum())
            unfaithful_output_tokens = token_counts[source_indices].sum()
            audit_safe_output_tokens = token_counts[source_indices[:audit_safe_n]].sum()
            extra_output_tokens_reg = unfaithful_output_tokens - faithful_output_tokens
            extra_output_tokens_audit = audit_safe_output_tokens - faithful_output_tokens

            if faithful_output_tokens > 0:
                relative_token_overcharge_reg = extra_output_tokens_reg / faithful_output_tokens
                relative_token_overcharge_audit = extra_output_tokens_audit / faithful_output_tokens
            else:
                relative_token_overcharge_reg = np.nan
                relative_token_overcharge_audit = np.nan
        else:
            faithful_output_tokens = np.nan
            unfaithful_output_tokens = np.nan
            audit_safe_output_tokens = np.nan
            extra_output_tokens_reg = np.nan
            extra_output_tokens_audit = np.nan
            relative_token_overcharge_reg = np.nan
            relative_token_overcharge_audit = np.nan

        record = {
            **meta,
            "alpha": alpha,
            "qid": qid,
            "K": K,
            "n_available": cap,
            "faithful_stopped": result["faithful_stopped"],
            "faithful_stop": faithful_n,
            "unfaithful_stop": adversarial_n,
            "audit_safe_stop": audit_safe_n,

            # Sample-level overcharge
            "extra_samples_reg": extra_reg,
            "extra_samples_audit": extra_audit,
            "relative_overcharge_reg": extra_reg / faithful_n if faithful_n > 0 else np.nan,
            "relative_overcharge_audit": extra_audit / faithful_n if faithful_n > 0 else np.nan,

            # Real pre-generated samples
            "real_sample_headroom": max(cap - faithful_n, 0),
            "hit_cap": result["hit_cap"],
            "exceeded_cap": result["exceeded_cap"],
            "used_empirical_extension": result["used_empirical_extension"],

            # Difficulty
            "p1": p1,
            "p2": p2,
            "gap": p1 - p2,

            # Token-level billing
            "faithful_output_tokens": faithful_output_tokens,
            "unfaithful_output_tokens": unfaithful_output_tokens,
            "audit_safe_output_tokens": audit_safe_output_tokens,
            "extra_output_tokens_reg": extra_output_tokens_reg,
            "extra_output_tokens_audit": extra_output_tokens_audit,
            "relative_token_overcharge_reg": relative_token_overcharge_reg,
            "relative_token_overcharge_audit": relative_token_overcharge_audit,
            "percent_token_overcharge_reg": 100 * relative_token_overcharge_reg,
            "percent_token_overcharge_audit": 100 * relative_token_overcharge_audit,
        }

        writer.writerow(record)
        output_file.flush()

        if completed is not None:
            key = (path_str, str(qid))
            completed.add(key)


def load_cached_results(output_path):
    """
    Load cached experiment results from CSV.
    """
    output_path = Path(output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        return pd.DataFrame(columns=utils.RESULT_FIELDS), set()
    try:
        cached_df = pd.read_csv(output_path, dtype={"path": str, "qid": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=utils.RESULT_FIELDS), set()

    # Do not append current-format rows under an old/different CSV header.
    if list(cached_df.columns) != utils.RESULT_FIELDS:
        raise ValueError( f"Cached CSV schema does not match RESULT_FIELDS: {output_path}\n")

    # Ignore any incomplete row that may have been left by an interrupted run.
    valid = (cached_df["path"].notna() & cached_df["alpha"].notna() & cached_df["qid"].notna())
    cached_df = cached_df.loc[valid].copy()
    completed = { (str(row.path), float(row.alpha), str(row.qid)) for row in cached_df[["path", "alpha", "qid"]].itertuples(index=False)}

    return cached_df, completed


def load_results(output_path):
    cached_df, _ = load_cached_results(output_path)
    return cached_df

# Simulate unfaithful provider running attack under audit for all hf_data files
def run_all_experiments(method, output_path):
    """
    Run all missing experiments and cache each completed row immediately.

    If output_path already exists, completed (path, alpha, qid) experiments
    are loaded from the CSV and skipped. The returned DataFrame contains both
    previously cached and newly computed rows.
    """
    if method == "asc":
        tau = utils.make_asc_stop_rule()
    elif method == "esc":
        tau = utils.make_esc_stop_rule(window=3)


    files = sorted(DATA_ROOT.glob("*/*.jsonl"))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cached_df, completed = load_cached_results(output_path)
    print(f"Loaded {len(cached_df)} cached result rows from {output_path}")

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=utils.RESULT_FIELDS, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        for path in files:
            print(path)
            for alpha in ALPHAS:
                before = len(completed)
                run_file(path=path, alpha=alpha, tau=tau, writer=writer, output_file=f, completed=completed)
                n_new = len(completed) - before
                if n_new:
                    print(f"  alpha={alpha}: cached {n_new} new rows")

    # Reload from disk so the caller gets one DataFrame containing both the
    # old cache and everything written during this run.
    results_df, _ = load_cached_results(output_path)
    return results_df


def run_bon_file(path, alpha, writer, output_file, completed=None):
    """
    Run adaptive best-of-N continuation experiments for one JSONL file.
    Uses the same output naming as run_file() for self-consistency.
    """
    meta = parse_file_metadata(path)
    data = pd.read_json(path, lines=True)
    path_str = str(path)
    for _, row in data.iterrows():
        qid = row["qid"]

        key = (path_str, float(alpha), str(qid))
        if completed is not None and key in completed:
            continue

        rewards = np.asarray(row["rewards"], dtype=float)
        token_counts = np.asarray(row["num_tokens"], dtype=float)

        cap = min(T_MAX, len(rewards))
        rewards = rewards[:cap]
        token_counts = token_counts[:cap]

        if cap == 0:
            continue

        reward_threshold = np.median(rewards) * 1.25
        tau = utils.make_threshold_bon_stop_rule(reward_threshold=reward_threshold, min_samples=1)

        # Plug-in estimate: q = P(R >= r*)
        audit_trigger_prob = np.mean(rewards >= reward_threshold)

        rng = np.random.default_rng(utils.stable_seed(path_str, qid))
        result = utils.simulate_adaptive_bon_provider(rewards, audit_trigger_prob, T_max=cap, alpha=alpha, stop_rule=tau, rng=rng)

        faithful_n = result["faithful_n"]
        adversarial_n = result["adversarial_n"]
        audit_safe_n = result["audit_safe_n"]

        # Sample-level overcharge
        extra_reg = adversarial_n - faithful_n
        extra_audit = audit_safe_n - faithful_n

        # Token-level billing
        if result["faithful_stopped"]:
            source_indices = np.asarray(result["generation_source_indices"], dtype=int)
            faithful_output_tokens = token_counts[:faithful_n].sum()
            unfaithful_output_tokens = token_counts[source_indices].sum()
            audit_safe_output_tokens = token_counts[source_indices[:audit_safe_n]].sum()
            extra_output_tokens_reg = unfaithful_output_tokens - faithful_output_tokens
            extra_output_tokens_audit = audit_safe_output_tokens - faithful_output_tokens

            if faithful_output_tokens > 0:
                relative_token_overcharge_reg = extra_output_tokens_reg / faithful_output_tokens
                relative_token_overcharge_audit = extra_output_tokens_audit / faithful_output_tokens
            else:
                relative_token_overcharge_reg = np.nan
                relative_token_overcharge_audit = np.nan

        else:
            faithful_output_tokens = np.nan
            unfaithful_output_tokens = np.nan
            audit_safe_output_tokens = np.nan

            extra_output_tokens_reg = np.nan
            extra_output_tokens_audit = np.nan

            relative_token_overcharge_reg = np.nan
            relative_token_overcharge_audit = np.nan

       
        record = {
            **meta,

            "alpha": alpha,
            "qid": qid,
            "n_available": cap,

            # BoN-specific quantities
            "reward_threshold": reward_threshold,
            "audit_trigger_prob": audit_trigger_prob,

            # Stopping times
            "faithful_stopped": result["faithful_stopped"],
            "faithful_stop": faithful_n,
            "unfaithful_stop": adversarial_n,
            "audit_safe_stop": audit_safe_n,

            # Sample-level overcharge
            "extra_samples_reg": extra_reg,
            "extra_samples_audit": extra_audit,

            "relative_overcharge_reg": (
                extra_reg / faithful_n
                if faithful_n > 0 else np.nan
            ),
            "relative_overcharge_audit": extra_audit / faithful_n if faithful_n > 0 else np.nan,

            # Reservoir information
            "real_sample_headroom": cap - faithful_n,
            "hit_cap": result["hit_cap"],
            "exceeded_cap": result["exceeded_cap"],
            "used_empirical_extension": result["used_empirical_extension"],

            # Token-level billing
            "faithful_output_tokens": faithful_output_tokens,
            "unfaithful_output_tokens":  unfaithful_output_tokens,
            "audit_safe_output_tokens": audit_safe_output_tokens,
            "extra_output_tokens_reg": extra_output_tokens_reg,
            "extra_output_tokens_audit": extra_output_tokens_audit,
            "relative_token_overcharge_reg": relative_token_overcharge_reg,
            "relative_token_overcharge_audit": relative_token_overcharge_audit,
            "percent_token_overcharge_reg":  100 * relative_token_overcharge_reg,
            "percent_token_overcharge_audit": 100 * relative_token_overcharge_audit,
        }

        writer.writerow(record)
        output_file.flush()

        if completed is not None:
            completed.add(key)


@lru_cache(maxsize=None)
def load_jsonl_by_qid(path):
    """
    Load one source JSONL and index rows by qid.
    Cached so we only read each JSONL file once.
    """
    rows = {}

    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            rows[str(item["qid"])] = item

    return rows


def load_cached_bon_results(output_path):
    output_path = Path(output_path)

    if not output_path.exists() or output_path.stat().st_size == 0:
        return pd.DataFrame(columns=utils.RESULT_FIELDS), set()

    df = pd.read_csv(output_path, dtype={"path": str, "qid": str})
    valid = (df["path"].notna() & df["alpha"].notna() & df["qid"].notna())

    df = df.loc[valid].copy()
    completed = {(str(row.path), float(row.alpha), str(row.qid)) for row in df[["path", "alpha", "qid"]].itertuples(index=False)}

    return df, completed


def run_all_bon_experiments(output_path):
    """
    Run all missing adaptive best-of-N experiments and cache each
    completed row immediately. If output_path already exists, 
    completed (path,  qid) experiments are loaded from the CSV
    and skipped. 
    """
    files = sorted(DATA_ROOT.glob("*/*.jsonl"))
  
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cached_df, completed = load_cached_bon_results(output_path)
    print(f"Loaded {len(cached_df)} cached result rows from {output_path}")

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=utils.RESULT_FIELDS, extrasaction="ignore")

        if not file_exists:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        for path in files:
            print(path)
            for alpha in ALPHAS:
                before = len(completed)
                run_bon_file(path=path, alpha=alpha, writer=writer, output_file=f, completed=completed)
                n_new = len(completed) - before
                if n_new:
                    print(f"  alpha={alpha}: cached {n_new} new rows")

    # Reload from disk so the caller gets both cached and newly
    # computed results.
    results_df, _ = load_cached_bon_results(output_path)

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=['asc', 'esc', 'threshold_bon'], required=True)
    args = parser.parse_args()
    results_path = f"../results/{args.method}_all_results.csv"

    if args.method == "threshold_bon":
        results = run_all_bon_experiments(results_path)
    else:
        results = run_all_experiments(args.method, results_path)

        

