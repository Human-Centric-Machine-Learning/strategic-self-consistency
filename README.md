# Strategic Self-Consistency

## Setup

Download the **Strategic Test-Time Compute (TTC) Games** dataset from Hugging Face ([Human-Centric-Machine-Learning/strategic-ttc-data](https://huggingface.co/datasets/Human-Centric-Machine-Learning/strategic-ttc-data)) into ``hf_data/``. There should be subdirectories for each of the problem-solving datasets used in the experiments:
```
hf_data/
├── AIME/
├── GPQA/
└── GSM8K/
```
If a downloaded file exists but only contains a Git LFS pointer rather than actual data, run `git lfs pull`. 

### Dependencies
All experiments were run using Python 3.13.5. Project dependencies can be installed from `requirements.txt`.

## Running Experiments

### LLM Experiments
The experiments in `llm_experiments/` simulate provider overcharging and likelihood-based auditing using response distributions from real model generations for:
- Llama-3-8B-Instruct, Llama-3.1-8B-Instruct,Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct,
- Qwen-2-0.5B-Instruct, Qwen-2-1.5B-Instruct, Qwen-2-7B-Instruct, Qwen-2.5-3B-Instruct, Qwen-2.5-7B-Instruct,
- DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B.

Run experiment for each model-dataset pair by specifying a stopping rule: 
```
python run_unfaithful.py --method=asc
```
for Adaptive Self-Consistency (ASC) with a Beta criterion,

```
python run_unfaithful.py --method=esc
```
for Early-Stopping Self-Consistency (ESC), or
```
python run_unfaithful.py --method=threshold_bon
```
for threshold-based adaptive best-of-N.

Experiment results are cached in `results/` so that interrupted runs can resume from previously completed queries. Results are stored separately for each stopping rule, e.g., `asc_all_results.csv`, `esc_all_results.csv`, and `threshold_bon_all_results.csv`. Plots for the LLM experiments can be generated using:
```
python plot_results.py --method={asc, esc, or threshold_bon}
```

### Synthetic Experiments
The experiments in `synthetic_experiments/` compare theoretical lower bounds on continuation attacks against empirical mean overcharge under controlled answer distributions for the following stopping rules:
1. **PPR-1v1**: one-vs-one prior-posterior ratio (PPR) martingale confidence sequences with $\delta=0.1$ and $\epsilon=0$.
2. **ASC**: adaptive self-consistency with a Beta stopping criterion and confidence level $\gamma=0.95$. 

To compute realized overcharge and compare it against the projected top-two lower bound, run:
```
python attack_top_two_lb.py
```

## Repository Structure
```
.
├── figures/
│   ├── asc/
│   │   ├── billing_increase/
│   │   ├── overcharge_by_difficulty/
│   │   ├── overcharge_distribution/
│   │   └── asc_audit_evasion_summary.pdf
│   ├── esc/
│   │   ├── ...
│   ├── threshold_bon/
│   │   ├── ...
│   └── attack_lower_bound.pdf
│
├── hf_data/
│   ├── AIME/
│   ├── GPQA/
│   └── GSM8K/
│
├── llm_experiments/
│   ├── plot_results.py
│   └── run_unfaithful.py
│
├── results/
│   ├── asc_all_results.csv
│   ├── esc_all_results.csv
│   ├── threshold_bon_all_results.csv
│   └── attack_lower_bound.csv
│
├── synthetic_experiments/
│   └── attack_top_two_lb.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── utils.py
```

- `hf_data/` contains the pre-generated LLM reasoning paths used in the real-model experiments.
- `llm_experiments/` contains the continuation-attack simulations and plotting code for real LLM response distributions.
- `synthetic_experiments/` contains controlled simulations used to evaluate the theoretical lower bounds.
- `results/` contains cached experimental results.
- `figures/` contains generated plots and paper figures.
- `utils.py` contains shared stopping rules, continuation-attack and likelihood-auditing routines, and supporting utilities.