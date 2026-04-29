# Rethinking SSL Evaluation

Companion repository for an AI workshop submission on semi-supervised learning (SSL) evaluation under controlled labeling budgets.

This project compares classical SSL training schemes on CIFAR-10 and SVHN with a shared backbone and a budget-aware evaluation protocol:

- Full Supervised (reference upper bound)
- Supervised with labeled-only subset
- MixMatch
- FixMatch
- SequenceMatch
- Variants with $\mu=1$ for tighter unlabeled-to-labeled ratios

## Why This Repo

Many SSL comparisons focus only on final accuracy. This repository emphasizes the trajectory under constrained supervision, including:

- test accuracy and weighted F1 over training
- explicit budget tracking per iteration
- aligned experimental structure across datasets and methods

The goal is to make method-to-method comparisons easier to audit and reproduce.

## Repository Layout

```text
.
|- datasets.py            # Dataset wrappers for labeled/unlabeled splits and transforms
|- utils.py               # Evaluation utilities (accuracy, weighted F1, dataset stats)
|- wideresnet2.py         # WideResNet implementation used across experiments
|- CIFAR-10/
|  |- Full_Sup.ipynb
|  |- Sup_labeled_only.ipynb
|  |- MixMatch.ipynb
|  |- FixMatch.ipynb
|  |- FixMatch_mu_1.ipynb
|  |- SequenceMatch.ipynb
|  |- SequenceMatch_mu_1.ipynb
|  `- Results.ipynb
`- SVHN/
	 |- Full_Sup.ipynb
	 |- MixMatch.ipynb
	 |- FixMatch.ipynb
	 |- FixMatch_mu_1.ipynb
	 |- SequenceMatch.ipynb
	 |- SequenceMatch_mu_1.ipynb
	 `- Results.ipynb
```

## Experimental Setup (At a Glance)

- Backbone: WideResNet-28-2
- Datasets: CIFAR-10, SVHN
- Metrics: weighted F1, accuracy
- Typical label regimes observed in notebooks:
	- CIFAR-10: 40 labeled examples
	- SVHN: 250 labeled examples
- Unlabeled ratio $\mu$:
	- Standard runs: $\mu=7$
	- Ablation runs: $\mu=1$

## Quick Start

### 1) Create environment

```bash
conda create -n ssl-eval python=3.11 -y
conda activate ssl-eval
pip install -r requirements.txt
```

### 2) Launch notebooks

```bash
jupyter lab
```

### 3) Run experiments

Run notebooks inside either dataset folder:

- `CIFAR-10/*.ipynb`
- `SVHN/*.ipynb`

Suggested order per dataset:

1. `Full_Sup.ipynb`
2. `Sup_labeled_only.ipynb` (CIFAR-10)
3. `MixMatch.ipynb`
4. `FixMatch.ipynb`
5. `FixMatch_mu_1.ipynb`
6. `SequenceMatch.ipynb`
7. `SequenceMatch_mu_1.ipynb`
8. `Results.ipynb`

## Outputs and Logging

Each training notebook stores JSON traces (for example under `results/...`) containing budget, test F1, and test accuracy, which are later aggregated by `Results.ipynb`.

If you rerun experiments, keep run naming consistent so aggregation notebooks can discover all runs.

## Reproducibility Notes

- Use the same random seed policy across methods.
- Keep augmentation definitions unchanged when comparing methods.
- Compare methods at matched budget points, not only final epoch.
- Torch CUDA wheels in `requirements.txt` are hardware-dependent; adjust if your local CUDA/runtime differs.

## Workshop Context

This repository is intended as the reproducibility package for a workshop paper.

If you want to include publication metadata, update this section with:

- workshop/conference name
- paper title
- authors
- arXiv/OpenReview/DOI link

## Citation

```bibtex
@misc{rethinking_ssl_evaluation_2026,
	title  = {Rethinking SSL Evaluation},
	author = {Anonymous},
	year   = {2026},
	note   = {Workshop companion repository}
}
```

## License

This project is released under the Apache-2.0 License. See `LICENSE`.
