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
|  |- Results.ipynb
|- SVHN/
|  |- Full_Sup.ipynb
|  |- Sup_labeled_only.ipynb
|  |- MixMatch.ipynb
|  |- FixMatch.ipynb
|  |- FixMatch_mu_1.ipynb
|  |- SequenceMatch.ipynb
|  |- SequenceMatch_mu_1.ipynb
|  |- Results.ipynb
|- STL-10/
   |- Sup_labeled_only.ipynb
   |- MixMatch.ipynb
   |- FixMatch.ipynb
   |- FixMatch_mu_1.ipynb
   |- SequenceMatch.ipynb
   |- SequenceMatch_mu_1.ipynb
   |- Results.ipynb
```

## Experimental Setup (At a Glance)

- Backbone: WideResNet-28-2
- Datasets: CIFAR-10, SVHN
- Metrics: weighted F1, accuracy
- Typical label regimes used in our experiments:
	- CIFAR-10: 40, 250, 4000 labeled examples
	- SVHN: 40, 250, 1000 labeled examples
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

## Dataset Normalization Statistics

Each notebook uses pre-computed per-channel mean and std for normalization. For CIFAR-10 the hardcoded values are:

```python
mean = torch.tensor([0.4914, 0.4822, 0.4465])
std  = torch.tensor([0.2023, 0.1994, 0.2010])
```

If you want to recompute these from the training split yourself, uncomment the three lines at the top of `CIFAR-10/FixMatch_mu_1.ipynb` (and analogous notebooks):

```python
from utils import compute_mean_std
loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
mean, std = compute_mean_std(loader)
```

`compute_mean_std` is implemented in `utils.py`. Replace the hardcoded tensors with the returned values before running any subsequent cells.

## Outputs and Logging

Pre-computed results are already available inside each dataset's `results/` folder:

- `CIFAR-10/results/` — runs for 40, 250, and 4000 labeled examples (3 runs each) plus full supervision
- `SVHN/results/` — runs for 40, 250, and 1000 labeled examples (3 runs each) plus full supervision
- `STL-10/results/` — runs for 40 and 1000 labeled examples (3 runs each)

Each JSON file contains per-iteration budget, test F1, and test accuracy traces. `Results.ipynb` in each dataset folder aggregates and plots these.

If you rerun experiments, keep run naming consistent so aggregation notebooks can discover all runs.

## Reproducibility Notes

- Use the same random seed policy across methods.
- Keep augmentation definitions unchanged when comparing methods.
- Compare methods at matched budget points, not only final epoch.
- Torch CUDA wheels in `requirements.txt` are hardware-dependent; adjust if your local CUDA/runtime differs.

## License

This project is released under the Apache-2.0 License. See `LICENSE`.
