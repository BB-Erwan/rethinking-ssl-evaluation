# Roadmap - Future Extensions

This document outlines potential extensions of the "Rethinking SSL Evaluation" project to broaden its experimental scope and further improve fairness in comparisons between semi-supervised methods.

## 1. Evaluate More Methods

Goal: broaden the set of algorithms evaluated under the same budget-aware protocol.

Proposed actions:
- Integrate additional SSL families (pseudo-labeling, consistency regularization, teacher-student, diffusion-based SSL, etc.).
- Add recent literature baselines (official implementations or carefully documented reproductions).
- Keep a unified protocol: same budget, same reference backbone, same metrics, same seed policy.

Success criterion: compare a wider range of methods without introducing configuration bias.

## 2. Evaluate More Datasets

Goal: test whether conclusions remain robust beyond CIFAR-10, SVHN, and STL-10.

Proposed actions:
- Add datasets with different difficulty profiles (resolution, number of classes, noise level, class imbalance).
- Define comparable label-budget regimes per dataset (low, medium, high) to preserve aligned comparison points.
- Document preprocessing and augmentation choices per dataset to ensure reproducibility.

Success criterion: verify whether performance and cost trends are stable across data domains.

## 3. Evaluate Additional Architectures

Goal: assess to what extent conclusions depend on WideResNet-28-2.

Proposed actions:
- Test alternative CNN architectures (ResNet, EfficientNet, ConvNeXt, etc.) and, when relevant, ViT-style backbones.
- Harmonize model capacity across compared architectures (parameter count, approximate FLOPs) to avoid conflating architecture effects with method effects.
- Report method sensitivity to architecture changes under the same budget conditions.

Success criterion: determine whether SSL method rankings remain stable or are architecture-dependent.

## 4. Further Refine Fair Comparison

Goal: strengthen fair evaluation of new methods based on actual efficiency.

Proposed actions:
- Define stricter fairness protocols (compute control, number of updates, effective batch size, wall-clock time, and energy when possible).
- Evaluate methods using "performance vs cost" curves (accuracy/F1 vs label budget, time, and compute).
- Include variance-focused analysis (more seeds, confidence intervals, statistical tests) to separate true gains from noise.
- Establish a standard evaluation checklist so future methods are compared transparently and reproducibly.

Success criterion: provide an evaluation framework that minimizes bias and highlights truly efficient methods under comparable conditions.

## Suggested Prioritization

1. Add more methods within the current protocol (fastest impact).
2. Extend to additional datasets (external validation of conclusions).
3. Extend to additional architectures (result generalization).
4. Formalize a stricter fairness protocol as a long-term project standard.
