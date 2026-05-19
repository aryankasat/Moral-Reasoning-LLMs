# Adversarial Moral Dilemmas Framework

## Overview

This repository extension introduces the **Adversarial Moral Dilemmas Framework**. The goal of this framework is to evaluate whether Large Language Models (LLMs) rely on rhetorically sophisticated language (high-stage Kohlberg reasoning) as a crutch to justify logically incorrect or morally harmful actions.

To achieve this, we designed ~20-30 adversarial dilemma pairs. Each pair contains:

1. **Base Dilemma:** A standard moral dilemma (e.g., the Trolley Problem).
2. **Adversarial Dilemma:** A modified version where high-stage moral rhetoric (e.g., appeals to universal rights, categorical imperatives, or social contracts) is injected into the prompt to push the model towards a logically flawed or harmful action.

## 1. Generating Dilemmas

The dilemmas are dynamically generated using NVIDIA's `nemotron-3-super-120b-a12b` model via the NIM API.

- **Script:** `generate_dilemmas.py`
- **Output:** `dilemmas.csv`

The resulting CSV is formatted for direct piloting with human participants to establish baseline human susceptibility to the rhetorical injections before evaluating LLMs.

## 2. Multi-Factorial Cross-Pair Coherence

Standard correlation only tracks relative linear shifts. Instead, we compute **Cross-Pair Coherence** across the entire multi-dimensional evaluation grid (13 Models $\times$ 3 Prompts $\times$ 6 Dilemmas) to measure absolute structural invariance.

We partition the reasoning variance via ANOVA to calculate the **Intraclass Correlation Coefficient (ICC)**, mathematically isolating whether variance is driven by the Model's identity or the Contextual Pairing.

- **Scripts:** `score_coherence.py` and `complex_statistical_tests.py`
- **Outputs:** `results/coherence_scores.csv` and `results/icc_coherence_results.json`
- **Metrics Computed:**
  1. **Global ICC2:** Measures absolute agreement across the grid.
  2. **Variance Proportions:** Precisely calculates the percentage of variance attributed to Model Identity ($\sigma^2_{\text{model}}$) vs. Contextual Nuance ($\sigma^2_{\text{context}}$) vs. Noise ($\sigma^2_{\text{error}}$).
  3. **Lexical Coherence:** Grouped TF-IDF semantic consistency across dilemma pairings.

## 3. Formal Statistical & Psychometric Testing

To ensure mathematical rigor, we implemented complex statistical evaluations designed for academic publications:

- **Scripts:** `statistical_tests.py` and `complex_statistical_tests.py`
- **Methods Used:**
  - **Cronbach's Alpha:** Measures internal consistency of the moral testing suite.
  - **Kruskal-Wallis H-Test:** A non-parametric test evaluating if a model's moral stage shifts significantly depending on the dilemma ($p < 0.05$ indicates incoherence).
  - **Normalized Shannon Entropy:** Measures the predictability of a model's moral stage. High entropy = functional randomness in ethical decision-making.
  - **Principal Component Analysis (PCA):** Reduces dimensionality to test for *unidimensionality* vs. *multidimensionality* in moral frameworks.
  - **Corrected Item-Total Correlation (CITC):** Identifies which specific scenarios break the internal coherence of the models the most.

## 4. Visualizations

The results are compiled into 9 publication-ready, premium graphs stored in `results/`, utilizing rigorous academic styling and colorblind-friendly scientific palettes:

- **Fig 1 (Kruskal Forest Plot):** Logarithmic p-value scatter marking statistically coherent vs. incoherent models.
- **Fig 2 (Stage Distributions):** Faceted KDE ridge plots showing probability densities of reasoning stages.
- **Fig 3 (Entropy Jointplot):** Correlates predictability (Shannon Entropy) with statistical significance via marginal density axes.
- **Fig 4 (PCA Scree Plot):** Tests for a unified moral construct by charting filled explained variance areas.
- **Fig 5 (Model Clustermap):** A correlation heatmap identifying models with highly similar ethical structures.
- **Fig 6 (PCA Biplot):** A complex vector plot showing both the model score locations and the specific dilemmas driving their variance.
- **Fig 7 (Variance Partitioning):** A 100% stacked horizontal bar chart breaking down the exact proportions of ICC variance.
- **Fig 8a (Absolute Clustermap):** A hierarchical `viridis` clustermap grouping models that share underlying moral topologies.
- **Fig 8b (Deviation Matrix):** A diverging `vlag` heatmap mapping every cell's deviation from its model's mean stage, providing striking visual proof of massive contextual variance.

## Usage

To reproduce the entire framework, run the pipeline scripts in order from within the `adversarial_dilemmas` directory:

```bash
# 1. Generate Dilemmas
python generate_dilemmas.py

# 2. Score Local Coherence
python score_coherence.py
python visualize_results.py

# 3. Advanced Statistical Tests
python statistical_tests.py
python visualize_statistics.py

# 4. Information-Theoretic, Dimensional & ICC Analysis
python complex_statistical_tests.py
python visualize_advanced.py
python visualize_icc.py
```
