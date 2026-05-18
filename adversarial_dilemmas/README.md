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

## 2. Cross-Pair Coherence Scoring
We analyzed the existing models across the repository to determine their baseline consistency when handling paired standard moral dilemmas. 
- **Script:** `score_coherence.py`
- **Output:** `results/coherence_scores.csv`
- **Metrics Computed:**
  1. **Lexical Coherence:** TF-IDF semantic consistency across dilemma responses.
  2. **Stage Coherence:** Standard deviation of the Kohlberg stages.

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
The results are compiled into 6 publication-ready, beautified graphs stored in `results/`.
- **Fig 1 (Kruskal Forest Plot):** Logarithmic p-value distributions marking statistically coherent vs. incoherent models.
- **Fig 2 (Stage Distributions):** Multi-panel violin plots corroborating the Kruskal-Wallis tests visually.
- **Fig 3 (Entropy Scatter):** Correlates predictability (Shannon Entropy) with statistical significance.
- **Fig 4 (PCA Scree Plot):** Tests for a unified moral construct by charting explained variance components.
- **Fig 5 (Hierarchical Clustermap):** A correlation heatmap identifying models with highly similar ethical structures.
- **Fig 6 (PCA Biplot):** A complex vector plot showing both the model score locations and the specific dilemmas driving their variance.

## Usage
To reproduce the entire framework, run the pipeline scripts in order:
```bash
# 1. Generate Dilemmas
python generate_dilemmas.py

# 2. Score Coherence Locally
python score_coherence.py
python visualize_results.py

# 3. Advanced Statistical Tests
python statistical_tests.py
python visualize_statistics.py

# 4. Information-Theoretic & Dimensional Analysis
python complex_statistical_tests.py
python visualize_advanced.py
```
