# Analysis 6 — Linguistic Reasoning Patterns (TF-IDF + PCA)

> **Research Question**: What language patterns do different models use when reasoning morally, and do model families share distinct "linguistic voices"?

---

## Motivation

Beyond *what stage* a model reasons at, *how* it expresses that reasoning matters. Do models at the same stage use the same moral vocabulary? Do model families (e.g., all Alibaba Qwen models) share a distinctive linguistic style? This analysis uses NLP techniques — TF-IDF keyword extraction, vocabulary richness metrics, and PCA dimensionality reduction — to map the linguistic landscape of moral reasoning across models.

---

## Statistical Methods

### 1. Stage-Level Vocabulary Analysis (`analyze_stage_vocabulary`)
- TF-IDF vectorisation per Kohlberg stage (all responses pooled)
- Top 20 distinctive terms per stage, extracted by mean TF-IDF weight
- Adaptive `min_df` for small corpora (≤10 docs → min_df=1)
- Uni- and bigram features (ngram_range=(1,2))

### 2. Model-Level Distinctive Terms (`analyze_model_distinctive_terms`)
- Top 5 TF-IDF-weighted terms per model
- Custom stop word list to filter moral-reasoning boilerplate

### 3. Target Keyword Usage (`evaluate_target_keyword_usage`)
- Predefined keyword lists for each Kohlberg stage (e.g., Stage 5: "rights," "contract," "greatest good")
- % of responses at each stage that contain expected keywords
- Heatmap of keyword presence across models × stages

### 4. Linguistic PCA (`compute_linguistic_pca`)
- One TF-IDF document per model (concatenated responses)
- PCA to 2 components → linguistic similarity space
- Explained variance reported per component
- Models plotted by provider, revealing family clustering

### 5. Qualitative Exemplars (`find_qualitative_exemplars`)
- TF-IDF centroid computed for each model's modal-stage responses
- Top 3 most representative (centroid-proximal) quotes identified per model
- Cosine similarity score and vocabulary richness reported

---

## Outputs

| File | Description |
|---|---|
| `fig1_stage_word_clouds.png` | Word clouds for each Kohlberg stage |
| `fig2_model_distinctive_terms.png` | Top distinctive terms per model |
| `fig3_moral_vocabulary_richness.png` | Vocabulary richness metrics per model |
| `fig4_target_keyword_heatmap.png` | Keyword hit rate across models × stages |
| `fig5_pca_linguistic_style.png` | PCA scatter of model linguistic styles |
| `distinctive_terms_by_model.csv` | Top 5 terms + weights per model |
| `target_keyword_usage.csv` | Keyword hit percentages |
| `linguistic_pca_coordinates.csv` | PCA coordinates for each model |
| `qualitative_exemplars.csv` | Representative quotes per model |

---

## Key Findings

- Model families share distinct "linguistic voices" — PCA clusters models by provider
- RLHF-aligned models demonstrate richer moral vocabulary regardless of size
- Stage 5–6 responses show higher semantic density and more abstract vocabulary
- Some smaller models achieve high stages using formulaic keyword patterns rather than genuine argumentative structure

## Usage

```bash
python analysis6/main.py
```

Requires: `scikit-learn` (TF-IDF, PCA), `wordcloud`
