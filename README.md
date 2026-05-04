# GeoFM — Predicting Pasture-to-Soy Conversion in the Brazilian Cerrado

[![OSF](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/c46je)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**Institution:** TNC Brasil  
**Author:** Mario Barroso Ramos Neto  
**OSF Pre-registration:** https://osf.io/c46je  
**Status:** ✅ COMPLETE — etapa13b (Multi-State Survival) is the final model  
**Continuation:** [lulc-fm-brazil](https://github.com/barroso2501/lulc-fm-brazil) — GeoFM pre-training on all Brazilian biomes

---

## Overview

This repository contains the full pipeline for predicting pasture-to-soy land conversion in the Brazilian Cerrado using MapBiomas LULC time series (1985–2024). The project combines deep learning with ecological domain knowledge, following rigorous open science practices including iterative OSF pre-registration.

The central methodological contribution is the demonstration that **label definition anchored in ecological domain knowledge** (5-year threshold based on the limpeza cycle) is more important than architectural complexity for model performance. The project also implements a Weibull survival model that extends binary classification to continuous-time prediction while incorporating censored data.

---

## Key Results

All classification models evaluated under hexagon-stratified spatial split (0% geographic overlap between train and test sets), Dataset v2 (5-year threshold).

| Model | Test F1 | Test AUC | Dataset | Note |
|-------|:-------:|:--------:|:-------:|------|
| Baseline MLP (etapa4) | 0.593 | 0.655 | v1 | 10-year threshold |
| GeoFM v3b (etapa6b) | 0.596 | 0.668 | v1 | Spatial attention |
| **Baseline MLP (etapa7b)** | **0.862** | **0.840** | **v2** | **5-year threshold** |
| TwoChannel (etapa7) | 0.858 | 0.849 | v2 | Routing by pattern_T |
| Weibull Survival (etapa8) | 0.774* | 0.846 | v2+ | *Not F1-optimized |
| Weibull Regeneration (etapa9) | — | **0.910** | PN | P→N, AUC only |
| **Competing Risks (etapa10)** | — | **P→S: 0.833 / P→N: 0.933** | v2+PN | r(P→S,P→N)=-0.911 |

> **Key finding:** The improvement from v1 to v2 results (F1 0.59 → 0.86) is entirely attributable to the ecological label definition (5-year threshold), not to architectural changes. A simple MLP on dataset v2 matches the TwoChannel architecture, confirming that label quality is the dominant factor.

> **Weibull note:** The F1=0.774 reflects threshold=0.5 applied to survival probabilities. The model was trained with Weibull log-likelihood (not F1), making direct F1 comparison misleading. AUC=0.846 is the appropriate metric.


## Prospective Validation

The Weibull survival model (etapa8) was prospectively validated against MapBiomas 2019–2024 data — a period entirely independent of the training data.

**Design:** 3,913 pasture pixels identified by the model as candidates for soybean conversion (not yet converted at training time) were checked against MapBiomas annual rasters 2019–2024.

| Risk Category | N pixels | Converted (2019-2024) | Precision |
|:-------------:|:--------:|:--------------------:|:---------:|
| CRITICAL | 99 | 99 | **100%** |
| HIGH | 573 | 573 | **100%** |
| MODERATE | 567 | 567 | **100%** |
| LOW | 2,674 | 2,652 | 99.2% |
| **TOTAL** | **3,913** | **3,891** | **99.4%** |

Every pixel flagged as CRITICAL, HIGH, or MODERATE was mapped as soybean (class 39) by MapBiomas in 2019–2024. The model was not retrained or adjusted after observing these outcomes.

> **Important caveat:** Target pixels are a pre-selected high-risk sample (pasture pixels without observed conversion in the training period). The 99.4% conversion rate should not be interpreted as a population-level conversion rate for all Cerrado pasture. Field validation is required before operational deployment at landscape scale.


## Competing Risks Model

The project culminates in a competing risks framework that jointly models two opposing pasture transitions:

**P→S (conversion to soybean):** driven by agricultural expansion pressure, requires intensive root removal (destoca). AUC=0.833 in competing model.

**P→N (regeneration to native vegetation):** enabled by Cerrado's underground woody structures (xylopodia) that persist after surface conversion. AUC=0.933 in competing model — improved from 0.910 (individual model).

**Key finding — negative feedback confirmed:** The correlation between P(P→S) and P(P→N) per pixel is **r = -0.911**. This confirms the ecological hypothesis that conversion and regeneration are mechanistically opposed: pixels under active conversion pressure have suppressed regeneration probability, and pixels with intact underground structures have low conversion risk. The shared encoder learned this structure from data without it being explicitly programmed.

```
Shared Encoder (290→256→128)
       ├── Head P→S → (k_S, lambda_S) → CIF_S(t)
       └── Head P→N → (k_N, lambda_N) → CIF_N(t)

k_S = 1.79  lambda_S = 15.8 yr  (moderate increasing hazard)
k_N = 4.09  lambda_N = 34.6 yr  (strongly increasing, slow onset)
```

> **Caveat:** k and lambda are predictive parameters minimizing Weibull log-likelihood, not direct measurements of ecological processes.

## Multi-State Survival Model (etapa13b — final)

The final model incorporates a third state: **P→P (stable pasture)** — pixels that remained in pasture for 20+ years without any observed transition. These pixels are included as doubly-censored observations with their real observation time (up to 38 years) and explicit weighting (weight_PP=2.0).

**Why weight_PP=2.0:** A pixel with 30 years of uninterrupted pasture is stronger evidence of low exit rate than a standard censored pixel with 5 years. The weighting encodes this difference explicitly.

**Effect on parameters:**
- lambda_S: 15.82 → 25.91 years (+10 years — more realistic)
- lambda_N: 34.59 → 44.77 years (+10 years — more realistic)

**Three-cluster structure (emerges from data):**
- High P(P→S) / Low P(P→N): active conversion pressure
- Low P(P→S) / High P(P→N): incomplete conversion, regeneration potential
- Low P(P→S) / Low P(P→N): structural stability — new cluster absent in etapa10

**Foundation model rationale:** weight_PP=2.0 was chosen over maximizing AUC because calibrated lambda values produce representations that generalize better to other biomes. The Weibull multi-state task is a pre-training pretext for the GeoFM, not the final product.

---

## Repository Structure

```
geofm-cerrado-github/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_etapa1_sampling.ipynb
│   │     Initial sampling from classified raster. Temporal split.
│   │
│   ├── 02_etapa2_spatial_patterns_multihead.ipynb
│   │     Hexagonal pattern integration (CC*/CD*). Stratified resampling.
│   │
│   ├── 03_etapa3_multihead_random_split.ipynb  ⚠️ SUPERSEDED
│   │     Multi-Head with random split. Results invalidated — OSF Update 4.
│   │
│   ├── 04_etapa4_baseline_spatial_split.ipynb
│   │     Baseline MLP with hexagon-stratified spatial split. Valid baseline.
│   │
│   ├── 05_etapa5_multihead_spatial_split.ipynb
│   │     Multi-Head with same spatial split as etapa4.
│   │
│   ├── 06_error_analysis.ipynb
│   │     Error analysis on etapa5 test set. Reveals FN patterns by T,
│   │     spatial pattern, and geography.
│   │
│   ├── 07_etapa6_geofmv3_spatial_attention.ipynb
│   │     GeoFM v3: spatial attention module (Option A). Attention collapsed
│   │     to uniform — performance gain from per-frame encoding, not attention.
│   │
│   ├── 08_etapa6b_geofmv3b_attention_corrected.ipynb
│   │     GeoFM v3b: cross-attention with learned query + entropy
│   │     regularization. Best result on dataset v1 (AUC=0.668).
│   │
│   ├── 09_attention_visualization.ipynb
│   │     Diagnostic: spatial and temporal attention weight analysis.
│   │     Confirmed uniform attention — patch is predominantly homogeneous.
│   │
│   ├── 10_fn_tn_diagnostic.ipynb
│   │     FN vs TN patch comparison. Signal weak but present (2/5 metrics
│   │     significant). Error partially epistemic.
│   │
│   ├── 11_dataset_v2_generation.ipynb
│   │     Dataset v2: 5-year ecological threshold, classes 21/41 as
│   │     transition states, recalculated from scratch.
│   │
│   ├── 12_hypothesis_cluster_rapido.ipynb
│   │     Tests CC* -> rapid conversion hypothesis. Confirmed: +12.6pp
│   │     separation (Cohen's h=0.260) with static Class8590.
│   │
│   ├── 13_temporal_pattern_join.ipynb
│   │     Contemporaneous pattern_T per pixel based on T. 70% reclassified.
│   │     Separation increases to +17.1pp (Cohen's h=0.361).
│   │
│   ├── 14_etapa7_twochannel.ipynb
│   │     Two-Channel architecture with deterministic routing by pattern_T
│   │     and informed gate prior. Performance equivalent to baseline on v2.
│   │
│   ├── 15_etapa7b_baseline_control.ipynb
│   │     Control experiment: baseline MLP on dataset v2. Confirms that
│   │     all gain comes from dataset (label definition), not architecture.
│   │
│   ├── 16_comparative_error_analysis.ipynb
│   │     Baseline vs TwoChannel error comparison on v2 test set.
│   │     BOTH_WRONG analysis reveals epistemic ceiling.
│   │
│   ├── 17_survival_analysis_classical.ipynb
│   │     Classical Weibull survival analysis by ecological group.
│   │     All groups k<1 (decreasing hazard) — heterogeneity non-observed.
│   │     Motivates neural survival model with per-pixel k and lambda.
│   │
│   ├── 18_etapa8_weibull_survival.ipynb
│   │     Neural survival model with Weibull head. Per-pixel (k, lambda)
│   │     learned from series. Censored pixels included in training.
│   │     AUC=0.846, predicts P(<=t) for any horizon without retraining.
│   │
│   └── 19_weibull_analysis.ipynb
│         Analysis of Weibull parameters by ecological group.
│         k and lambda as predictive parameters, not direct ecological
│         measurements — interpretation requires independent validation.
│
├── scripts/
│   └── spatial_split.py
│         Standalone script for hexagon-stratified spatial split.
│         No external dependencies beyond numpy and pandas.
│
└── results/
    ├── spatial_split_metadata.json
    ├── predictions_test.csv          (etapa5 test set predictions)
    ├── comparative_predictions.csv   (etapa7/7b comparison)
    ├── dataset_v2/
    │   ├── dataset_v2_metadata.json
    │   ├── dataset_v2_distribution.png
    │   ├── hypothesis_cluster_rapido.png
    │   └── pattern_comparison_static_vs_temporal.png
    ├── geofmv3b/
    │   └── results_*.json
    └── survival_analysis/
        ├── kaplan_meier.png
        ├── weibull_fit.png
        └── survival_analysis_results.json
```

---

## Data

### Required (not included — external sources)

| File | Source | Description |
|------|--------|-------------|
| `brazil_coverage_{year}_Cerrado.tif` | [MapBiomas Collection 10](https://mapbiomas.org) | Annual LULC rasters 1985–2024, 30m |
| `hex_Cerrado_class_Change.shp` | TNC Brasil | Hexagonal grid with spatial pattern codes for 7 periods |

### Datasets and Model Weights (Zenodo)

| Resource | DOI |
|----------|-----|
| Dataset v1 (treino_balanceado_FINAL.csv) | [10.5281/zenodo.19021498](https://doi.org/10.5281/zenodo.19021498) |
| Model checkpoints | [10.5281/zenodo.19021415](https://doi.org/10.5281/zenodo.19021415) |

---

## Installation

```bash
git clone https://github.com/barroso2501/geofm-cerrado-github.git
cd geofm-cerrado-github
pip install -r requirements.txt
```

Update path constants at the top of each notebook:

```python
DATA_DIR  = r"D:\Projetos\Cerrado\LULC"
BASE_DIR  = r"D:\Projetos\Cerrado\GeoFM_sampling"
```

---

## Dataset v2 — Ecological Label Definition

The critical methodological decision: defining the label based on the **ecological mechanism**, not statistical convenience.

```
Label=1 (rapid): Pixel reaches class 39 (soy) within <=5 years of T
                 Classes 21 (mosaic) and 41 (other crops) accepted as
                 transition states (MapBiomas classification artifact)
Label=0 (slow):  Pixel does not reach class 39 within 5 years of T
Censored:        T > 2019 (less than 5 years observation before 2024)
T:               First year with class 15 (pasture) after native vegetation

Ecological basis: The N->P->S conversion requires intensive root removal
(limpeza). Capitalized farmers complete this in 12-18 months; typical
duration is 2-4 years; 5 years is the maximum for an intentional process.
Conversions beyond 5 years reflect a qualitatively different mechanism.
```

**Impact:** Moving from 10-year to 5-year threshold improved test F1 from 0.593 to 0.862. This improvement is entirely from label quality, not model architecture.

---

## Weibull Survival Model (etapa8)

The Weibull survival model extends binary classification in three ways:

**Continuous-time prediction:** P(convert in ≤t years) for any t without retraining. Operationally useful for monitoring at different time horizons.

**Censored data:** 7,333 pixels excluded from classification (insufficient observation time) contribute to survival model training via right-censored likelihood.

**Per-pixel heterogeneity:** Each pixel gets its own (k, λ) parameters. k varies by pixel (std=0.660), capturing individual risk profiles beyond group-level patterns.

**Important caveat:** k and λ are parameters learned to minimize Weibull log-likelihood, not direct measurements of ecological processes. The relationship between these parameters and specific ecological mechanisms (e.g., pressure accumulation in active frontiers) requires independent validation and should be interpreted with caution.

```python
# Load and use the survival model
checkpoint = torch.load('weibull_survival_*.pth', map_location='cpu')
model = WeibullSurvivalModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict P(convert in <=t years) for any t
with torch.no_grad():
    p_5yr  = model.conversion_prob(features, t=5.0)   # <=5 years
    p_10yr = model.conversion_prob(features, t=10.0)  # <=10 years
```

---

## Experimental History

| Stage | Notebook | Model | Split | Dataset | Status |
|-------|----------|-------|-------|---------|--------|
| etapa1 | `01` | Logistic Regression | Temporal | v1 | ✅ Frozen |
| etapa2 | `02` | MLP Embedding | Spatial block N/S | v1 | ✅ Frozen |
| etapa3 | `03` | Multi-Head | Random pixel | v1 | ⚠️ Superseded |
| etapa4 | `04` | Baseline MLP | Hexagon-stratified | v1 | ✅ Frozen |
| etapa5 | `05` | Multi-Head | Hexagon-stratified | v1 | ✅ Frozen |
| etapa6 | `07` | GeoFM v3 (spatial attn) | Hexagon-stratified | v1 | ⚠️ Superseded |
| etapa6b | `08` | GeoFM v3b (cross-attn) | Hexagon-stratified | v1 | ✅ Frozen |
| etapa7 | `14` | TwoChannel | Hexagon-stratified | v2 | ✅ Frozen |
| etapa7b | `15` | Baseline MLP (control) | Hexagon-stratified | v2 | ✅ Frozen |
| etapa8 | `18` | Weibull Survival | Hexagon-stratified | v2+ | ✅ Frozen |
| etapa9 | `23` | Weibull P→N (Regeneration) | Hexagon-stratified | PN | ✅ Frozen |
| etapa10 | `24` | Competing Risks (P→S + P→N) | Hexagon-stratified | v2+PN | ✅ Frozen |
| etapa11 | `25` | Dataset P→P (stable pasture) | Hexagon-stratified | PP | ✅ Frozen |
| etapa13b | `26` | Multi-State (P→S + P→N + P→P) | Hexagon-stratified | v2+PN+PP | ✅ Frozen |

### OSF Pre-registration updates

| Update | Date | Content |
|--------|------|---------|
| Update 1 | Mar 8, 2026 | Initial registration |
| Update 2 | Mar 11, 2026 | Multi-Head frozen (etapa3) |
| Update 3 | Mar 14, 2026 | Gate weight analysis — implicit ensemble |
| Update 4 | Mar 16, 2026 | **Critical correction** — validation regime incompatibility |
| Update 5 | Mar 18, 2026 | Experimental phase v2 closure |
| Update 6 | Apr 24, 2026 | Dataset v2 + two-channel architectural direction |
| Update 7 | Apr 26, 2026 | Weibull survival model — etapa8 results |
| Update 8 | Apr 28, 2026 | Prospective validation 99.4% precision |
| Update 9 | Apr 30, 2026 | Weibull P→N (etapa9) + Competing Risks (etapa10) |
| Update 10 | May 1, 2026 | Multi-State etapa13b (P→S + P→N + P→P, weight_PP=2.0) |

---

## Key Findings

### What the project learned

**Label quality dominates architecture.** The single most impactful decision was changing the conversion threshold from 10 to 5 years, based on knowledge of the limpeza process. This improved F1 from 0.593 to 0.862 — far more than any architectural change.

**Spatial validation is essential.** Random pixel-level splits produced 72% geographic overlap between train and test, inflating apparent performance. Hexagon-stratified splits with 0% overlap are necessary for valid generalization claims.

**The epistemic ceiling.** Error analysis identified pixels that neither model can predict: T=2010-2015 (short observation window before T), CD* pattern (individual mechanism without collective signal), and late conversions (T+4, T+5 years). These are genuine limits of what the available data can predict.

**Attention does not help with homogeneous patches.** The 7×7 patch is predominantly pasture class 15 in most positions. Attention mechanisms collapse to uniform weights because there is insufficient spatial gradient to discriminate positions. Performance gains in v3/v3b come from per-frame encoding, not attention.

**Weibull survival extends, not replaces, classification.** AUC is equivalent (0.846 vs 0.840 for baseline), but the survival model adds continuous-time prediction and incorporates censored data. The Weibull parameters (k, λ) are predictive artifacts, not direct ecological measurements.

---

## Continuation — lulc-fm-brazil

This repository is **complete**. The Cerrado Pilot demonstrated that MapBiomas LULC time series contain sufficient signal for ecological process prediction at the pixel level, with results validated prospectively (99.4% precision) and documented across 10 OSF pre-registration updates.

The next phase of development is in a separate repository:

**[lulc-fm-brazil](https://github.com/barroso2501/lulc-fm-brazil)** — a geospatial foundation model for land use and land cover dynamics in Brazil, pre-trained on 40 years of MapBiomas LULC time series across all 6 Brazilian biomes.

The Cerrado Pilot contributes three things to that project:

- **Proof of concept:** AUC=0.854 (P→S) and 99.4% prospective precision confirm that the signal exists and is learnable from LULC time series alone
- **Downstream benchmark tasks:** P→S, P→N, P→P are the fine-tuning tasks that evaluate whether GeoFM representations transfer from continental pre-training to Cerrado-specific prediction
- **Encoder baseline:** The etapa13b encoder (weight_PP=2.0, λ_S=25.9yr, λ_N=44.8yr) is the reference representation for fine-tuning evaluation

---


## lulc-fm-brazil — GeoFM Foundation Model

The [lulc-fm-brazil](https://github.com/barroso2501/lulc-fm-brazil) repository continues this project as a geospatial foundation model pre-trained on 40 years of MapBiomas LULC time series across all Brazilian biomes.

### Phase 1 — Sampling (complete)
- 80 cells of 2×2 degrees, stratified by biome
- 4,067,807,469 pixels extracted across 6 biomes
- 11 ecological processes decomposed by mechanism
- 6 aggregated classes: N, P, A, U, W, T

### Phase 2A — Pre-training (complete)
- Architecture: Temporal Transformer (2 layers, 4 heads, dim=128)
- Objective: masked temporal prediction (block masking, 5 years)
- Result: val_loss=0.15, val_acc=0.95 — 2x better than MLP baseline
- Training time: ~2 hours (6x faster than etapa13b due to RAM pipeline)

### Phase 2B — Evaluation (partial)
- Criterion 3 (latent space): Silhouette=0.528 ✅ STRONG (>0.40)
- Criterion 1 (few-shot): 5/8 processes encoder > baseline ✅ MAJORITY
- Criterion 4 (Cerrado Pilot): pending dedicated script
- Criterion 2 (cross-biome): Phase 3

## Future Directions

**Enriching pre-T signal:** The main epistemic barrier is insufficient signal before T for CD* pixels and recent T values. External data (distance to active agricultural frontier, road infrastructure, land tenure, soy price history) could address this.

**Survival model refinement:** A frailty model (random effects per pixel) would better handle unobserved heterogeneity. Per-group Weibull initialization based on classical survival analysis results could improve convergence.

**Foundation model potential:** MapBiomas LULC time series cover all of Brazil (1985–2024) at 30m resolution with a consistent 30-class vocabulary. Self-supervised pre-training on continental-scale transition sequences could learn general land-use trajectory representations, transferable to other biomes and transition types.

---

## Citation

```bibtex
@misc{barroso2026geofm,
  author    = {Barroso Ramos Neto, Mario},
  title     = {GeoFM: Predicting Pasture-to-Soy Conversion
               in the Brazilian Cerrado},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/barroso2501/geofm-cerrado-github},
  note      = {OSF pre-registration: https://osf.io/c46je}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
MapBiomas Collection 10 data is subject to [MapBiomas terms of use](https://mapbiomas.org/en/terms-of-use) (CC-BY 4.0).
MIT License — see [LICENSE](LICENSE) for details.  
MapBiomas Collection 10 data is subject to [MapBiomas terms of use](https://mapbiomas.org/en/terms-of-use) (CC-BY 4.0).

MIT License — see [LICENSE](LICENSE) for details.  
MapBiomas Collection 10 data is subject to [MapBiomas terms of use](https://mapbiomas.org/en/terms-of-use) (CC-BY 4.0).
