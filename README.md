# GeoFM — Predicting Pasture-to-Soy Conversion in the Brazilian Cerrado

[![OSF](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/c46je)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**Institution:** TNC Brasil  
**Author:** Mario Barroso Ramos Neto  
**OSF Pre-registration:** https://osf.io/c46je  
**Status:** Active — etapa7 (Two-Channel architecture) in development

---

## Overview

This repository contains the full pipeline for predicting pasture-to-soy land conversion in the Brazilian Cerrado using MapBiomas LULC time series (1985–2024). The project combines deep learning with ecological domain knowledge, following rigorous open science practices including iterative OSF pre-registration.

The project has gone through two dataset versions and multiple architectural experiments. Each stage is fully documented with frozen results and OSF updates. Current development focuses on a Two-Channel architecture with deterministic routing based on contemporaneous spatial conversion patterns.

---

## Key Results

All models evaluated under hexagon-stratified spatial split (0% geographic overlap between train and test sets).

| Model | Test Accuracy | Test F1 | Test AUC | Val–Test Gap | Dataset |
|-------|:---:|:---:|:---:|:---:|:---:|
| Baseline MLP (etapa4) | 61.3% | 0.593 | 0.655 | −3.7pp | v1 |
| Multi-Head v2 (etapa5) | 60.3% | 0.578 | 0.648 | −4.3pp | v1 |
| GeoFM v3 (etapa6) | 63.1% | 0.596 | 0.666 | −2.7pp | v1 |
| **GeoFM v3b (etapa6b)** | **62.5%** | **0.596** | **0.668** | **−2.2pp** | v1 |
| etapa7 (planned) | — | — | — | — | v2 |

> **Note on etapa3:** An earlier experiment reported 68.2% accuracy for the Multi-Head model. This comparison was invalidated by incompatible validation regimes (see OSF Update 4). Etapa3 is retained for audit purposes only.

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
│   │     Error analysis on etapa5 test set. Identifies FN patterns by T,
│   │     spatial pattern, and geography.
│   │
│   ├── 07_etapa6_geofmv3_spatial_attention.ipynb
│   │     GeoFM v3: spatial attention module (Option A: separate spatial
│   │     then temporal). Attention collapsed to uniform — diagnosed.
│   │
│   ├── 08_etapa6b_geofmv3b_attention_corrected.ipynb
│   │     GeoFM v3b: cross-attention with learned query + entropy
│   │     regularization. Attention still uniform — performance gain
│   │     comes from per-frame spatial encoding, not attention mechanism.
│   │
│   ├── 09_attention_visualization.ipynb
│   │     Diagnostic: spatial and temporal attention weight analysis.
│   │     Confirmed uniform attention in both v3 and v3b.
│   │
│   ├── 10_fn_tn_diagnostic.ipynb
│   │     Diagnostic: FN vs TN patch comparison. Tests whether conversion
│   │     signal exists in data before T. Result: signal weak but present
│   │     (2/5 metrics significant). Error partially epistemic.
│   │
│   ├── 11_dataset_v2_generation.ipynb
│   │     Dataset v2: recalculated from scratch with 5-year ecological
│   │     threshold. Classes 21/41 accepted as transition states.
│   │
│   ├── 12_hypothesis_cluster_rapido.ipynb
│   │     Tests hypothesis: CC* patterns -> rapid conversion,
│   │     CD*/NC -> slow. Confirmed: +12.6pp separation (Cohen's h=0.260).
│   │
│   └── 13_temporal_pattern_join.ipynb
│         Assigns contemporaneous spatial pattern (pattern_T) to each
│         pixel based on T. 70% of pixels reclassified vs static Class8590.
│         Separation increases to +17.1pp (Cohen's h=0.361).
│
├── scripts/
│   └── spatial_split.py
│         Standalone script for hexagon-stratified spatial split.
│         No external dependencies beyond numpy and pandas.
│
└── results/
    ├── spatial_split_metadata.json
    ├── predictions_test.csv          (etapa5 test set predictions)
    ├── dataset_v2/
    │   ├── dataset_v2_metadata.json
    │   ├── dataset_v2_distribution.png
    │   ├── hypothesis_cluster_rapido.png
    │   └── pattern_comparison_static_vs_temporal.png
    └── geofmv3b/
        ├── results_20260423_071506.json
        ├── spatial_weights_20260423_071506.npy
        └── temporal_weights_20260423_071506.npy
```

---

## Data

### Required (not included — external sources)

| File | Source | Description |
|------|--------|-------------|
| `brazil_coverage_{year}_Cerrado.tif` | [MapBiomas Collection 10](https://mapbiomas.org) | Annual LULC rasters 1985–2024, 30m |
| `hex_Cerrado_class_Change.shp` | TNC Brasil | Hexagonal grid with spatial pattern codes (CC*/CD*) for 7 periods |

### Generated (included)

| File | Description |
|------|-------------|
| `results/spatial_split_metadata.json` | Split: 6,152 train / 1,646 val / 1,505 test pixels |
| `results/predictions_test.csv` | etapa5 test predictions: prob, pred, error_type, gate weights |
| `results/dataset_v2/dataset_v2_metadata.json` | Dataset v2: 9,226 pixels, 5-year threshold, pattern_T |

### Datasets and Model Weights (Zenodo)

| Resource | DOI |
|----------|-----|
| Dataset v1 (treino_balanceado_FINAL.csv) | [10.5281/zenodo.19021498](https://doi.org/10.5281/zenodo.19021498) |
| Model checkpoints (.pth files) | [10.5281/zenodo.19021415](https://doi.org/10.5281/zenodo.19021415) |

---

## Installation

```bash
git clone https://github.com/barroso2501/geofm-cerrado-github.git
cd geofm-cerrado-github
pip install -r requirements.txt
```

Update path constants at the top of each notebook before running:

```python
DATA_DIR  = r"D:\Projetos\Cerrado\LULC"
BASE_DIR  = r"D:\Projetos\Cerrado\GeoFM_sampling"
SPLIT_DIR = BASE_DIR / "spatial_split"
```

---

## Running the Pipeline

### Full pipeline (from scratch)

```
01 → 02 → scripts/spatial_split.py → 04 → 05 → 06
         → 11 → 13 → 12            → etapa7 (planned)
```

### Dataset v2 generation

```bash
# After running 11_dataset_v2_generation.ipynb
# Run 13_temporal_pattern_join.ipynb to add pattern_T
# Run 12_hypothesis_cluster_rapido.ipynb to verify routing hypothesis
```

### Spatial split only

```bash
python scripts/spatial_split.py
```

---

## Feature Engineering

### Feature vector — Dataset v1 (287 dimensions, etapa4/5/6/6b)

```
X = [LULC_series (39), Spatial_patch (245), Auxiliary (3)]

LULC_series:   Annual class 1985 to T-1, left-padded to length 39
Spatial_patch: 7x7 neighborhood for 5 years before T, flattened (5x7x7=245)
Auxiliary:     [prop years class 21, consecutive pasture years, class at T-1]
```

### Feature vector — Dataset v2 (etapa7, planned)

```
LULC_series:   Same as v1 (39)
Spatial_patch: 7x7 neighborhood, 5 years (5x7x7) — NOT flattened for attention
Auxiliary:     [prop years class 21, consecutive pasture years, class at T-1,
                is_CC, is_CD, is_NC]  ← pattern_T context features added
```

---

## Dataset v2 — Ecological Threshold

**Label definition (5-year threshold):**

The 5-year threshold reflects the N→P→S conversion process in the Cerrado. Well-capitalized farmers complete root removal (limpeza) in 12–18 months; the typical duration is 2–4 years; 5 years is the maximum for an intentional planned conversion. Conversions beyond 5 years reflect a qualitatively different mechanism.

```
Label=1 (rapid):  Pixel reaches class 39 (soy) within <=5 years of T
                  Classes 21 (mosaic) and 41 (other crops) accepted as
                  transition states before class 39 appears
Label=0 (slow):   Pixel does not reach class 39 within 5 years of T
Censored:         T > 2019 (less than 5 years observation before 2024)
T:                First year with class 15 (pasture) after native vegetation
```

**Contemporaneous spatial pattern (pattern_T):**

Each pixel is assigned the hexagon conversion pattern from the period most contemporaneous and prior to T, selected from 7 available periods (Class8590 through Class1520). 70% of pixels have a different pattern_T compared to the static Class8590.

**Routing hypothesis (confirmed):**

```
CC* (cluster conversion):  73.3% rapid conversion
CD* (dispersed):           56.2% rapid conversion
NC  (no conversion):       55.7% rapid conversion

CC* vs CD* separation: +17.1pp  (p<0.0001, Cohen's h=0.361)
```

---

## Architecture

### GeoFM v3b — Current Best (etapa6b)

```
Input:
  serie (39) + aux (3) → Temporal Encoder (42→256→128) → vector (128)
  patch (5,7,7)        → SpatialAttentionV2             → vector (64)
                              ↓
                   Concatenate [128 + 64] = 192
                              ↓
                   Prediction head (192→64→1, sigmoid)

SpatialAttentionV2:
  Cross-attention with learned query (nn.Parameter)
  Separate key_proj and val_proj (no circular dependency)
  Position embedding per grid cell
  Entropy regularization (lambda=0.01)

Parameters: ~110,000
Note: Attention weights converged to uniform in both v3 and v3b.
      Performance gain over v2 comes from per-frame spatial encoding,
      not from the attention mechanism itself.
```

### etapa7 — Two-Channel with Informed Routing (planned)

```
pattern_T
    ↓
  CC*     → Channel Cluster  — learns frontier expansion dynamics
  CD*+NC  → Channel Stable   — learns slow/stable conversion dynamics
    ↓
Soft gate initialized with empirical log-odds (not random)
Context features is_CC, is_CD, is_NC injected in both channels

Ecological basis:
  CC*: coordinated multi-owner pressure, expansion wave mechanism
  CD*: individual decisions, dispersed signal
  NC:  stable area, equilibrium disruption mechanism
```

---

## Experimental History

| Stage | Notebook | Model | Split | Dataset | Status |
|-------|----------|-------|-------|---------|--------|
| etapa1 | `01_etapa1_sampling` | Logistic Regression | Temporal | v1 | ✅ Frozen |
| etapa2 | `02_etapa2_spatial_patterns` | MLP Embedding | Spatial block N/S | v1 | ✅ Frozen |
| etapa3 | `03_etapa3_multihead_random_split` | Multi-Head | Random pixel | v1 | ⚠️ Superseded |
| etapa4 | `04_etapa4_baseline_spatial_split` | Baseline MLP | Hexagon-stratified | v1 | ✅ Frozen |
| etapa5 | `05_etapa5_multihead_spatial_split` | Multi-Head | Hexagon-stratified | v1 | ✅ Frozen |
| etapa6 | `07_etapa6_geofmv3_spatial_attention` | GeoFM v3 | Hexagon-stratified | v1 | ⚠️ Superseded |
| etapa6b | `08_etapa6b_geofmv3b_attention_corrected` | GeoFM v3b | Hexagon-stratified | v1 | ✅ Frozen |
| etapa7 | — | Two-Channel | Hexagon-stratified | v2 | 🔄 Planned |

### OSF Pre-registration updates

| Update | Date | Content |
|--------|------|---------|
| Update 1 | Mar 8, 2026 | Initial registration |
| Update 2 | Mar 11, 2026 | Multi-Head frozen (etapa3) |
| Update 3 | Mar 14, 2026 | Gate weight analysis — implicit ensemble |
| Update 4 | Mar 16, 2026 | **Critical correction** — validation regime incompatibility |
| Update 5 | Mar 18, 2026 | Experimental phase v2 closure |
| Update 6 | Apr 24, 2026 | Dataset v2 + two-channel architectural direction |

---

## Key Findings

### Error Analysis (etapa5 test set)

- **Temporal pattern:** FN rate 45–57% for pixels with T=1990–2000. Model cannot distinguish long-duration pasture that will eventually convert from stable pasture.
- **Spatial pattern:** CCPD achieves only 38% accuracy, dominated by false negatives.
- **Geographic concentration:** FN cluster in specific raster region (col ~40,000–50,000).

### Attention Analysis (etapa6/6b)

Spatial and temporal attention weights converged to uniform in all experiments (std < 0.001). Root cause: patch 7×7 is predominantly homogeneous (pasture class 15 in most positions), providing insufficient spatial gradient for attention to discriminate positions. Performance gains in v3/v3b come from per-frame encoding of the 7×7 grid, not from attention weighting.

### Dataset v2 Findings

- Ecological threshold of 5 years produces a more coherent label definition than the arbitrary 10-year threshold.
- 70% of pixels have a different spatial pattern when using contemporaneous pattern_T vs static Class8590.
- CC* patterns show 73.3% rapid conversion vs 56.2% for CD* — +17.1pp separation (Cohen's h=0.361), supporting deterministic routing.

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
---



Data derived from MapBiomas Collection 10 is subject to [MapBiomas terms of use](https://mapbiomas.org/en/terms-of-use) (CC-BY 4.0).
