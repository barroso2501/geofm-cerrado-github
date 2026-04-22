# GeoFM v2 — Predicting Pasture-to-Soy Conversion in the Brazilian Cerrado

[![OSF](https://img.shields.io/badge/OSF-Pre--registered-blue)](https://osf.io/c46je)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**Institution:** TNC Brasil  
**Author:** Mario Barroso Ramos Neto  
**OSF Pre-registration:** https://osf.io/c46je  
**Status:** Experimental phase closed — GeoFM v3 in development

---

## Overview

This repository contains the full pipeline for GeoFM v2, a deep learning system for predicting pasture-to-soy land conversion in the Brazilian Cerrado using MapBiomas LULC time series (1985–2024).

The project went through five experimental stages, each documented with frozen results and OSF pre-registration updates. The experimental phase is now closed. Key findings inform the design of GeoFM v3, which incorporates a spatial attention module to capture landscape-level conversion dynamics.

---

## Key Results

Under rigorous hexagon-stratified spatial validation (0% geographic overlap between train and test sets):

| Model | Test Accuracy | Test F1 | Test AUC | Val–Test Gap |
|-------|:-------------:|:-------:|:--------:|:------------:|
| Baseline MLP (etapa4) | 61.3% | 0.593 | 0.655 | −3.7pp |
| Multi-Head (etapa5) | 60.3% | 0.578 | 0.648 | −4.3pp |

Neither architecture demonstrated a significant performance advantage under spatial validation. Error analysis revealed that model failures concentrate in pixels with **early pasture entry (T=1990–2000)** — where long-duration pasture trajectories provide insufficient pre-conversion signal — and in **CCPD spatial patterns** (clustered conversion, dispersed land use), where landscape-level dynamics dominate individual pixel signals.

> **Note:** An earlier experiment (etapa3) using a random pixel-level split reported 68.2% accuracy for the Multi-Head model. This comparison was invalidated by incompatible validation regimes between baseline and Multi-Head models and is documented as superseded (see OSF Update 4).

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
│   │     Initial sampling from classified raster (soy2024_origin_classes).
│   │     Temporal split: train T=1990–2010, val T=2011–2015, test T=2016–2019.
│   │     Output: indice_treino/val/teste.csv
│   │
│   ├── 02_etapa2_spatial_patterns_multihead.ipynb
│   │     Hexagonal spatial pattern integration (hex_Cerrado_class_Change.shp).
│   │     Stratified resampling by CC*/CD* pattern codes.
│   │     Multi-Head training with random split (preliminary).
│   │     Output: treino_balanceado_FINAL.csv, stability_results.json
│   │
│   ├── 03_etapa3_multihead_random_split.ipynb   ⚠️ SUPERSEDED
│   │     Multi-Head architecture with random pixel-level split.
│   │     Results invalidated — see OSF Update 4.
│   │     Kept for reproducibility and audit trail.
│   │
│   ├── 04_etapa4_baseline_spatial_split.ipynb
│   │     Baseline MLP (287→256→128→64→1) with hexagon-stratified spatial split.
│   │     Valid baseline for architectural comparison.
│   │     Output: baseline_spatial_frozen/
│   │
│   ├── 05_etapa5_multihead_spatial_split.ipynb
│   │     Multi-Head architecture with same spatial split as etapa4.
│   │     Valid comparison against baseline.
│   │     Output: multihead_spatial_frozen/
│   │
│   └── 06_error_analysis.ipynb
│         Error analysis on etapa5 test set predictions.
│         Identifies spatial, temporal, and confidence patterns in model failures.
│         Output: predictions_test.csv, error_analysis_*.png
│
├── scripts/
│   └── spatial_split.py
│         Standalone script for hexagon-stratified spatial split.
│         No external dependencies beyond numpy and pandas.
│         Assigns pixels to hexagons via Albers Equal Area projection,
│         then splits hexagons 70/15/15 stratified by Class8590 pattern.
│
└── results/
    ├── spatial_split_metadata.json   — split composition and statistics
    └── predictions_test.csv          — per-pixel predictions with error type
```

---

## Data

### Required (not included — external sources)

| File | Source | Description |
|------|--------|-------------|
| `brazil_coverage_{year}_Cerrado.tif` | [MapBiomas Collection 10](https://mapbiomas.org) | Annual LULC rasters, 1985–2024, 30m |
| `hex_Cerrado_class_Change.shp` | TNC Brasil | Hexagonal grid with spatial pattern codes (CC*/CD*) |
| `soy2024_origin_classes_N4_P15_S39_k4_mask.tif` | Derived from MapBiomas | Pre-classified pasture trajectory raster (etapa1 only) |

### Generated (included)

| File | Description |
|------|-------------|
| `results/spatial_split_metadata.json` | Split composition: 6,152 train / 1,646 val / 1,505 test pixels |
| `results/predictions_test.csv` | Test set predictions: prob, pred, error_type, gate weights, Class8590 |

### Dataset

The training dataset (`treino_balanceado_FINAL.csv`, 9,303 valid samples) is available at:  
**Zenodo:** [DOI: 10.5281/zenodo.19021498](https://doi.org/10.5281/zenodo.19021498)

### Model weights

Frozen model checkpoints are available at:  
**Zenodo:** [DOI: 10.5281/zenodo.19021415](https://doi.org/10.5281/zenodo.19021415)

---

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU recommended (CPU supported, slower)

### Setup

```bash
git clone https://github.com/barroso2501/geofm-cerrado-github.git
cd geofm-cerrado-github
pip install -r requirements.txt
```

### Data paths

Update the path constants at the top of each notebook before running:

```python
DATA_DIR  = r"D:\Projetos\Cerrado\LULC"           # MapBiomas rasters
BASE_DIR  = r"D:\Projetos\Cerrado\GeoFM_sampling"  # Working directory
SPLIT_DIR = BASE_DIR / "spatial_split"             # Split CSVs
```

---

## Running the Pipeline

### Full pipeline (from scratch)

Run notebooks in order:

```
01 → 02 → scripts/spatial_split.py → 04 → 05 → 06
```

Notebook 03 is superseded and does not need to be run.

### Spatial split only

If you already have `treino_balanceado_FINAL.csv` and the hexagonal shapefile:

```bash
python scripts/spatial_split.py
```

Edit the path constants at the top of the script before running. Output:
```
spatial_split/
├── spatial_split_train.csv   (6,152 pixels, 921 hexagons)
├── spatial_split_val.csv     (1,646 pixels, 197 hexagons)
├── spatial_split_test.csv    (1,505 pixels, 198 hexagons)
└── spatial_split_metadata.json
```

### Inference only (using frozen model)

Load the frozen Multi-Head model from the checkpoint:

```python
import torch
from notebooks.models import MultiHeadSpatialModel  # or copy class definition

checkpoint = torch.load('multihead_spatial_frozen_20260318_060216.pth',
                        map_location='cpu')
model = MultiHeadSpatialModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Architecture

### Feature vector (287 dimensions)

```
X = [LULC_series (39), Spatial_patch (245), Auxiliary (3)]

LULC_series:   Annual land-use class (1985 to T-1), left-padded to length 39
Spatial_patch: 7×7 pixel neighborhood for 5 years before T, flattened (5×7×7=245)
Auxiliary:     [proportion years with class 21, consecutive pasture years, class at T-1]

T = year of pasture entry (pixel transitions from native vegetation to pasture)
Label = 1 if pixel converts to soy within <10 years of T, else 0
```

### Baseline MLP

```
Input (287) → Dense(256, ReLU) → Dropout(0.3)
            → Dense(128, ReLU) → Dropout(0.3)
            → Dense(64,  ReLU) → Dropout(0.3)
            → Dense(1, Sigmoid)

Parameters: ~95,000
```

### Multi-Head with Soft Gate

```
Input (287) → Shared Encoder (287→256→128)
                    ├── Head Disperso (128→64→1, Sigmoid)
                    ├── Head Cluster  (128→64→1, Sigmoid)
                    └── Gate Network  ((128+3)→32→2, Softmax)

Output = w_disperso × pred_disperso + w_cluster × pred_cluster

Parameters: 127,556
Gate behavior: soft weighting (~0.55/0.45 mean), 91.5% samples use balanced weights
```

---

## Experimental History

| Stage | Notebook | Model | Split | Status |
|-------|----------|-------|-------|--------|
| etapa1 | `01_etapa1_sampling` | Logistic Regression | Temporal block | ✅ Frozen |
| etapa2 | `02_etapa2_spatial_patterns` | MLP Embedding | Spatial block N/S | ✅ Frozen |
| etapa3 | `03_etapa3_multihead_random_split` | Multi-Head | Random pixel | ⚠️ Superseded |
| etapa4 | `04_etapa4_baseline_spatial_split` | Baseline MLP | Hexagon-stratified | ✅ Frozen |
| etapa5 | `05_etapa5_multihead_spatial_split` | Multi-Head | Hexagon-stratified | ✅ Frozen |

### OSF Pre-registration updates

| Update | Date | Content |
|--------|------|---------|
| Update 1 | Mar 8, 2026 | Initial registration — experimental design |
| Update 2 | Mar 11, 2026 | Multi-Head frozen (etapa3) |
| Update 3 | Mar 14, 2026 | Gate weight analysis — implicit ensemble interpretation |
| Update 4 | Mar 16, 2026 | **Critical correction** — validation regime incompatibility |
| Update 5 | Mar 18, 2026 | Experimental phase closure |

---

## Key Findings from Error Analysis

Error analysis on the etapa5 test set (`06_error_analysis.ipynb`) revealed:

**Temporal pattern:** Model fails most severely on pixels with T between 1990–2000 (FN rate 45–57%). Pixels entering pasture in this period and converting decades later provide insufficient pre-conversion signal in the available temporal window. Pixels with T=2010–2015 achieve 80% accuracy.

**Spatial pattern:** CCPD pattern (clustered conversion, dispersed land use) achieves only 38% accuracy, dominated by false negatives. The model systematically misses conversions in areas where landscape-level dynamics are most complex.

**Geographic concentration:** False negatives concentrate in a specific geographic zone (col ~40,000–50,000, row ~30,000–45,000 in raster coordinates), suggesting systematic blind spots that are not captured by the current feature set.

**Implication for GeoFM v3:** The dominant failure mode is temporal, not spatial. The model lacks the capacity to detect long-duration pasture trajectories with accumulating external pressure. This motivates a temporal attention module as the primary architectural addition in v3.

---

## GeoFM v3 — Next Steps

GeoFM v3 will incorporate a **spatial attention module** over the 7×7 neighborhood patch, replacing the current flat patch representation. The module learns directional and temporal weights over the spatial neighborhood, capturing landscape-level conversion pressure without imposing a fixed geographic taxonomy.

Planned architecture:
```
LULC series (39)    → Temporal encoder        → vector (128)
Patch (5×7×7)       → Spatial attention module → vector (64)
                                                    ↓
                          Concatenate [temporal + spatial] (192)
                                                    ↓
                              Prediction head → P(conversion)
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{barroso2026geofm,
  author    = {Barroso Ramos Neto, Mario},
  title     = {GeoFM v2: Predicting Pasture-to-Soy Conversion 
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

Data derived from MapBiomas Collection 10 is subject to [MapBiomas terms of use](https://mapbiomas.org/en/terms-of-use) (CC-BY 4.0).
