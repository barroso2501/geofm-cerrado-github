# GeoFM v2 - Predicting Pasture-to-Soy Conversion in the Brazilian Cerrado

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Multi-Head deep learning architecture for predicting land-use conversion trajectories in the Brazilian Cerrado using MapBiomas LULC time series (1985-2024).

---

## 📊 Key Results

| Metric | Baseline | Multi-Head | Improvement |
|--------|----------|------------|-------------|
| **Test Accuracy** | 55.3% | **68.2%** | +12.9pp (+23%) |
| **F1 Score** | 0.24 | **0.62** | +0.38 (+158%) |
| **Precision** | ~50% | **75.6%** | +25.6pp (+51%) |
| **Val-Test Gap** | +26pp | **-2.3pp** | -28.3pp (-91%) |

**Key Achievement:** Reduced spatial overfitting from 26pp gap to -2.3pp through implicit ensemble learning.

---

## 🏗️ Architecture

```
Multi-Head Spatial Model with Soft Gating

Input Features (287):
  ├─ LULC Time Series [39] (padded, 1985-T)
  ├─ Spatial Patch [245] (5 years × 7×7 grid)
  └─ Auxiliary [3] (pasture history, years_past, class_t-1)

Architecture:
  ┌─ Shared Encoder: 287 → 256 → 128
  ├─ Head 1 (Disperso): 128 → 64 → 1 (sigmoid)
  ├─ Head 2 (Cluster): 128 → 64 → 1 (sigmoid)
  └─ Gate Network: (128+3) → 32 → 2 (softmax)

Output: w₁ × pred₁ + w₂ × pred₂

Parameters: 127,556 (~128k)
Model Size: 0.50 MB
Inference: ~0.10 ms/sample
```

---

## 📁 Repository Structure

```
geofm-cerrado-github/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── train_multihead_cerrado.ipynb          # Training notebook
└── multihead_frozen_20260311_133725.pth   # Trained model weights
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/geofm-cerrado-github.git
cd geofm-cerrado-github
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Dataset

Dataset is hosted on Zenodo (DOI: 10.5281/zenodo.YYYYYYY):
- `treino_balanceado_FINAL.csv` (9,303 samples, ~5 MB)

Download and place in project root directory.

### 4. Run Training

```bash
# Open Jupyter notebook
jupyter notebook train_multihead_cerrado.ipynb

# Or run from command line (if using Python script)
# python train.py
```

---

## 📖 Data Description

### Dataset

- **Source:** MapBiomas Collection 10 (1985-2024)
- **Samples:** 9,303 valid pixels (Label 0: 4,805, Label 1: 4,498)
- **Labels:**
  - **0 (Slow/No Conversion):** Pasture pixels staying ≥10 years before converting to soy
  - **1 (Rapid Conversion):** Pasture pixels converting to soy <10 years
- **Temporal Coverage:** T (pasture entry year) ranges from 1986 to 2022
- **Spatial Coverage:** Brazilian Cerrado biome
- **Splits:** 70% train (6,512), 15% val (1,395), 15% test (1,396)

### LULC Rasters

Land-use/land-cover rasters are publicly available from:
- **MapBiomas:** https://mapbiomas.org
- **Collection:** 10
- **Years:** 1985-2024
- **Resolution:** 30m
- **File pattern:** `brazil_coverage_{year}_Cerrado.tif`

Place rasters in: `D:\Projetos\Cerrado\LULC\` (or update path in notebook)

---

## 🔬 Scientific Context

### Research Question

**"Which pasture pixels in the Cerrado will convert to soybean cropland?"**

### Challenge: Spatial Heterochrony

Standard machine learning models trained on multi-region datasets learn to exploit geographic patterns rather than temporal processes, resulting in poor spatial generalization:

- **Baseline MLP:** 81% validation accuracy → 55% test accuracy (26pp gap)
- **Root Cause:** Identical temporal sequences have different conversion rates by region
  - Frontier regions (Matopiba): Rapid conversion
  - Consolidated regions (Mato Grosso): Slow conversion

### Solution: Multi-Head Architecture

Our Multi-Head model learns region-agnostic temporal representations through:
1. **Shared Encoder:** Learns common temporal patterns
2. **Dual Heads:** Provide architectural diversity via different initializations
3. **Soft Gate:** Combines heads with learned sample-specific weights

**Mechanism:** Implicit ensemble (not Mixture of Experts)
- Gate weights: Mean ~0.55/0.45 (unimodal distribution)
- 70% of samples use balanced weights (0.4 < w < 0.6)
- Both heads contribute complementarily

---

## 📊 Results Details

### Test Set Performance (n=1,396 samples)

```
Accuracy:   68.2%
Precision:  75.6%
Recall:     52.3%
F1 Score:   0.619
ROC AUC:    75.4%
```

### Gate Weight Analysis

```
Head 1 (Disperso): Mean=0.550, Std=0.073, Range=[0.389, 0.714]
Head 2 (Cluster):  Mean=0.450, Std=0.073, Range=[0.286, 0.611]

Distribution: Unimodal (centered at 0.5)
Interpretation: Soft weighting, implicit ensemble behavior
```

### Cost-Benefit

- **Computational Overhead:** +30% (parameters, training time)
- **Performance Gain:** +56% (accuracy, F1, gap reduction)
- **ROI:** +87% (1.87% gain per 1% invested)

---

## 📄 Citation

If you use this code or model in your research, please cite:

```bibtex
@article{ramos2026geofm,
  title={Predicting Pasture-to-Soy Conversion in the Brazilian Cerrado: 
         A Multi-Head Deep Learning Approach to Mitigate Spatial Heterochrony},
  author={Ramos Neto, Mario Barroso and [Co-authors]},
  journal={Remote Sensing of Environment},
  year={2026},
  note={In preparation}
}
```

**Pre-registration:** https://osf.io/c46je

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data License:** CC-BY 4.0 (Zenodo dataset)

---

## 🙏 Acknowledgments

- **MapBiomas** for providing LULC data (Collection 10)
- **TNC Brasil** for institutional support
- **OSF** for pre-registration infrastructure

---

## 🔗 Related Links

- **Paper:** [arXiv link when available]
- **Dataset:** [Zenodo DOI: 10.5281/zenodo.YYYYYYY]
- **Pre-registration:** https://osf.io/c46je
- **MapBiomas:** https://mapbiomas.org

---

## 📧 Contact

**Mario Barroso Ramos Neto**  
The Nature Conservancy Brasil  
Email: [your email]  
ORCID: [your ORCID]

---

## 🐛 Issues & Contributions

Found a bug or have a suggestion? Please open an issue on GitHub.

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Last Updated:** March 14, 2026  
**Version:** 1.0
