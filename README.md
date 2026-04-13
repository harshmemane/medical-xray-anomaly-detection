<div align="center">

# 🏥 Medical X-Ray Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.12.3-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.37%25-brightgreen?style=for-the-badge)]()
[![AUC](https://img.shields.io/badge/AUC-0.9966-blue?style=for-the-badge)]()

**Deep learning-based chest X-ray disease detection using DenseNet121**
*Achieving 97.37% accuracy with comprehensive bias analysis*

[📊 View Results](#-key-results) • [🚀 Quick Start](#-installation) • [📁 Project Structure](#-project-structure) • [📖 Notebooks](#-notebooks)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Results](#-key-results)
- [Model Architecture](#-model-architecture)
- [Bias Analysis](#-bias-analysis)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Notebooks](#-notebooks)
- [Screenshots](#-screenshots)
- [Citation](#-citation)
- [License](#-license)

---

## 🔬 Overview

This project implements an automated **chest X-ray anomaly detection system** using deep learning. It was developed as an M.Tech Mini Project to detect pathological findings in chest X-rays from the NIH ChestX-ray14 dataset.

### 🎯 What This Project Does:
- Detects **14 different chest pathologies** from X-ray images
- Uses **transfer learning** with a pre-trained DenseNet121 backbone
- Performs **fairness/bias analysis** across gender and age groups
- Provides detailed performance metrics and visualizations

### 💡 Why It Matters:
Automated X-ray analysis can assist radiologists in high-workload settings, reducing diagnosis time and improving consistency — especially in resource-limited healthcare environments.

---

## 📊 Key Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Overall Accuracy** | 97.37% | > 95% | ✅ Exceeded |
| **AUC Score** | 0.9966 | > 0.95 | ✅ Exceeded |
| **Gender Bias Gap** | 1.24% | < 3% | ✅ Within Limit |
| **Age Bias Gap** | 1.29% | < 3% | ✅ Within Limit |

> 🏆 **All performance targets exceeded!**

---

## 🧠 Model Architecture

```
Input: Chest X-Ray Image (224×224×3)
         │
         ▼
┌─────────────────────┐
│   DenseNet121       │  ← Pre-trained on ImageNet
│   (Transfer Learn)  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Global Avg Pooling │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Dense + Dropout    │  ← Custom classification head
└─────────────────────┘
         │
         ▼
Output: 14-class Pathology Prediction
```

### Why DenseNet121?
- **Dense connections** between layers improve feature reuse
- **Pre-trained weights** from ImageNet provide strong visual features
- **Proven architecture** widely used in medical imaging research
- Achieves state-of-the-art results on ChestX-ray benchmarks

---

## ⚖️ Bias Analysis

A key feature of this project is its **fairness evaluation** — ensuring the model performs equitably across demographic groups.

| Group | Accuracy | Gap vs Overall |
|-------|----------|----------------|
| **Male patients** | ~98.1% | — |
| **Female patients** | ~96.9% | 1.24% ← within acceptable range |
| **Younger patients** | ~97.8% | — |
| **Older patients** | ~96.5% | 1.29% ← within acceptable range |

> ✅ Both bias gaps are **well below the 3% threshold**, indicating the model is fair across demographic groups.

---

## 📁 Project Structure

```
medical_xray_project/
│
├── 📓 notebooks/               # Jupyter notebooks (main workflow)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── ... (9 notebooks total)
│
├── 📂 src/                     # Source code modules
│
├── 📊 outputs/
│   ├── plots/                  # 12 visualization charts (PNG)
│   └── results/                # Metrics, reports (CSV/JSON)
│
├── 📄 requirements.txt         # Python dependencies
├── 📄 .gitignore               # Files excluded from Git
└── 📄 README.md                # This file
│
│   ── NOT INCLUDED IN REPO (see below) ──
├── 🗃️ data/                    # 2GB X-ray images (download separately)
├── 🤖 models/                  # Trained model weights (download separately)
└── 🗄️ database/                # SQLite metadata database
```

---

## 🚀 Installation

### Prerequisites
- Python 3.12.3
- pip (comes with Python)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/harshmemane/medical-xray-anomaly-detection.git
cd medical-xray-anomaly-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset & Model
> ⚠️ The dataset (2GB) and model weights are not included in this repository due to size limits.

**Dataset (NIH ChestX-ray14):**
- Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Place images in: `data/images/`
- Place `Data_Entry_2017.csv` in: `data/`

**Trained Model Weights:**
- Download from: - Download from: [densenet_best.keras]https://drive.google.com/file/d/1g2swCViJDxmZAy06icVuO2ncRFBdT9Bn/view?usp=sharing
- Place in: `models/densenet_best.keras`

---

## ▶️ Usage

### Run Notebooks in Order:
```bash
jupyter notebook
```
Then open notebooks in the `notebooks/` folder, starting from `01_data_exploration.ipynb`.

---

## 🗃️ Dataset

**NIH ChestX-ray14** — One of the largest publicly available chest X-ray datasets.

| Property | Value |
|----------|-------|
| Total Images | 112,120 |
| Images Used | 5,840 |
| Image Size | 1024×1024 px |
| Pathologies | 14 |
| Source | National Institutes of Health |

📥 **Download:** https://nihcc.app.box.com/v/ChestXray-NIHCC

---

## 📓 Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | Data Exploration | EDA, class distribution, image statistics |
| 02 | Preprocessing | Normalization, augmentation, train/val/test split |
| 03 | Model Training | DenseNet121 transfer learning, callbacks |
| 04 | Evaluation | Accuracy, AUC, confusion matrix |
| ... | ... | ... |

---

## 🖼️ Screenshots

*(Add screenshots of your output plots here after uploading)*

---

## 📖 Citation

If you use this project in your research or academic work, please cite:

```bibtex
@misc{medical_xray_anomaly_detection,
  author    = {Harsh Memane},
  title     = {Medical X-Ray Anomaly Detection using DenseNet121},
  year      = {2024},
  publisher = {GitHub},
  url       = {url       = {https://github.com/harshmemane/medical-xray-anomaly-detection}
}
```

**Original Dataset:**
> Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* CVPR.

---

## 👨‍🎓 About

Developed as an **M.Tech Mini Project** demonstrating the application of deep learning in medical imaging with fairness-aware evaluation.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
⭐ If you found this project useful, please give it a star!

Made with ❤️ for advancing AI in healthcare
</div>