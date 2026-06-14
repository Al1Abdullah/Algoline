---
title: Algoline
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<br>

# Algoline

### Automated Machine Learning Platform

<br>

[![Python](https://img.shields.io/badge/Python-3.10-4f46e5?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-6366f1?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-818cf8?style=for-the-badge)](https://pycaret.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-a78bfa?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-Open_Source-c4b5fd?style=for-the-badge)](LICENSE)

<br>

<p align="center">
<strong>Upload a raw dataset. Get a production ready model. No code required.</strong>
</p>

<p align="center">
Algoline handles the entire machine learning pipeline, from data profiling and exploratory analysis<br>through automated model selection, hyperparameter optimization, and artifact export,<br>inside a single polished browser interface.
</p>

<br>

[**Try Live Demo →**](https://huggingface.co/spaces/Al1Abdullah/AutoML)

<br>

</div>

<br>

## ✦ Overview

Algoline is not a notebook. It is not a drag and drop toy. It is a complete, self contained ML platform that runs a FastAPI backend, profiles your data on upload, trains and cross validates every relevant algorithm in parallel, ranks them on a leaderboard, tunes the winner with Bayesian optimization, generates evaluation diagnostics, and packages the final model into a downloadable `.pkl` pipeline. Everything happens in the browser. Everything adapts to your dataset.

<br>

## ✦ Core Capabilities

<table>
<tr>
<td width="50%" valign="top">

### 📊 Intelligent Data Profiling

The moment you upload a file, Algoline computes row and column counts, scans for missing values and duplicates, infers data types, and generates a statistical summary. It flags quality issues automatically through contextual insights, telling you what matters before you even ask.

</td>
<td width="50%" valign="top">

### 🔍 18 Interactive Visualizations

Distribution analysis (histograms, KDE, box plots, violin plots), correlation heatmaps, pair plots, scatter matrices, missing value heatmaps, joint plots, pie charts, count plots, mean target breakdowns, grouped box plots, and faceted small multiples. Every chart is rendered with Plotly, fully zoomable, and theme aware.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### ⚡ Automated Model Training

One click triggers a full compare across every relevant algorithm: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVMs, KNN, and more. PyCaret handles the heavy lifting. You get a ranked leaderboard with cross validated metrics, a highlighted best model, comparison bar charts, confusion matrices, and prediction distributions.

</td>
<td width="50%" valign="top">

### 🎯 Hyperparameter Optimization

Tune the winning model with Optuna (Bayesian), random search, or grid search. Configure iteration count, and Algoline re evaluates performance, regenerates all evaluation plots, and tells you whether the tuned version actually improved. If it didn't, the original is preserved.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🧪 Configurable Preprocessing

Toggle eight preprocessing steps independently: duplicate removal, outlier detection, normalization, multicollinearity filtering, skewness transformation, feature selection, polynomial feature generation, and class imbalance correction. Set your test split ratio and cross validation folds. Full control, zero boilerplate.

</td>
<td width="50%" valign="top">

### 📦 One Click Export

Download three artifacts when you are satisfied. The **Pipeline** is a serialized `.pkl` file ready to load into any Python environment. The **Leaderboard** is a CSV of all compared models and their metrics. The **Predictions** CSV contains the model's output on the holdout set. Load the pipeline, call `predict()`, and you are in production.

</td>
</tr>
</table>

<br>

## ✦ Design Philosophy

The interface ships with a dual theme system. The dark theme uses glassmorphism surfaces with translucent cards, indigo ambient gradients, and subtle glow effects. The light theme uses an indigo tinted color palette with layered shadows, visible gridlines, and a gradient sidebar. Every chart, card, table, and interactive element is designed for both modes without visual compromise. A single toggle switches between them instantly.

Typography is set in Inter. Spacing is deliberate. Animations are subtle. Nothing is there for decoration; everything communicates.

<br>

## ✦ Architecture

```
algoline/
├── main.py                 FastAPI server · 30+ API endpoints · PyCaret integration
├── static/
│   └── index.html          Single page application · CSS variables · Plotly.js
├── requirements.txt        Python dependencies
├── Dockerfile              Production container
└── README.md
```

| Layer | Stack |
|:--|:--|
| **Server** | FastAPI, Uvicorn, Python 3.10 |
| **ML Engine** | PyCaret 3.3 (scikit learn, XGBoost, LightGBM, CatBoost under the hood) |
| **Optimization** | Optuna 3.5+ with Bayesian TPE sampler |
| **Charts** | Plotly 5.24, server side figure generation, client side rendering |
| **Frontend** | Vanilla HTML/CSS/JS, CSS custom properties for theming, zero frameworks |
| **Deployment** | Docker multi stage build, Hugging Face Spaces |

<br>

## ✦ Quick Start

**Local**

```bash
git clone https://huggingface.co/spaces/Al1Abdullah/AutoML
cd AutoML
pip install -r requirements.txt
python main.py
```

Open `http://localhost:7860` in your browser.

**Docker**

```bash
docker build -t algoline .
docker run -p 7860:7860 algoline
```

<br>

## ✦ API Surface

<details>
<summary><strong>Data Endpoints</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/upload` | Upload dataset (CSV, TSV, XLSX) with automatic profiling |
| `POST` | `/api/target` | Set target column, auto detect task type |

</details>

<details>
<summary><strong>Exploration Endpoints (18 visualization types)</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/explore/distribution` | Histogram for a selected feature |
| `POST` | `/api/explore/kde` | Kernel density estimation plot |
| `POST` | `/api/explore/boxplot` | Box plot with outlier markers |
| `POST` | `/api/explore/violin` | Violin plot with density shape |
| `POST` | `/api/explore/missing` | Missing value bar chart |
| `POST` | `/api/explore/missing_heatmap` | Missing value pattern heatmap |
| `POST` | `/api/explore/correlation` | Feature correlation heatmap |
| `POST` | `/api/explore/pairplot` | Pair plot scatter matrix |
| `POST` | `/api/explore/scatter_xy` | Two feature scatter plot |
| `POST` | `/api/explore/jointplot` | Joint distribution plot |
| `POST` | `/api/explore/countplot` | Categorical count plot |
| `POST` | `/api/explore/pie` | Proportional pie chart |
| `POST` | `/api/explore/target` | Target column distribution |
| `POST` | `/api/explore/counts` | Class distribution breakdown |
| `POST` | `/api/explore/mean_target` | Mean target per category |
| `POST` | `/api/explore/scatter_index` | Feature values vs row index |
| `POST` | `/api/explore/grouped_box` | Grouped box plot by category |
| `POST` | `/api/explore/facetgrid` | Faceted small multiples |
| `POST` | `/api/explore/quality` | Feature level quality statistics |

</details>

<details>
<summary><strong>Training and Tuning Endpoints</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/train` | Compare all models, return leaderboard + evaluation plots |
| `POST` | `/api/compare` | Re render comparison chart for a different metric |
| `POST` | `/api/tune` | Hyperparameter optimization (Optuna / Random / Grid) |

</details>

<details>
<summary><strong>Export Endpoints</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `GET` | `/api/export/pipeline` | Download finalized model as `.pkl` |
| `GET` | `/api/export/leaderboard` | Download model comparison as `.csv` |
| `GET` | `/api/export/predictions` | Download holdout predictions as `.csv` |
| `GET` | `/api/summary` | Retrieve pipeline metadata |

</details>

<br>

## ✦ Supported Algorithms

**Classification**: Logistic Regression, K Nearest Neighbors, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, SVM (Linear + RBF), Ridge, LDA, QDA

**Regression**: Linear Regression, Lasso, Ridge, Elastic Net, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, SVR, KNN, Huber, Passive Aggressive

The platform automatically selects which algorithms to run based on your task type and applies a time budget to ensure training completes within reasonable bounds on free CPU infrastructure.

<br>

<div align="center">

<br>

Built by [**Al1Abdullah**](https://huggingface.co/Al1Abdullah)

<br>

</div>
