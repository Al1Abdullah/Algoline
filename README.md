<p align="center">
  <img src="./assets/banner.png" alt="Algoline — Automated Machine Learning Platform" width="720" />
</p>

<p align="center">
  <a href="#overview">Overview</a>
  <span> · </span>
  <a href="#getting-started">Getting Started</a>
  <span> · </span>
  <a href="#platform-walkthrough">Walkthrough</a>
  <span> · </span>
  <a href="#architecture">Architecture</a>
  <span> · </span>
  <a href="#api-reference">API Reference</a>
  <span> · </span>
  <a href="https://huggingface.co/spaces/Al1Abdullah/AutoML">Live Demo</a>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10-4f46e5?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.100+-6366f1?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://pycaret.org"><img src="https://img.shields.io/badge/PyCaret-3.3-818cf8?style=flat-square" alt="PyCaret" /></a>
  <a href="https://optuna.org"><img src="https://img.shields.io/badge/Optuna-3.5+-a78bfa?style=flat-square" alt="Optuna" /></a>
  <a href="https://docker.com"><img src="https://img.shields.io/badge/Docker-Ready-c4b5fd?style=flat-square&logo=docker&logoColor=white" alt="Docker" /></a>
  <a href="https://github.com/Al1Abdullah/Algoline/actions"><img src="https://img.shields.io/github/actions/workflow/status/Al1Abdullah/Algoline/ci.yml?branch=main&style=flat-square&label=CI" alt="CI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-4f46e5?style=flat-square" alt="License" /></a>
</p>

<br>

## Overview

Algoline is a web-based machine learning platform that takes a raw dataset and turns it into a deployable model pipeline, entirely in the browser. There is no notebook to configure, no boilerplate to write, and no environment to set up. You upload a file, explore the data through interactive visualizations, train and compare models with a single click, optionally tune with Bayesian optimization, and download a production-ready `.pkl` file.

Under the hood, a FastAPI server handles all computation. PyCaret orchestrates model comparison across scikit-learn, XGBoost, LightGBM, and CatBoost. Optuna drives hyperparameter search. Plotly renders every chart. The frontend is vanilla HTML, CSS, and JavaScript with zero build dependencies. The whole thing ships as a Docker container.

<p align="center">
  <a href="https://huggingface.co/spaces/Al1Abdullah/AutoML">
    <img src="https://img.shields.io/badge/Open_Live_Demo-6366f1?style=for-the-badge&logo=huggingface&logoColor=white" alt="Live Demo" />
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/Al1Abdullah/Algoline/issues">
    <img src="https://img.shields.io/badge/Report_Issue-818cf8?style=for-the-badge&logo=github&logoColor=white" alt="Report Issue" />
  </a>
</p>

<br>

## Getting Started

### Local Setup

```bash
git clone https://github.com/Al1Abdullah/Algoline.git
cd Algoline
pip install -r requirements.txt
python main.py
```

The server starts at `http://localhost:7860`.

### Docker

```bash
docker build -t algoline .
docker run -p 7860:7860 algoline
```

<br>

## Platform Walkthrough

### Data Ingestion and Profiling

Drop a CSV, TSV, or Excel file into the upload zone. Algoline parses it on the spot and returns four headline metrics (rows, columns, missing values, duplicates), a full statistical breakdown, inferred column types, and automated quality insights. These insights flag class imbalance, high-cardinality categoricals, constant columns, correlated features, and missing value patterns before you even ask. Select a target column and the system infers whether you are working on classification or regression.

### Exploratory Analysis

Eighteen interactive chart types are available, organized into five analytical categories. Every chart is rendered with Plotly, so you can zoom, pan, hover for values, and export.

| Category | Visualizations |
|:--|:--|
| **Distribution** | Histogram, KDE, Box plot, Violin |
| **Missing Data** | Missing value bar chart, Missing value heatmap |
| **Correlation** | Correlation heatmap, Pair plot, Scatter, Joint plot |
| **Target Analysis** | Count plot, Pie chart, Class distribution, Target histogram, Mean target per category |
| **Advanced** | Scatter vs index, Grouped box plot, Faceted small multiples |

### Model Training and Comparison

Eight preprocessing toggles can be configured independently before training:

| Option | Default | Purpose |
|:--|:--|:--|
| Drop Duplicates | On | Remove exact row copies |
| Remove Outliers | Off | Filter statistical outliers |
| Normalize | On | Scale numeric features |
| Drop Multicollinear | On | Remove highly correlated pairs |
| Transform Skew | Off | Power transforms on skewed distributions |
| Feature Selection | Off | Automatic dimensionality reduction |
| Polynomial Features | Off | Generate interaction terms |
| Fix Class Imbalance | Off | Balance underrepresented classes |

Set the test split ratio and cross-validation fold count, then train. PyCaret runs every relevant algorithm, scores each with cross-validation, and returns a ranked leaderboard. Evaluation diagnostics (confusion matrices, ROC curves, precision-recall curves, residual plots, feature importance) are generated automatically from real model predictions.

### Hyperparameter Tuning

Three tuning strategies are available after initial training:

| Strategy | Engine | Best For |
|:--|:--|:--|
| **Bayesian** | Optuna TPE | Smart, sample-efficient search |
| **Random Search** | scikit-learn | Broad exploration |
| **Grid Search** | scikit-learn | Exhaustive evaluation |

If the tuned model does not outperform the original, Algoline preserves the original automatically.

### Export

| Artifact | Format | Description |
|:--|:--|:--|
| **Pipeline** | `.pkl` | Serialized model ready for `predict()` in any Python environment |
| **Leaderboard** | `.csv` | Full model comparison with cross-validated metrics |
| **Predictions** | `.csv` | Model output on the holdout test set |

Load the pipeline with `joblib` or PyCaret's `load_model()`, pass new data, and get predictions. No retraining required.

<br>

## Supported Algorithms

<table>
<tr>
<td width="50%" valign="top">

**Classification**

Logistic Regression, K-Nearest Neighbors, Naive Bayes, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, SVM (Linear and RBF), Ridge Classifier, LDA, QDA

</td>
<td width="50%" valign="top">

**Regression**

Linear Regression, Lasso, Ridge, Elastic Net, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, SVR, KNN Regressor, Huber Regressor, Passive Aggressive Regressor

</td>
</tr>
</table>

<br>

## Architecture

### How It Works

```
                    ┌─────────────────────────────────────────────┐
                    │                  Browser                     │
                    │         index.html + style.css + app.js      │
                    └──────────────────┬──────────────────────────┘
                                       │ fetch()
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │               FastAPI Server                 │
                    │                                              │
                    │   ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                    │   │  data.py │  │explore.py│  │ build.py │  │
                    │   │  upload  │  │ 18 chart │  │  train   │  │
                    │   │  target  │  │endpoints │  │  tune    │  │
                    │   └──────────┘  └──────────┘  └──────────┘  │
                    │                                              │
                    │   ┌──────────┐  ┌──────────────────────────┐ │
                    │   │export.py │  │     helpers.py            │ │
                    │   │ download │  │ state.py (session store)  │ │
                    │   └──────────┘  └──────────────────────────┘ │
                    │                                              │
                    │   PyCaret ─── scikit-learn ─── XGBoost       │
                    │   LightGBM ── CatBoost ── Optuna ── Plotly   │
                    └─────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|:--|:--|
| **Server** | FastAPI with Uvicorn |
| **Runtime** | Python 3.10 |
| **ML Engine** | PyCaret 3.3 (wraps scikit-learn, XGBoost, LightGBM, CatBoost) |
| **Optimization** | Optuna 3.5+ with Tree-structured Parzen Estimator |
| **Visualization** | Plotly 5.24 |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker |
| **Hosting** | Hugging Face Spaces |

### Project Structure

```
algoline/
├── .github/
│   └── workflows/
│       └── ci.yml                  CI pipeline (lint, validate, build, smoke test)
├── app/
│   ├── __init__.py
│   ├── state.py                    Shared session state and Plotly configuration
│   ├── helpers.py                  Utility functions (serialization, inference, formatting)
│   └── routes/
│       ├── __init__.py
│       ├── data.py                 Upload, profiling, target selection
│       ├── explore.py              18 interactive visualization endpoints
│       ├── build.py                Model training, comparison, hyperparameter tuning
│       └── export.py               Pipeline, leaderboard, and prediction downloads
├── static/
│   ├── index.html                  Single-page application structure
│   ├── css/
│   │   └── style.css               Design system (dark/light themes, glassmorphism)
│   └── js/
│       └── app.js                  Frontend logic (navigation, API calls, chart rendering)
├── main.py                         Application entrypoint
├── requirements.txt                Python dependencies
├── Dockerfile                      Production container
└── LICENSE                         MIT
```

### Design

The interface ships with a dual-theme system. Dark mode uses glassmorphism with translucent card surfaces, ambient indigo gradients, and subtle glow effects. Light mode uses an indigo-tinted palette with layered shadows and clean gridlines. Typography is set in Inter, and every Plotly chart re-renders to match the active theme. A toggle in the top-right corner switches between the two instantly.

<br>

## Continuous Integration

The CI pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push and pull request to `main`.

**Stage 1: Lint and Validate** installs dependencies, confirms the server module imports cleanly, and verifies that all expected API routes are registered on the FastAPI application.

**Stage 2: Docker Build** builds the production image, starts the container, waits for the server to respond, and runs a smoke test against the root endpoint. If anything fails, container logs are captured for debugging.

<br>

## API Reference

<details>
<summary><strong>Data Endpoints</strong></summary>
<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/upload` | Upload dataset (CSV, TSV, XLSX) with automatic profiling |
| `POST` | `/api/target` | Set target column and auto-detect task type |

</details>

<details>
<summary><strong>Exploration Endpoints (18 chart types)</strong></summary>
<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/explore/distribution` | Histogram with marginal box plot |
| `POST` | `/api/explore/kde` | Kernel density estimation |
| `POST` | `/api/explore/boxplot` | Box plot (grouped by target if classification) |
| `POST` | `/api/explore/violin` | Violin plot |
| `POST` | `/api/explore/missing` | Missing value bar chart |
| `POST` | `/api/explore/missing_heatmap` | Missing value heatmap |
| `POST` | `/api/explore/correlation` | Annotated correlation heatmap |
| `POST` | `/api/explore/pairplot` | Pair plot of top correlated features |
| `POST` | `/api/explore/scatter_xy` | Two-feature scatter plot |
| `POST` | `/api/explore/jointplot` | Joint distribution with marginal histograms |
| `POST` | `/api/explore/countplot` | Count plot for categorical features |
| `POST` | `/api/explore/pie` | Pie chart of target distribution |
| `POST` | `/api/explore/target` | Target variable distribution |
| `POST` | `/api/explore/counts` | Class distribution |
| `POST` | `/api/explore/mean_target` | Mean target per category or bin |
| `POST` | `/api/explore/scatter_index` | Feature values vs row index |
| `POST` | `/api/explore/grouped_box` | Grouped box plot by target class |
| `POST` | `/api/explore/facetgrid` | Faceted small multiples |
| `POST` | `/api/explore/quality` | Feature quality statistics table |

</details>

<details>
<summary><strong>Training and Tuning Endpoints</strong></summary>
<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/train` | Compare all models, return leaderboard and evaluation plots |
| `POST` | `/api/compare` | Re-render metric comparison chart |
| `POST` | `/api/tune` | Hyperparameter optimization (Bayesian, random, or grid) |

</details>

<details>
<summary><strong>Export Endpoints</strong></summary>
<br>

| Method | Route | Description |
|:--|:--|:--|
| `GET` | `/api/export/pipeline` | Download finalized model (.pkl) |
| `GET` | `/api/export/leaderboard` | Download model comparison (.csv) |
| `GET` | `/api/export/predictions` | Download holdout predictions (.csv) |
| `GET` | `/api/summary` | Pipeline metadata |

</details>

<br>

## Contributing

Contributions are welcome. If you find a bug or want to suggest a feature, please [open an issue](https://github.com/Al1Abdullah/Algoline/issues). For code contributions, fork the repository, create a feature branch, and submit a pull request.

## License

Open source under the [MIT License](LICENSE).
