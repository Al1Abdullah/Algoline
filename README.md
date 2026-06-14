<div align="center">

<br>

# **Algoline**

<br>

<a href="https://huggingface.co/spaces/Al1Abdullah/AutoML">
  <img src="https://readme-typing-svg.demolab.com?font=Inter&weight=500&size=18&duration=4000&pause=800&color=A78BFA&center=true&vCenter=true&width=500&height=30&lines=Automated+Machine+Learning+Platform" alt="Automated Machine Learning Platform" />
</a>

<br><br>

[![Python](https://img.shields.io/badge/Python-3.10-4f46e5?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-6366f1?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-818cf8?style=for-the-badge)](https://pycaret.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.5+-a78bfa?style=for-the-badge)](https://optuna.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-c4b5fd?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-4f46e5?style=for-the-badge)](LICENSE)

<br>

A complete machine learning platform that takes you from a raw dataset to a deployable model pipeline, entirely in your browser. No notebooks, no boilerplate, no setup friction.

<br>

<a href="https://huggingface.co/spaces/Al1Abdullah/AutoML">
  <img src="https://img.shields.io/badge/Open_Live_Demo-6366f1?style=for-the-badge&logo=huggingface&logoColor=white" alt="Live Demo" />
</a>
&nbsp;&nbsp;
<a href="https://github.com/Al1Abdullah/Algoline/issues">
  <img src="https://img.shields.io/badge/Report_Issue-818cf8?style=for-the-badge&logo=github&logoColor=white" alt="Report Issue" />
</a>

<br><br>

</div>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Overview

Algoline is a web based ML platform built for speed and clarity. You upload a dataset, the system profiles it on arrival, and you can explore it through eighteen different interactive visualizations before any model is trained. When you are ready, a single click triggers a full cross validated comparison across every relevant algorithm, from logistic regression and random forests through XGBoost, LightGBM, and CatBoost. The best performer is surfaced on a ranked leaderboard, and you can optionally tune it with Bayesian optimization through Optuna.

The result is a downloadable `.pkl` pipeline file that contains the complete trained model. Load it anywhere Python runs, call `predict()`, and you are in production.

Under the hood, it is a single FastAPI server handling the computation, a single HTML file handling the interface, and a Docker container handling the deployment. No React. No Vue. No build step. Everything is deliberate.

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Platform Walkthrough

### Data Ingestion and Profiling

You drop a CSV, TSV, or Excel file into the upload zone. Algoline immediately parses it and returns four summary metrics (rows, columns, missing values, duplicates), a full statistical breakdown, inferred column types, and automated quality insights that flag issues before you have to go looking for them. You then select a target column, and the system infers whether you are working on a classification or regression task based on the target's value distribution.

<br>

### Exploratory Analysis

Eighteen interactive chart types are available, organized into five analytical categories. Each one is rendered with Plotly, so you can zoom, pan, hover for values, and export individual charts.

| Category | Visualizations |
|:--|:--|
| **Distribution** | Histogram, KDE plot, Box plot, Violin plot |
| **Missing Data** | Missing value bar chart, Missing value heatmap |
| **Correlation** | Correlation heatmap, Pair plot, Scatter plot, Joint plot |
| **Target Analysis** | Count plot, Pie chart, Class distribution, Target histogram, Mean target per category |
| **Advanced** | Scatter vs index, Grouped box plot, Faceted small multiples |

The goal is to give you deep familiarity with your data before committing to a modeling strategy. These are not static images; they are fully interactive and re-render when you switch themes.

<br>

### Model Training and Comparison

Configure eight preprocessing toggles independently:

| Option | Default | Purpose |
|:--|:--|:--|
| Drop Duplicates | On | Remove exact row copies |
| Remove Outliers | Off | Filter statistical outliers |
| Normalize | On | Scale numeric features |
| Drop Multicollinear | On | Remove highly correlated feature pairs |
| Transform Skew | Off | Apply power transforms to skewed distributions |
| Feature Selection | Off | Reduce dimensionality automatically |
| Polynomial Features | Off | Generate interaction terms |
| Fix Class Imbalance | Off | Balance underrepresented classes |

Set your test split ratio and cross validation fold count, then hit train. PyCaret runs every relevant algorithm, scores them with cross validation, and returns a ranked leaderboard. The winning model gets a highlighted card with key metrics, and you can compare all models visually across any metric through an interactive bar chart.

Evaluation diagnostics are generated automatically: confusion matrices, prediction distributions, and performance curves, all inside the same interface.

<br>

### Hyperparameter Tuning

After training, you can push the best model further. Three tuning strategies are available:

| Strategy | Engine | Best for |
|:--|:--|:--|
| **Bayesian (Optuna)** | TPE sampler | Smart, sample-efficient search (recommended) |
| **Random Search** | scikit-learn | Broad exploration of parameter space |
| **Grid Search** | scikit-learn | Exhaustive evaluation of defined ranges |

Set the number of iterations, run the tuner, and Algoline re-evaluates performance, regenerates all evaluation plots, and tells you whether the tuned version actually improved. If it did not, the original model is preserved automatically.

<br>

### Export and Deployment

Three artifacts are available for download once you are satisfied with the results:

| Artifact | Format | Description |
|:--|:--|:--|
| **Pipeline** | `.pkl` | Serialized model, ready for `predict()` in any Python environment |
| **Leaderboard** | `.csv` | Full model comparison with cross validated metrics |
| **Predictions** | `.csv` | Model output on the holdout test set |

The pipeline file is a self contained object. You load it with `joblib` or PyCaret's `load_model()`, pass new data, and get predictions. No retraining, no environment dependencies beyond the model's own libraries.

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

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

A time budget is applied during training so the comparison stays practical. You still get a thorough evaluation; it simply will not spend twenty minutes stuck on a single slow estimator.

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Interface Design

The interface is built with a dual theme system. A toggle in the top right corner switches between dark and light modes instantly.

The **dark theme** uses glassmorphism with translucent card surfaces, ambient indigo gradient effects, and subtle glow animations. The **light theme** uses an indigo-tinted color palette with layered card shadows, visible table gridlines, and a gradient sidebar. Every Plotly chart re-renders its text, gridlines, and background colors to match the active theme. The two modes are completely independent; nothing from one bleeds into the other.

Typography is set in Inter, spacing is intentional, and animations are kept minimal. Every visual element is there to communicate, not to decorate.

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Tech Stack

| Layer | Technology |
|:--|:--|
| **Server** | FastAPI with Uvicorn |
| **Runtime** | Python 3.10 |
| **ML Engine** | PyCaret 3.3 (wraps scikit-learn, XGBoost, LightGBM, CatBoost) |
| **Optimization** | Optuna 3.5+ with Tree-structured Parzen Estimator sampling |
| **Visualization** | Plotly 5.24 (server-side generation, client-side rendering) |
| **Frontend** | Vanilla HTML, CSS custom properties, JavaScript |
| **Containerization** | Docker |
| **Hosting** | Hugging Face Spaces |

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Project Structure

```
algoline/
├── .github/
│   └── workflows/
│       └── ci.yml              CI pipeline (lint, validate, build, smoke test)
├── static/
│   └── index.html              Single-page frontend application
├── main.py                     FastAPI backend with 30+ API endpoints
├── requirements.txt            Python dependencies
├── Dockerfile                  Production container configuration
├── LICENSE                     MIT License
└── README.md
```

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Getting Started

**Local Setup**

```bash
git clone https://github.com/Al1Abdullah/Algoline.git
cd Algoline
pip install -r requirements.txt
python main.py
```

Open `http://localhost:7860` in your browser.

<br>

**Docker**

```bash
docker build -t algoline .
docker run -p 7860:7860 algoline
```

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## Continuous Integration

The project includes a GitHub Actions pipeline (`.github/workflows/ci.yml`) that triggers on every push and pull request to `main`. The pipeline runs in two stages:

**Lint and Validate** installs all dependencies, verifies that the server module imports cleanly, and checks that every expected API route (`/api/upload`, `/api/train`, `/api/tune`, `/api/export/pipeline`) is registered on the FastAPI app.

**Docker Build** builds the production image and runs a smoke test, starting the container, waiting for the server to come up, and confirming it responds to HTTP requests. If anything fails, container logs are dumped for debugging.

<br>

<!-- ═══════════════════════════════════════════════════════════════════ -->

## API Reference

<details>
<summary><strong>Data Endpoints</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/upload` | Upload dataset (CSV, TSV, XLSX) with automatic profiling |
| `POST` | `/api/target` | Set target column, auto-detect task type |

</details>

<details>
<summary><strong>Exploration Endpoints (18 chart types)</strong></summary>

<br>

| Method | Route | Description |
|:--|:--|:--|
| `POST` | `/api/explore/distribution` | Histogram |
| `POST` | `/api/explore/kde` | Kernel density estimation |
| `POST` | `/api/explore/boxplot` | Box plot |
| `POST` | `/api/explore/violin` | Violin plot |
| `POST` | `/api/explore/missing` | Missing value bar chart |
| `POST` | `/api/explore/missing_heatmap` | Missing value heatmap |
| `POST` | `/api/explore/correlation` | Correlation heatmap |
| `POST` | `/api/explore/pairplot` | Pair plot matrix |
| `POST` | `/api/explore/scatter_xy` | Two-feature scatter plot |
| `POST` | `/api/explore/jointplot` | Joint distribution |
| `POST` | `/api/explore/countplot` | Count plot |
| `POST` | `/api/explore/pie` | Pie chart |
| `POST` | `/api/explore/target` | Target distribution |
| `POST` | `/api/explore/counts` | Class distribution |
| `POST` | `/api/explore/mean_target` | Mean target per category |
| `POST` | `/api/explore/scatter_index` | Feature values vs row index |
| `POST` | `/api/explore/grouped_box` | Grouped box plot |
| `POST` | `/api/explore/facetgrid` | Faceted small multiples |
| `POST` | `/api/explore/quality` | Feature quality statistics |

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

<!-- ═══════════════════════════════════════════════════════════════════ -->

## License

Open source under the [MIT License](LICENSE).
