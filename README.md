<div align="center">

<br>

# **Algoline**

**Automated Machine Learning Platform**

<br>

[![Python](https://img.shields.io/badge/Python-3.10-4f46e5?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-6366f1?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3-818cf8?style=for-the-badge)](https://pycaret.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.5+-a78bfa?style=for-the-badge)](https://optuna.org)
[![CI](https://img.shields.io/github/actions/workflow/status/Al1Abdullah/Algoline/ci.yml?style=for-the-badge&label=CI&color=4f46e5)](https://github.com/Al1Abdullah/Algoline/actions)

<br>

A complete machine learning platform that takes you from raw data to a deployable model,
entirely in your browser. No notebooks. No boilerplate. Just results.

<br>

[**Try the Live Demo**](https://huggingface.co/spaces/Al1Abdullah/AutoML)

<br>

</div>

<br>

## What is this

Algoline is a web based ML platform where you upload a dataset, explore it through interactive visualizations, and train models against each other automatically. The system handles preprocessing, runs cross validated comparisons across a wide set of algorithms, surfaces the best performer, and lets you tune it further with Bayesian optimization. When you are happy with the results, you download the trained pipeline as a `.pkl` file and use it wherever Python runs.

The whole thing is a single FastAPI server serving a single page frontend. No React, no Vue, no build step. The backend does the heavy computation, the frontend handles the interaction. It deploys as a Docker container.

<br>

## How it works

The platform is organized into four steps that mirror a natural ML workflow.

**Data** — You drop a CSV, TSV, or Excel file into the upload area. Algoline immediately profiles it: row count, column count, missing values, duplicates, inferred types, and descriptive statistics. It generates contextual insights about data quality issues so you know what you are dealing with before doing anything else. You pick a target column, and the system infers whether you are looking at a classification or regression problem.

**Explore** — Eighteen interactive chart types help you understand the data. These are grouped into distribution analysis (histograms, KDE, box plots, violin plots), missing data patterns (bar charts and heatmaps), correlations (heatmaps, pair plots, scatter and joint plots), target analysis (count plots, pie charts, class distributions, mean target breakdowns), and advanced views (scatter vs index, grouped box plots, faceted small multiples). Everything is rendered with Plotly, so you can zoom, pan, and hover for details.

**Build** — You configure preprocessing toggles (duplicate removal, outlier handling, normalization, multicollinearity filtering, skewness transforms, polynomial features, feature selection, class imbalance correction), set your test split and fold count, and hit train. PyCaret runs a cross validated comparison across every relevant algorithm and returns a ranked leaderboard. The best model gets its own highlighted card. You can view comparison charts across any metric, inspect confusion matrices and prediction distributions, and then optionally tune the winner using Optuna, random search, or grid search.

**Export** — Three downloads: the finalized model pipeline (`.pkl`), the full leaderboard (`.csv`), and the holdout predictions (`.csv`). The pipeline file contains everything needed to call `predict()` on new data in any Python environment.

<br>

## Tech stack

| Layer | What |
|:--|:--|
| **Server** | FastAPI with Uvicorn, Python 3.10 |
| **ML** | PyCaret 3.3, which wraps scikit-learn, XGBoost, LightGBM, and CatBoost |
| **Tuning** | Optuna 3.5+ with TPE (Tree-structured Parzen Estimator) sampling |
| **Charts** | Plotly 5.24, generated server side, rendered client side |
| **Frontend** | Vanilla HTML, CSS, and JavaScript. No frameworks, no build tools |
| **Deployment** | Docker container, runs on Hugging Face Spaces |

<br>

## Algorithms

The platform automatically picks which set to run based on your task type.

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

A time budget is applied during training to keep things practical on free tier infrastructure. You still get a thorough comparison, it just won't spend twenty minutes on a single slow model.

<br>

## Theming

The interface supports dark and light modes, toggled from the top right corner. The dark theme uses glassmorphism with translucent surfaces and indigo ambient effects. The light theme uses an indigo-tinted palette with layered card shadows and visible table gridlines. All Plotly charts re-render their colors to match the active theme. Both modes are fully independent and nothing bleeds between them.

<br>

## Project structure

```
algoline/
├── .github/
│   └── workflows/
│       └── ci.yml              GitHub Actions CI pipeline
├── static/
│   └── index.html              Complete frontend (single file SPA)
├── main.py                     FastAPI backend, 30+ API endpoints
├── requirements.txt            Python dependencies
├── Dockerfile                  Production container config
└── README.md
```

<br>

## Getting started

**Run locally**

```bash
git clone https://github.com/Al1Abdullah/Algoline.git
cd Algoline
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:7860`.

**Run with Docker**

```bash
docker build -t algoline .
docker run -p 7860:7860 algoline
```

<br>

## CI/CD

The project includes a GitHub Actions workflow that runs on every push and pull request to `main`. It validates that all Python imports resolve correctly, checks that every expected API route is registered, builds the Docker image, and runs a smoke test against the container to confirm the server starts and responds.

<br>

## API reference

<details>
<summary><strong>Data</strong></summary>

<br>

| Method | Route | What it does |
|:--|:--|:--|
| `POST` | `/api/upload` | Upload a dataset with automatic profiling |
| `POST` | `/api/target` | Set the target column, auto-detect task type |

</details>

<details>
<summary><strong>Explore</strong></summary>

<br>

| Method | Route | What it does |
|:--|:--|:--|
| `POST` | `/api/explore/distribution` | Histogram |
| `POST` | `/api/explore/kde` | Kernel density estimation |
| `POST` | `/api/explore/boxplot` | Box plot |
| `POST` | `/api/explore/violin` | Violin plot |
| `POST` | `/api/explore/missing` | Missing value bar chart |
| `POST` | `/api/explore/missing_heatmap` | Missing value heatmap |
| `POST` | `/api/explore/correlation` | Correlation heatmap |
| `POST` | `/api/explore/pairplot` | Pair plot matrix |
| `POST` | `/api/explore/scatter_xy` | Scatter plot (two features) |
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
<summary><strong>Training and tuning</strong></summary>

<br>

| Method | Route | What it does |
|:--|:--|:--|
| `POST` | `/api/train` | Compare models, return leaderboard and eval plots |
| `POST` | `/api/compare` | Re-render comparison chart for a different metric |
| `POST` | `/api/tune` | Hyperparameter optimization (Optuna / random / grid) |

</details>

<details>
<summary><strong>Export</strong></summary>

<br>

| Method | Route | What it does |
|:--|:--|:--|
| `GET` | `/api/export/pipeline` | Download finalized model (.pkl) |
| `GET` | `/api/export/leaderboard` | Download model comparison (.csv) |
| `GET` | `/api/export/predictions` | Download holdout predictions (.csv) |
| `GET` | `/api/summary` | Pipeline metadata |

</details>

<br>

## License

Open source under the [MIT License](LICENSE).
