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

# Algoline

**End to end machine learning, simplified.**

Upload your data. Explore it visually. Train, compare, and tune models automatically.
Export production ready pipelines. All from your browser.

[Live Demo](https://huggingface.co/spaces/Al1Abdullah/AutoML)

</div>

<br>

## What It Does

Algoline takes a raw dataset and walks you through every step of the machine learning workflow without writing a single line of code. It handles classification and regression tasks, automatically selects the best performing model from a pool of algorithms, and lets you fine tune it with advanced optimization. Every decision point is backed by interactive charts, statistical summaries, and clear visual feedback.

<br>

## The Workflow

### 1. Data

Drop a CSV, TSV, or Excel file. Algoline immediately profiles your dataset and gives you four key metrics at a glance: total rows, total columns, missing value count, and duplicate count. It then generates automatic insights about your data quality, flags potential issues, and shows you a full preview with column types and descriptive statistics.

You pick a target column, and Algoline automatically infers whether your problem is classification or regression based on the target's distribution.

### 2. Explore

Eighteen interactive visualizations organized into five categories help you understand your data before modeling begins.

**Distribution Analysis** includes histograms, KDE plots, box plots, and violin plots for understanding how individual features are spread.

**Missing Data** shows missing value bar charts and heatmaps so you can spot gaps and patterns of missingness.

**Correlation and Relationships** covers correlation heatmaps, pair plots, scatter plots, and joint plots for identifying which features move together.

**Target Analysis** provides count plots, pie charts, class distributions, target histograms, and mean target per category breakdowns.

**Advanced** visualizations include scatter vs index plots (for detecting data ordering effects), grouped box plots, and faceted small multiples for multivariate exploration.

Every chart is rendered with Plotly, fully interactive, and adapts to your current theme.

### 3. Build

This is where the automation lives. Configure your preprocessing options (duplicate removal, normalization, outlier handling, multicollinearity filtering, skewness transforms, polynomial features, and class imbalance correction), set your test split and cross validation folds, and hit train.

Algoline uses PyCaret under the hood to compare every relevant algorithm, then ranks them on a leaderboard. You see the best model highlighted, a metric comparison bar chart, and full evaluation plots including confusion matrices, prediction distributions, and performance curves.

If the automatic selection is close, you can fine tune the winner using Bayesian optimization (Optuna), random search, or grid search, with configurable iteration counts.

### 4. Export

Download three artifacts when you are satisfied.

**Pipeline** is a serialized `.pkl` file containing the finalized model, ready to load and predict on new data in any Python environment.

**Leaderboard** is a CSV of all compared models and their cross validated metrics.

**Predictions** is a CSV of the model's predictions on the hold out test set.

<br>

## Tech Stack

| Layer | Technology |
|:--|:--|
| Backend | FastAPI, Python 3.10 |
| ML Engine | PyCaret 3.3 (wraps scikit learn, XGBoost, LightGBM, CatBoost) |
| Hyperparameter Tuning | Optuna 3.5+ |
| Visualization | Plotly 5.24 |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Deployment | Docker, Hugging Face Spaces |

<br>

## Running Locally

```bash
git clone https://huggingface.co/spaces/Al1Abdullah/AutoML
cd AutoML
pip install -r requirements.txt
python main.py
```

The app starts on `http://localhost:7860`.

<br>

## Docker

```bash
docker build -t algoline .
docker run -p 7860:7860 algoline
```

<br>

## Project Structure

```
algoline/
├── main.py              Backend server with all API endpoints
├── static/
│   └── index.html       Complete frontend (single page application)
├── requirements.txt     Python dependencies
├── Dockerfile           Container configuration
└── README.md
```

<br>

## API Reference

**Data**
| Method | Endpoint | Purpose |
|:--|:--|:--|
| POST | `/api/upload` | Upload a dataset (CSV, TSV, XLSX) |
| POST | `/api/target` | Set the target column |

**Explore**
| Method | Endpoint | Purpose |
|:--|:--|:--|
| POST | `/api/explore/distribution` | Histogram for a feature |
| POST | `/api/explore/correlation` | Correlation heatmap |
| POST | `/api/explore/boxplot` | Box plot for a feature |
| POST | `/api/explore/scatter` | Scatter matrix |
| POST | `/api/explore/quality` | Feature quality table |
| POST | `/api/explore/target` | Target distribution chart |
| POST | `/api/explore/missing` | Missing value bar chart |
| POST | `/api/explore/kde` | KDE density plot |
| POST | `/api/explore/violin` | Violin plot |
| POST | `/api/explore/pairplot` | Pair plot matrix |
| POST | `/api/explore/scatter_xy` | Two feature scatter |
| POST | `/api/explore/jointplot` | Joint distribution plot |
| POST | `/api/explore/countplot` | Category count plot |
| POST | `/api/explore/pie` | Pie chart |
| POST | `/api/explore/mean_target` | Mean target per category |
| POST | `/api/explore/grouped_box` | Grouped box plot |
| POST | `/api/explore/facetgrid` | Faceted small multiples |

**Build**
| Method | Endpoint | Purpose |
|:--|:--|:--|
| POST | `/api/train` | Train and compare all models |
| POST | `/api/compare` | Get comparison chart for a metric |
| POST | `/api/tune` | Tune the best model |

**Export**
| Method | Endpoint | Purpose |
|:--|:--|:--|
| GET | `/api/export/pipeline` | Download trained model (.pkl) |
| GET | `/api/export/leaderboard` | Download leaderboard (.csv) |
| GET | `/api/export/predictions` | Download predictions (.csv) |
| GET | `/api/summary` | Get pipeline summary |

<br>

## Design

The interface ships with both dark and light themes. The dark theme uses glassmorphism with translucent surfaces, indigo accent gradients, and subtle ambient lighting effects. The light theme uses an indigo tinted palette with layered shadows for depth, visible gridlines, and a gradient sidebar. You can switch between them with the toggle in the top right corner. Every chart, card, table, and interactive element adapts to both themes without visual compromises.

<br>

## License

This project is open source.

<br>

<div align="center">

Built by [Al1Abdullah](https://huggingface.co/Al1Abdullah)

</div>
