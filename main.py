"""Algoline — Automated Machine Learning Platform"""

import warnings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import state to initialize Plotly config on startup
import app.state  # noqa: F401

from app.routes.data import router as data_router
from app.routes.explore import router as explore_router
from app.routes.build import router as build_router
from app.routes.export import router as export_router

warnings.filterwarnings("ignore")

app = FastAPI(title="Algoline")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(data_router)
app.include_router(explore_router)
app.include_router(build_router)
app.include_router(export_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
