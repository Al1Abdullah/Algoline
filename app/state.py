# Shared application state and configuration

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Single-user session state (free tier)
S = {}

# Color palette
COLORS = ["#6366f1","#22d3ee","#f59e0b","#a78bfa","#f43f5e",
          "#14b8a6","#e879f9","#3b82f6","#84cc16","#fb923c"]
px.defaults.color_discrete_sequence = COLORS

# Plotly template: bar outlines + dark theme defaults
_tpl = go.layout.Template()
_tpl.data.bar = [go.Bar(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.15)")))]
_tpl.data.histogram = [go.Histogram(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.12)")))]
go.layout.Template()
px.defaults.template = _tpl
