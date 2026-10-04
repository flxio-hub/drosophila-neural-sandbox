"""Grafici della pagina Dati: composizione della selezione di neuroni e anteprima degli stimoli della libreria."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..network.connectome import NT_SIGN
from .base import minmax_decimate, style
from .theme import Theme


def _nt_color(label: str, theme: Theme) -> str:
    """Colore semantico del trasmettitore, coerente con la heatmap di W: eccitatorio rosso, inibitorio blu."""
    sign = NT_SIGN.get(str(label).lower())
    return theme.exc if sign == 1.0 else theme.inh if sign == -1.0 else theme.control


def category_figure(counts: pd.Series, theme: Theme, x_title: str, transmitters: bool = False) -> go.Figure:
    """Barre orizzontali (max 8 categorie, il resto in "altro"). Un solo colore, o il segno del trasmettitore."""
    labels = [str(i) for i in counts.index]
    colors = ([_nt_color(label, theme) for label in labels] if transmitters
              else [theme.control if label == "altro" else theme.series[0] for label in labels])
    fig = go.Figure(go.Bar(
        x=counts.to_numpy(), y=labels, orientation="h", marker=dict(color=colors),
        text=[f"{v:,.0f}".replace(",", ".") for v in counts.to_numpy()], textposition="outside",
        textfont=dict(color=theme.ink_secondary), cliponaxis=False,
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    style(fig, theme, height=int(np.clip(90 + 30 * len(labels), 200, 360)), legend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor=theme.grid, rangemode="tozero")
    fig.update_layout(barcornerradius=4, bargap=0.35, margin=dict(l=120, r=48, t=16, b=48))
    return fig


def waveform_figure(t, values, theme: Theme, name: str) -> go.Figure:
    """Forma d'onda di uno stimolo della libreria (colore "stimolo esterno", con etichetta diretta)."""
    tt, vv = minmax_decimate(np.asarray(t, dtype=float), np.asarray(values, dtype=float))
    fig = go.Figure(go.Scatter(x=tt, y=vv, mode="lines", name=name, line=dict(color=theme.stim, width=1.6),
                               hovertemplate="%{x:.1f} ms: %{y:.2f} pA<extra></extra>"))
    style(fig, theme, height=260, legend=False)
    fig.update_xaxes(title_text="Tempo (ms)")
    fig.update_yaxes(title_text="Corrente (pA)")
    return fig
