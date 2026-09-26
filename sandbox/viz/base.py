"""Elementi condivisi dai grafici: stile, decimazione dei tracciati, etichette dirette."""

import numpy as np
import plotly.graph_objects as go

from .theme import FONT_FAMILY, Theme

MAX_POINTS = 4000  # punti per traccia dopo la decimazione


def minmax_decimate(t: np.ndarray, y: np.ndarray, max_points: int = MAX_POINTS):
    """Riduce i punti tenendo minimo e massimo di ogni finestra: gli spike restano visibili."""
    if t.size <= max_points:
        return t, y
    per_bin = int(np.ceil(t.size / (max_points // 2)))
    n_bins = t.size // per_bin
    tt = t[: n_bins * per_bin].reshape(n_bins, per_bin)
    yy = y[: n_bins * per_bin].reshape(n_bins, per_bin)
    i_min, i_max = yy.argmin(axis=1), yy.argmax(axis=1)
    first, second = np.minimum(i_min, i_max), np.maximum(i_min, i_max)
    rows = np.arange(n_bins)
    t_out = np.column_stack([tt[rows, first], tt[rows, second]]).ravel()
    y_out = np.column_stack([yy[rows, first], yy[rows, second]]).ravel()
    return t_out, y_out


def with_alpha(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, theme: Theme, height: int, legend: bool = True) -> go.Figure:
    """Stile comune: sfondo trasparente, griglia hairline, testo con gli inchiostri del tema."""
    fig.update_layout(
        template="none",
        height=height,
        margin=dict(l=64, r=16, t=40 if legend else 16, b=48),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=13, color=theme.ink_secondary),
        hoverlabel=dict(bgcolor=theme.surface, bordercolor=theme.axis,
                        font=dict(family=FONT_FAMILY, color=theme.ink)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=theme.ink_secondary), bgcolor="rgba(0,0,0,0)"),
    )
    axis_common = dict(zeroline=False, linecolor=theme.axis, linewidth=1, showline=True,
                       ticks="outside", tickcolor=theme.axis, ticklen=4,
                       tickfont=dict(color=theme.muted), title_font=dict(color=theme.ink_secondary, size=12))
    fig.update_xaxes(showgrid=False, **axis_common)
    fig.update_yaxes(showgrid=True, gridcolor=theme.grid, gridwidth=1, **axis_common)
    return fig


def peak_label(fig: go.Figure, x, y, text: str, theme: Theme, row=None, col=None) -> None:
    """Etichetta diretta sul massimo di una serie: utile quando più serie finiscono sullo stesso valore."""
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.any(y):
        return
    k = int(np.nanargmax(y))
    kwargs = dict(row=row, col=col) if row is not None else {}
    fig.add_annotation(x=x[k], y=y[k], text=text, showarrow=False, yshift=10, xanchor="left", xshift=4,
                       font=dict(color=theme.ink_secondary, size=11), **kwargs)


def end_label(fig: go.Figure, x, y, text: str, theme: Theme, row=None, col=None, yshift: int = 0) -> None:
    """Etichetta diretta alla fine di una serie (testo nell'inchiostro, non nel colore della serie).

    ``yshift`` separa le etichette di serie che terminano su valori vicini.
    """
    if len(x) == 0:
        return
    kwargs = dict(row=row, col=col) if row is not None else {}
    fig.add_annotation(x=x[-1], y=y[-1], text=text, showarrow=False, xanchor="left", xshift=4, yshift=yshift,
                       font=dict(color=theme.ink_secondary, size=11), **kwargs)
