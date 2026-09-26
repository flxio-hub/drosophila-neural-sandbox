"""Grafici della scheda Analisi: conduttanze, frequenza istantanea, ISI, correlogrammi, STA, spettri, controllo."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..simulation import SimulationResult, synaptic_currents
from .base import minmax_decimate, peak_label, style, with_alpha
from .theme import Theme


def conductance_figure(result: SimulationResult, j: int, theme: Theme) -> go.Figure:
    """Conduttanze sinaptiche (e degli stimoli) di un neurone e le correnti che producono."""
    name = result.neurons[j]
    i_exc, i_inh = synaptic_currents(result)
    series = [("g eccitatoria", result.g_exc[:, j].astype(float), i_exc[:, j], theme.exc),
              ("g inibitoria", result.g_inh[:, j].astype(float), i_inh[:, j], theme.inh)]
    if result.g_ext is not None and np.any(result.g_ext[:, j]):
        g = result.g_ext[:, j]
        # corrente degli stimoli a conduttanza: Σg·E − Σg·V
        series.append(("g stimoli", g, result.ge_ext[:, j] - g * result.v[:, j], theme.stim))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for label, g, current, color in series:
        for row, y, unit in ((1, g, "nS"), (2, current, "pA")):
            tx, yy = minmax_decimate(result.t, y)
            fig.add_trace(go.Scattergl(
                x=tx, y=yy, mode="lines", line=dict(color=color, width=1.5), name=label, legendgroup=label,
                showlegend=row == 1, hovertemplate=f"%{{y:.2f}} {unit}<extra>{label}</extra>",
            ), row=row, col=1)
        peak_label(fig, result.t, g, label, theme, row=1, col=1)
    style(fig, theme, height=420)
    fig.update_yaxes(title_text="Conduttanza [nS]", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="Corrente [pA]", row=2, col=1)
    fig.update_xaxes(title_text=f"Tempo (ms) — {name}", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return fig


def rate_lines_figure(t: np.ndarray, rates: dict[str, np.ndarray], colors: dict[str, str], theme: Theme,
                      sigma_ms: float) -> go.Figure:
    fig = go.Figure()
    for name, rate in rates.items():
        tx, yy = minmax_decimate(t, rate)
        fig.add_trace(go.Scattergl(x=tx, y=yy, mode="lines", name=name,
                                   line=dict(color=colors.get(name, theme.ink), width=2),
                                   hovertemplate="%{y:.1f} Hz<extra>" + name + "</extra>"))
    style(fig, theme, height=320)
    fig.update_xaxes(title_text="Tempo (ms)")
    fig.update_yaxes(title_text=f"Frequenza (Hz, σ = {sigma_ms:g} ms)", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return fig


def _bars(x, y, theme, x_title, y_title, hover, color=None, height=300):
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=color or theme.series[0]), hovertemplate=hover))
    style(fig, theme, height=height, legend=False)
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    fig.update_layout(bargap=0.1, barcornerradius=2)
    return fig


def isi_figure(centers, counts, theme: Theme, name: str) -> go.Figure:
    return _bars(centers, counts, theme, f"Intervallo inter-spike (ms) — {name}", "Conteggi",
                 "ISI %{x:.1f} ms: %{y}<extra></extra>")


def ccg_figure(lags, rate, theme: Theme, ref: str, target: str) -> go.Figure:
    fig = _bars(lags, rate, theme, f"Ritardo di {target} rispetto a {ref} (ms)", f"Frequenza di {target} (Hz)",
                "ritardo %{x:.1f} ms: %{y:.1f} Hz<extra></extra>")
    fig.add_vline(x=0, line=dict(color=theme.muted, width=1))
    return fig


def sta_figure(lags, sta, theme: Theme, unit: str, n_spikes: int, name: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=lags, y=sta, mode="lines", line=dict(color=theme.series[0], width=2),
                               hovertemplate=f"%{{x:.1f}} ms: %{{y:.2f}} {unit}<extra>STA</extra>"))
    fig.add_vline(x=0, line=dict(color=theme.muted, width=1),
                  annotation=dict(text="spike", font=dict(color=theme.muted, size=11)), annotation_position="top")
    style(fig, theme, height=300, legend=False)
    fig.update_xaxes(title_text=f"Tempo relativo allo spike (ms) — {name}, {n_spikes} spike")
    fig.update_yaxes(title_text=f"Media pre-spike [{unit}]")
    return fig


def psd_figure(freqs, psd, theme: Theme, label: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=freqs, y=psd, mode="lines", line=dict(color=theme.series[0], width=2),
                               fill="tozeroy", fillcolor=with_alpha(theme.series[0], 0.15),
                               hovertemplate="%{x:.1f} Hz<extra></extra>"))
    if psd.size:
        k = int(np.argmax(psd))
        fig.add_annotation(x=freqs[k], y=psd[k], text=f"picco {freqs[k]:.1f} Hz", showarrow=False, yshift=12,
                           font=dict(color=theme.ink_secondary, size=11))
    style(fig, theme, height=300, legend=False)
    fig.update_xaxes(title_text=f"Frequenza (Hz) — {label}")
    fig.update_yaxes(title_text="Densità spettrale", rangemode="tozero")
    return fig


def voltage_overlay_figure(t, v_control, v_current, name: str, color: str, theme: Theme) -> go.Figure:
    """Tracciato attuale sopra quello di controllo (grigio neutro)."""
    fig = go.Figure()
    for label, v, c, width in (("Controllo", v_control, theme.control, 1.5), ("Attuale", v_current, color, 1.5)):
        tx, yy = minmax_decimate(t, v)
        fig.add_trace(go.Scattergl(x=tx, y=yy, mode="lines", name=label, line=dict(color=c, width=width),
                                   hovertemplate="%{y:.1f} mV<extra>" + label + "</extra>"))
    style(fig, theme, height=320)
    fig.update_xaxes(title_text=f"Tempo (ms) — {name}")
    fig.update_yaxes(title_text="V(t) [mV]")
    fig.update_layout(hovermode="x unified")
    return fig
