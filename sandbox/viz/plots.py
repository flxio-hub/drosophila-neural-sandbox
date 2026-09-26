"""Grafici principali della Sandbox.

1. Protocollo di stimolo: segnali fisici, correnti, conduttanze e temperatura, ognuno
   nel proprio pannello con la propria unità (mai due scale y sullo stesso grafico)
2. Potenziali di membrana V(t) sovrapposti
3. Raster plot degli spike + frequenza di popolazione
4. Heatmap delle matrici sinaptiche (chimica ed elettrica) e frequenza media per neurone
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..simulation import SimulationResult, firing_rates, population_rate
from .base import minmax_decimate, style, with_alpha
from .theme import Theme


def stimulus_figure(t: np.ndarray, channels: list, theme: Theme, gain: float = 1.0,
                    temperature: np.ndarray | None = None) -> go.Figure:
    """Grafico 1 — un pannello per grandezza fisica: segnale sensoriale/luce, corrente (pA),
    conduttanza (nS), temperatura (°C). ``channels`` sono ChannelOutput; le correnti sono
    mostrate già moltiplicate per il guadagno globale."""
    panels = []
    if any(ch.signal is not None for ch in channels):
        panels.append(("signal", "Segnale (norm.)", 0.8))
    if any(ch.unit == "pA" for ch in channels) or not channels:
        panels.append(("pA", "Corrente [pA]", 1.2))
    if any(ch.unit == "nS" for ch in channels):
        panels.append(("nS", "Conduttanza [nS]", 1.0))
    if temperature is not None:
        panels.append(("temp", "Temperatura [°C]", 0.7))
    heights = [p[2] for p in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[h / sum(heights) for h in heights])
    row_of = {p[0]: i + 1 for i, p in enumerate(panels)}

    for k, ch in enumerate(channels):
        color = theme.series[k % len(theme.series)]
        shown = False
        if ch.signal is not None:
            ts, ys = minmax_decimate(t, ch.signal)
            fig.add_trace(go.Scatter(
                x=ts, y=ys, mode="lines", line=dict(color=color, width=2), name=ch.label, legendgroup=ch.label,
                showlegend=True, hovertemplate=f"{ch.signal_label or 'segnale'}: %{{y:.3f}}<extra>{ch.label}</extra>",
            ), row=row_of["signal"], col=1)
            shown = True
        scale = gain if ch.unit == "pA" else 1.0
        tc, yc = minmax_decimate(t, ch.trace * scale)
        unit = "pA" if ch.unit == "pA" else "nS"
        extra = f" · E_rev {ch.e_rev:g} mV" if ch.e_rev is not None else ""
        fig.add_trace(go.Scatter(
            x=tc, y=yc, mode="lines", line=dict(color=color, width=2), name=ch.label, legendgroup=ch.label,
            showlegend=not shown, hovertemplate=f"%{{y:.2f}} {unit}{extra}<extra>{ch.label}</extra>",
        ), row=row_of[ch.unit], col=1)
    if temperature is not None:
        tt, yt = minmax_decimate(t, temperature)
        fig.add_trace(go.Scatter(x=tt, y=yt, mode="lines", line=dict(color=theme.ink_secondary, width=2),
                                 name="Temperatura", showlegend=False,
                                 hovertemplate="%{y:.1f} °C<extra>temperatura</extra>"),
                      row=row_of["temp"], col=1)
    for key, title, _ in panels:
        fig.update_yaxes(title_text=title, row=row_of[key], col=1)
    fig.update_xaxes(title_text="Tempo (ms)", row=len(panels), col=1)
    fig.update_layout(hovermode="x unified")
    return style(fig, theme, height=160 + 130 * len(panels))


def voltage_figure(result: SimulationResult, visible: list[str], colors: dict[str, str],
                   theme: Theme, threshold: float | None = None) -> go.Figure:
    """Grafico 2 — potenziali di membrana sovrapposti dei neuroni selezionati."""
    fig = go.Figure()
    for name in visible:
        i = result.neurons.index(name)
        ts, vs = minmax_decimate(result.t, result.v[:, i])
        fig.add_trace(go.Scattergl(
            x=ts, y=vs, mode="lines", name=name, line=dict(color=colors.get(name, theme.ink), width=1.5),
            hovertemplate="%{y:.1f} mV<extra>" + name + "</extra>",
        ))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=theme.muted, width=1),
                      annotation=dict(text="soglia di base", font=dict(color=theme.muted, size=11)),
                      annotation_position="top left")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Tempo (ms)")
    fig.update_yaxes(title_text="V(t) [mV]")
    return style(fig, theme, height=340)


def raster_figure(result: SimulationResult, colors: dict[str, str], theme: Theme,
                  bin_ms: float = 10.0) -> go.Figure:
    """Grafico 3 — raster degli spike (una riga per neurone) e PSTH di popolazione."""
    n = len(result.neurons)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.72, 0.28])
    tick_size = int(np.clip(260 / max(n, 1), 6, 18))
    for i, (name, spikes) in enumerate(zip(result.neurons, result.spike_times)):
        color = colors.get(name, theme.ink_secondary)
        fig.add_trace(go.Scatter(
            x=spikes, y=np.full(spikes.size, i), mode="markers", name=name, showlegend=False,
            marker=dict(symbol="line-ns-open", size=tick_size, color=color, line=dict(width=1.5, color=color)),
            hovertemplate="%{x:.1f} ms<extra>" + name + "</extra>",
        ), row=1, col=1)
    centers, rate = population_rate(result, bin_ms)
    fig.add_trace(go.Scatter(
        x=centers, y=rate, mode="lines", line=dict(color=theme.series[0], width=2, shape="hvh"),
        fill="tozeroy", fillcolor=with_alpha(theme.series[0], 0.18), showlegend=False,
        name="popolazione", hovertemplate="%{y:.1f} Hz<extra>popolazione</extra>",
    ), row=2, col=1)
    style(fig, theme, height=int(np.clip(140 + 24 * n, 320, 900)), legend=False)
    # range esplicito (invertito: primo neurone in alto) così anche i neuroni silenti hanno la loro riga;
    # la griglia orizzontale fa da guida per ogni riga del raster
    fig.update_yaxes(tickvals=list(range(n)), ticktext=result.neurons, range=[n - 0.5, -0.5], row=1, col=1)
    fig.update_yaxes(title_text="Hz / neurone", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(range=[0, result.duration], row=1, col=1)
    fig.update_xaxes(title_text="Tempo (ms)", row=2, col=1)
    return fig


def _matrix_heatmap(matrix, neurons, theme, colorscale, zmin, zmax, zmid, x_title, y_title, hover):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=neurons, y=neurons, zmin=zmin, zmax=zmax, zmid=zmid,
        colorscale=[list(stop) for stop in colorscale], xgap=2, ygap=2,
        colorbar=dict(title=dict(text="nS", font=dict(color=theme.ink_secondary)),
                      tickfont=dict(color=theme.muted), outlinewidth=0, thickness=12),
        hovertemplate=hover,
    ))
    style(fig, theme, height=int(np.clip(160 + 28 * len(neurons), 320, 720)), legend=False)
    fig.update_yaxes(autorange="reversed", title_text=y_title, showgrid=False, showline=False, ticks="")
    fig.update_xaxes(title_text=x_title, tickangle=-45 if len(neurons) > 6 else 0,
                     showgrid=False, showline=False, ticks="")
    return fig


def connectivity_heatmap(weights: np.ndarray, neurons: list[str], theme: Theme) -> go.Figure:
    """Grafico 4a — sinapsi chimiche W[pre, post] con scala divergente centrata sullo zero."""
    limit = float(np.abs(weights).max()) or 1.0
    return _matrix_heatmap(weights, neurons, theme, theme.diverging, -limit, limit, 0, "postsinaptico",
                           "presinaptico", "pre: %{y}<br>post: %{x}<br>W = %{z:.2f} nS<extra></extra>")


def gap_heatmap(gap: np.ndarray, neurons: list[str], theme: Theme) -> go.Figure:
    """Gap junction (simmetriche, ≥ 0): scala sequenziale a un solo tono."""
    return _matrix_heatmap(gap, neurons, theme, theme.sequential, 0.0, float(gap.max()) or 1.0, None,
                           "neurone", "neurone", "%{y} ↔ %{x}<br>G = %{z:.2f} nS<extra></extra>")


def firing_rate_figure(result: SimulationResult, theme: Theme, control: SimulationResult | None = None) -> go.Figure:
    """Grafico 4b — frequenza di scarica media (Hz) per neurone, eventualmente accanto al controllo."""
    rates = firing_rates(result)
    show_values = len(result.neurons) <= 16
    fig = go.Figure()
    if control is not None:
        control_rates = firing_rates(control)
        fig.add_trace(go.Bar(
            x=control_rates, y=result.neurons, orientation="h", name="Controllo",
            marker=dict(color=theme.control), hovertemplate="%{y}: %{x:.1f} Hz<extra>controllo</extra>",
        ))
    fig.add_trace(go.Bar(
        x=rates, y=result.neurons, orientation="h", name="Attuale", marker=dict(color=theme.series[0]),
        text=[f"{r:.0f}" for r in rates] if show_values else None, textposition="outside",
        textfont=dict(color=theme.ink_secondary), cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f} Hz<extra>attuale</extra>",
    ))
    style(fig, theme, height=int(np.clip(160 + (40 if control else 28) * len(result.neurons), 320, 900)),
          legend=control is not None)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Frequenza media (Hz)", showgrid=True, gridcolor=theme.grid, rangemode="tozero")
    fig.update_layout(barcornerradius=4, barmode="group", bargap=0.35, bargroupgap=0.1)
    return fig
