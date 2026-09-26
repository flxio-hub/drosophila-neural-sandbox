"""Grafici della scheda Protocolli."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import end_label, minmax_decimate, style, with_alpha
from .theme import Theme


def fi_figure(table: pd.DataFrame, theme: Theme, theory: tuple | None = None) -> go.Figure:
    """Curva f-I: frequenza iniziale, a regime e media (più la teoria del LIF, se disponibile)."""
    fig = go.Figure()
    series = [("Frequenza iniziale (Hz)", "iniziale", theme.series[1], 0),
              ("Frequenza a regime (Hz)", "a regime", theme.series[0], -9),
              ("Frequenza media (Hz)", "media", theme.series[2], 9)]
    x = table["Corrente (pA)"].to_numpy()
    for column, label, color, shift in series:
        y = table[column].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label, line=dict(color=color, width=2),
                                 marker=dict(size=8, line=dict(width=2, color=theme.surface)),
                                 hovertemplate="%{x:g} pA: %{y:.1f} Hz<extra>" + label + "</extra>"))
        finite = np.isfinite(y)
        end_label(fig, x[finite], y[finite], label, theme, yshift=shift)
    if theory is not None:
        fig.add_trace(go.Scatter(x=theory[0], y=theory[1], mode="lines", name="teoria LIF (Δθ = 0)",
                                 line=dict(color=theme.control, width=1.5),
                                 hovertemplate="%{x:.1f} pA: %{y:.1f} Hz<extra>teoria</extra>"))
    style(fig, theme, height=360)
    fig.update_xaxes(title_text="Corrente iniettata (pA)")
    fig.update_yaxes(title_text="Frequenza (Hz)", rangemode="tozero")
    fig.update_layout(margin=dict(r=80))
    return fig


def traces_figure(t, examples: dict, theme: Theme, window: tuple[float, float]) -> go.Figure:
    """Tracciati d'esempio della curva f-I, uno per pannello (piccoli multipli)."""
    fig = make_subplots(rows=len(examples), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[f"{c:g} pA" for c in examples])
    for row, (current, v) in enumerate(examples.items(), start=1):
        tx, yy = minmax_decimate(t, v)
        fig.add_trace(go.Scattergl(x=tx, y=yy, mode="lines", line=dict(color=theme.series[0], width=1.2),
                                   showlegend=False, hovertemplate="%{y:.1f} mV<extra>" + f"{current:g} pA</extra>"),
                      row=row, col=1)
        fig.add_vrect(x0=window[0], x1=window[1], fillcolor=with_alpha(theme.series[0], 0.06), line_width=0,
                      row=row, col=1)
    style(fig, theme, height=120 + 130 * len(examples), legend=False)
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=theme.ink_secondary, size=12)
    fig.update_xaxes(title_text="Tempo (ms)", row=len(examples), col=1)
    fig.update_yaxes(title_text="mV")
    return fig


def sweep_lines_figure(table: pd.DataFrame, param: str, metric: str, colors: dict, theme: Theme) -> go.Figure:
    fig = go.Figure()
    for name, group in table.groupby("Neurone", sort=False):
        group = group.sort_values(param)
        x, y = group[param].to_numpy(), group[metric].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                                 line=dict(color=colors.get(name, theme.ink_secondary), width=2),
                                 marker=dict(size=8, line=dict(width=2, color=theme.surface)),
                                 hovertemplate=f"%{{x:g}}: %{{y:.2f}}<extra>{name}</extra>"))
        if table["Neurone"].nunique() <= 4:
            finite = np.isfinite(y)
            end_label(fig, x[finite], y[finite], name, theme)
    style(fig, theme, height=380)
    fig.update_xaxes(title_text=param)
    fig.update_yaxes(title_text=metric)
    fig.update_layout(margin=dict(r=80))
    return fig


def sweep_heatmap_figure(table: pd.DataFrame, param: str, param2: str, metric: str, neuron: str,
                         theme: Theme) -> go.Figure:
    sub = table[table["Neurone"] == neuron].pivot_table(index=param2, columns=param, values=metric)
    fig = go.Figure(go.Heatmap(
        z=sub.to_numpy(), x=[f"{v:g}" for v in sub.columns], y=[f"{v:g}" for v in sub.index],
        colorscale=[list(s) for s in theme.sequential], xgap=2, ygap=2,
        colorbar=dict(title=dict(text=metric, font=dict(color=theme.ink_secondary)),
                      tickfont=dict(color=theme.muted), outlinewidth=0, thickness=12),
        hovertemplate=f"{param}: %{{x}}<br>{param2}: %{{y}}<br>{metric}: %{{z:.2f}}<extra>{neuron}</extra>",
    ))
    style(fig, theme, height=420, legend=False)
    fig.update_xaxes(title_text=param, type="category", showline=False, ticks="")
    fig.update_yaxes(title_text=param2, type="category", showgrid=False, showline=False, ticks="")
    return fig


def trials_figure(trials: list[np.ndarray], psth: tuple, duration: float, name: str, theme: Theme) -> go.Figure:
    """Raster per trial e PSTH medio ± errore standard."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.62, 0.38])
    n = len(trials)
    tick = int(np.clip(260 / max(n, 1), 5, 16))
    x = np.concatenate(trials) if trials else np.array([])
    y = np.concatenate([np.full(s.size, k + 1) for k, s in enumerate(trials)]) if trials else np.array([])
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", showlegend=False,
                             marker=dict(symbol="line-ns-open", size=tick, color=theme.ink_secondary,
                                         line=dict(width=1.5, color=theme.ink_secondary)),
                             hovertemplate="trial %{y}: %{x:.1f} ms<extra></extra>"), row=1, col=1)
    centers, mean, sem = psth
    fig.add_trace(go.Scatter(x=np.concatenate([centers, centers[::-1]]),
                             y=np.concatenate([mean + sem, (mean - sem)[::-1]]), fill="toself",
                             fillcolor=with_alpha(theme.series[0], 0.2), line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=centers, y=mean, mode="lines", line=dict(color=theme.series[0], width=2, shape="hvh"),
                             showlegend=False, hovertemplate="%{y:.1f} Hz<extra>media</extra>"), row=2, col=1)
    style(fig, theme, height=int(np.clip(260 + 12 * n, 360, 760)), legend=False)
    fig.update_yaxes(title_text="Trial", range=[n + 0.5, 0.5], row=1, col=1)
    fig.update_yaxes(title_text="Hz (media ± SEM)", rangemode="tozero", row=2, col=1)
    fig.update_xaxes(range=[0, duration], row=1, col=1)
    fig.update_xaxes(title_text=f"Tempo (ms) — {name}", row=2, col=1)
    return fig


def ppr_figure(pairs: pd.DataFrame, theme: Theme) -> go.Figure:
    fig = go.Figure()
    for column, label, color, shift in (("PPR (efficacia)", "efficacia sinaptica", theme.series[0], 9),
                                        ("PPR (PSP)", "ampiezza PSP", theme.series[1], -9)):
        x, y = pairs["Intervallo (ms)"].to_numpy(), pairs[column].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label, line=dict(color=color, width=2),
                                 marker=dict(size=8, line=dict(width=2, color=theme.surface)),
                                 hovertemplate="%{x:g} ms: %{y:.2f}<extra>" + label + "</extra>"))
        end_label(fig, x, y, label, theme, yshift=shift)
    fig.add_hline(y=1.0, line=dict(color=theme.muted, width=1))
    style(fig, theme, height=340)
    fig.update_xaxes(title_text="Intervallo fra i due spike (ms)", type="log")
    fig.update_yaxes(title_text="Rapporto paired-pulse (2° / 1°)")
    fig.update_layout(margin=dict(r=110))
    return fig


def train_figure(test, theme: Theme) -> go.Figure:
    """Risposta a un treno regolare: V postsinaptico, conduttanza e ampiezze normalizzate."""
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.1, row_heights=[0.4, 0.25, 0.35])
    tx, vv = minmax_decimate(test.t, test.v)
    fig.add_trace(go.Scatter(x=tx, y=vv, mode="lines", line=dict(color=theme.series[0], width=1.5),
                             showlegend=False, hovertemplate="%{y:.2f} mV<extra>V post</extra>"), row=1, col=1)
    tg, gg = minmax_decimate(test.t, test.g)
    fig.add_trace(go.Scatter(x=tg, y=gg, mode="lines", line=dict(color=theme.exc, width=1.5),
                             showlegend=False, hovertemplate="%{y:.2f} nS<extra>g sinaptica</extra>"), row=2, col=1)
    train = test.train
    fig.add_trace(go.Bar(x=train["Impulso"], y=train["PSP (norm.)"], marker=dict(color=theme.series[0]),
                         showlegend=False, hovertemplate="impulso %{x}: %{y:.2f}<extra>PSP normalizzato</extra>"),
                  row=3, col=1)
    style(fig, theme, height=520, legend=False)
    fig.update_yaxes(title_text="V post [mV]", row=1, col=1)
    fig.update_yaxes(title_text="g [nS]", rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title_text="PSP / PSP₁", rangemode="tozero", row=3, col=1)
    fig.update_xaxes(title_text="Tempo (ms)", row=2, col=1)
    fig.update_xaxes(title_text="Numero dell'impulso nel treno", row=3, col=1, dtick=1)
    fig.update_layout(bargap=0.3, barcornerradius=4)
    return fig


def impedance_figure(profile, theme: Theme, theory: np.ndarray | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35])
    fig.add_trace(go.Scatter(x=profile.freqs, y=profile.magnitude, mode="lines", name="|Z| simulata",
                             line=dict(color=theme.series[0], width=2),
                             hovertemplate="%{x:.1f} Hz: %{y:.3f} GΩ<extra>|Z|</extra>"), row=1, col=1)
    if theory is not None:
        fig.add_trace(go.Scatter(x=profile.freqs, y=theory, mode="lines", name="RC passivo teorico",
                                 line=dict(color=theme.control, width=1.5),
                                 hovertemplate="%{x:.1f} Hz: %{y:.3f} GΩ<extra>teoria</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=profile.freqs, y=profile.phase, mode="lines", name="fase",
                             line=dict(color=theme.series[1], width=2), showlegend=False,
                             hovertemplate="%{x:.1f} Hz: %{y:.0f}°<extra>fase</extra>"), row=2, col=1)
    fig.add_vline(x=profile.resonance_hz, line=dict(color=theme.muted, width=1), row=1, col=1,
                  annotation=dict(text=f"f_res {profile.resonance_hz:.1f} Hz", font=dict(color=theme.muted, size=11)))
    style(fig, theme, height=460)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Frequenza (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="|Z| [GΩ = mV/pA]", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Fase (°)", row=2, col=1)
    return fig


def vclamp_figure(vc, commands: list[float], theme: Theme) -> go.Figure:
    """Correnti di Na⁺ e K⁺ per i potenziali di comando scelti (≤ 5, rampa ordinale)."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["I_Na (entrante < 0)", "I_K (uscente > 0)"])
    chosen = [c for c in commands if c in list(vc.commands)][:5]
    ramp = theme.ordinal[-len(chosen):] if chosen else ()
    for color, v_cmd in zip(ramp, sorted(chosen)):
        k = list(vc.commands).index(v_cmd)
        label = f"{v_cmd:g} mV"
        for row, current in ((1, vc.i_na[k]), (2, vc.i_k[k])):
            fig.add_trace(go.Scatter(x=vc.t, y=current, mode="lines", name=label, legendgroup=label,
                                     showlegend=row == 1, line=dict(color=color, width=2),
                                     hovertemplate="%{y:.0f} pA<extra>" + label + "</extra>"), row=row, col=1)
    style(fig, theme, height=460)
    for annotation in fig.layout.annotations:
        annotation.font = dict(color=theme.ink_secondary, size=12)
    fig.update_yaxes(title_text="pA")
    fig.update_xaxes(title_text="Tempo (ms)", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return fig


def iv_figure(iv: pd.DataFrame, theme: Theme) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (("I_Na di picco (pA)", "I_Na di picco", theme.series[0]),
                                 ("I_K a regime (pA)", "I_K a regime", theme.series[1])):
        x, y = iv["V comando (mV)"].to_numpy(), iv[column].to_numpy(dtype=float)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label, line=dict(color=color, width=2),
                                 marker=dict(size=8, line=dict(width=2, color=theme.surface)),
                                 hovertemplate="%{x:g} mV: %{y:.0f} pA<extra>" + label + "</extra>"))
        end_label(fig, x, y, label, theme)
    fig.add_hline(y=0, line=dict(color=theme.axis, width=1))
    style(fig, theme, height=360)
    fig.update_xaxes(title_text="Potenziale di comando (mV)")
    fig.update_yaxes(title_text="Corrente (pA)")
    fig.update_layout(margin=dict(r=100))
    return fig
