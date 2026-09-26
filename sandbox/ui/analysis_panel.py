"""Scheda "Analisi": approfondimenti sul risultato corrente e confronto con il controllo."""

import numpy as np
import streamlit as st

from ..simulation import (SimulationResult, compare_rates, cross_correlogram, instantaneous_rate, isi_cv,
                          isi_histogram, power_spectrum, synaptic_currents, spike_triggered_average)
from ..viz import MAX_SERIES, Theme, neuron_colors
from ..viz.analysis_plots import (ccg_figure, conductance_figure, isi_figure, psd_figure, rate_lines_figure,
                                  sta_figure, voltage_overlay_figure)
from . import state


def _chart(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def analysis_panel(result: SimulationResult, control: SimulationResult | None, sig: str, theme: Theme) -> None:
    neurons = result.neurons
    p = f"{sig}_an"

    st.subheader("Conduttanze e correnti sinaptiche")
    st.caption("Chi spinge e chi frena un neurone: g eccitatoria (rosso), inibitoria (blu) e degli stimoli a "
               "conduttanza (acqua). Le correnti sono positive quando depolarizzano.")
    name = state.select("Neurone", f"{p}_cond", neurons)
    j = neurons.index(name)
    _chart(conductance_figure(result, j, theme))
    i_exc, i_inh = synaptic_currents(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Carica eccitatoria", f"{i_exc[:, j].sum() * result.dt / 1000:.2f} pC")
    c2.metric("Carica inibitoria", f"{i_inh[:, j].sum() * result.dt / 1000:.2f} pC")
    total = abs(i_exc[:, j].sum()) + abs(i_inh[:, j].sum())
    c3.metric("Quota eccitatoria", f"{100 * abs(i_exc[:, j].sum()) / total:.0f}%" if total else "—",
              help="|Q_exc| / (|Q_exc| + |Q_inh|): bilancio eccitazione/inibizione ricevuto.")

    st.divider()
    st.subheader("Frequenza istantanea")
    c1, c2 = st.columns([0.7, 0.3])
    default = [n for n in neurons if result.spike_times[neurons.index(n)].size][:4] or neurons[:1]
    chosen = state.multi("Neuroni (max 8)", f"{p}_rate_neurons", neurons, default, where=c1,
                         max_selections=MAX_SERIES)
    sigma = state.slider("σ del kernel (ms)", f"{p}_sigma", 1.0, 100.0, 10.0, 1.0, where=c2)
    colors = neuron_colors(neurons, chosen, theme)
    rates = {n: instantaneous_rate(result.spike_times[neurons.index(n)], result.t, sigma) for n in chosen}
    _chart(rate_lines_figure(result.t, rates, colors, theme, sigma))

    st.divider()
    col_isi, col_ccg = st.columns(2, gap="large")
    with col_isi:
        st.subheader("Intervalli inter-spike")
        name = state.select("Neurone", f"{p}_isi", neurons, default[0])
        spikes = result.spike_times[neurons.index(name)]
        bin_ms = state.slider("Bin (ms)", f"{p}_isi_bin", 0.5, 20.0, 2.0, 0.5)
        centers, counts = isi_histogram(spikes, bin_ms)
        if counts.size:
            _chart(isi_figure(centers, counts, theme, name))
            cv = isi_cv(spikes)
            st.caption(f"{spikes.size} spike · ISI medio {np.diff(spikes).mean():.1f} ms · CV "
                       f"{cv:.2f} ({'regolare' if cv < 0.3 else 'irregolare' if cv > 0.7 else 'intermedio'})")
        else:
            st.info("Servono almeno 2 spike.")
    with col_ccg:
        st.subheader("Cross-correlogramma")
        c1, c2 = st.columns(2)
        ref = state.select("Riferimento", f"{p}_ccg_a", neurons, default[0], where=c1)
        target = state.select("Bersaglio", f"{p}_ccg_b", neurons, default[min(1, len(default) - 1)], where=c2)
        window = state.slider("Finestra (ms)", f"{p}_ccg_w", 5.0, 200.0, 50.0, 5.0)
        lags, rate = cross_correlogram(result.spike_times[neurons.index(ref)],
                                       result.spike_times[neurons.index(target)], window, max(window / 50, 0.5))
        _chart(ccg_figure(lags, rate, theme, ref, target))
        st.caption("Picco a ritardo positivo: il bersaglio scarica dopo il riferimento (connessione eccitatoria, "
                   "latenza sinaptica). Un avvallamento indica inibizione.")

    st.divider()
    col_sta, col_psd = st.columns(2, gap="large")
    with col_sta:
        st.subheader("Spike-triggered average")
        c1, c2 = st.columns(2)
        name = state.select("Neurone", f"{p}_sta", neurons, default[0], where=c1)
        source = state.select("Segnale", f"{p}_sta_src", ["Corrente iniettata", "Corrente sinaptica (E + I)",
                                                          "Potenziale di membrana"], where=c2)
        window = state.slider("Finestra pre-spike (ms)", f"{p}_sta_w", 5.0, 200.0, 50.0, 5.0)
        j = neurons.index(name)
        signal, unit = {"Corrente iniettata": (result.i_ext[:, j], "pA"),
                        "Corrente sinaptica (E + I)": (i_exc[:, j] + i_inh[:, j], "pA"),
                        "Potenziale di membrana": (result.v[:, j], "mV")}[source]
        lags, sta, used = spike_triggered_average(signal, result.dt, result.spike_times[j], window)
        if used:
            _chart(sta_figure(lags, sta, theme, unit, used, name))
            st.caption("Con uno stimolo di rumore gaussiano la STA stima il filtro lineare del neurone "
                       "(correlazione inversa).")
        else:
            st.info("Nessuno spike utilizzabile: servono spike dopo la finestra iniziale.")
    with col_psd:
        st.subheader("Spettro di potenza")
        options = ["Attività di popolazione"] + neurons
        target = state.select("Segnale", f"{p}_psd", options)
        if target == "Attività di popolazione":
            spikes = np.concatenate(result.spike_times)
            signal = instantaneous_rate(spikes, result.t, 2.0) / len(neurons)
            label = "attività di popolazione"
        else:
            signal, label = result.v[:, neurons.index(target)], f"V di {target}"
        if result.duration >= 200 and np.any(signal != signal.mean()):
            freqs, psd = power_spectrum(signal, result.dt)
            _chart(psd_figure(freqs, psd, theme, label))
            st.caption("Un picco netto indica un'oscillazione (es. il ritmo del CPG half-center).")
        else:
            st.info("Segnale costante o simulazione troppo breve (< 200 ms).")

    st.divider()
    st.subheader("Confronto con il controllo")
    if control is None or control.neurons != neurons:
        st.info("Nessun controllo per questo circuito: nella scheda Simulazione premi **📌 Usa come controllo**, "
                "poi cambia parametri, stimoli o manipolazioni.")
        return
    name = state.select("Neurone", f"{p}_ctrl", neurons)
    j = neurons.index(name)
    if control.t.size == result.t.size:
        _chart(voltage_overlay_figure(result.t, control.v[:, j], result.v[:, j], name,
                                      neuron_colors(neurons, [name], theme).get(name, theme.series[0]), theme))
    else:
        st.info("Durata o dt diversi dal controllo: confronto dei tracciati non disponibile.")
    st.dataframe(compare_rates(control, result), hide_index=True, width="stretch")
