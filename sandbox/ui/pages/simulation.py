"""Pagina "Sandbox": circuito, stimoli e manipolazioni nella barra laterale e nelle schede, grafici a destra."""


import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sandbox.experiment import Experiment, from_json, prepare, run_experiment, to_json
from sandbox.simulation import DEFAULT_BACKEND, summary_table
from sandbox.stimuli import time_vector
from sandbox.ui import (analysis_panel, circuit_controls, experiment_box, gap_editor, guide_panel, manipulation_panel,
                        model_controls, protocols_panel, signature, simulation_controls, stimulus_panel,
                        synapse_controls, weight_editor)
from sandbox.ui import look, state
from sandbox.ui.look import chart, explain, section
from sandbox.ui.texts import TIPS
from sandbox.viz import (MAX_SERIES, connectivity_heatmap, firing_rate_figure, gap_heatmap, neuron_colors,
                         raster_figure, stimulus_figure, voltage_figure)

# passi × neuroni oltre cui chiediamo di ridurre durata o risoluzione (tempo e memoria per V(t))
MAX_WORK = 5_000_000 if DEFAULT_BACKEND == "numba" else 2_000_000
BACKEND_LABEL = {"numba": "Numba JIT", "numpy": "NumPy (installa numba per ~50× di velocità)"}


@st.cache_data(max_entries=8, show_spinner=False)
def run_cached(experiment_json: str):
    """L'esperimento, serializzato, fa da chiave della cache: si ricalcola solo se cambia qualcosa."""
    start = time.perf_counter()
    result, inputs = run_experiment(from_json(experiment_json))
    return result, inputs, time.perf_counter() - start


# ============================================================================ intestazione
col_title, col_intro = st.columns([0.8, 0.2], vertical_alignment="center")
col_title.title("🔬 Drosophila Neural Sandbox")
look.welcome_toggle(col_intro)
look.welcome_panel()

# ============================================================================ barra laterale
with st.sidebar:
    circuit, w_signature, circuit_warnings = circuit_controls()
    if circuit is None:
        st.info("Completa la scelta del circuito nella barra laterale per iniziare.")
        st.stop()
    neurons = circuit.neurons
    sig = signature(tuple(neurons))
    model, lif, hh = model_controls()
    syn, stp, stp_targets = synapse_controls(neurons, sig)
    sim = simulation_controls(model)
    st.divider()
    live = state.toggle("Simulazione in tempo reale", "live", True)
    run_clicked = st.button("▶ Esegui simulazione", type="primary", disabled=live, width="stretch",
                            help=TIPS["btn_run"])
    experiment_slot = st.container()

theme = look.current_theme()
st.caption(f"**{circuit.name}** — {circuit.description}")
for message in circuit_warnings:
    st.warning(message)
notes_slot = st.container()
metrics_box = st.container()
t = time_vector(sim.duration, sim.dt)

tab_sim, tab_analysis, tab_protocols, tab_guide = st.tabs(
    ["🔬 Simulazione", "📈 Analisi", "🧪 Protocolli", "📖 Guida"])

# ============================================================================ pannello di controllo
with tab_sim:
    col_controls, col_plots = st.columns([0.4, 0.6], gap="large")
    with col_controls:
        t_stim, t_manip, t_w, t_gap = st.tabs(["⚡ Stimoli", "🧬 Manipolazioni", "🔌 Sinapsi chimiche",
                                               "〰️ Gap junction"])
        with t_stim:
            defaults = circuit.default_stimuli or [{"kind": "Onda quadra", "targets": neurons[:1]}]
            specs, gain = stimulus_panel(neurons, sig, defaults, sim.duration, model)
        with t_manip:
            manipulations = manipulation_panel(neurons, sig)
        with t_w:
            weights, w_scale = weight_editor(circuit, w_signature)
        with t_gap:
            gap, gap_scale = gap_editor(circuit, w_signature)

exp = Experiment(neurons=neurons, weights=weights, gap=gap, model=model, lif=lif, hh=hh, syn=syn, stp=stp,
                 stp_targets=stp_targets, sim=sim, stimuli=specs, manipulations=manipulations, stim_gain=gain,
                 weight_scale=w_scale, gap_scale=gap_scale, name=circuit.name)

# ============================================================================ simulazione
result, inputs, elapsed = None, None, 0.0
work = t.size * len(neurons)
if work > MAX_WORK:
    st.error(f"Simulazione troppo pesante ({t.size:,} passi × {len(neurons)} neuroni): "
             "riduci la durata, aumenta dt o usa meno neuroni.")
elif live or run_clicked:
    with st.spinner("Integro le equazioni… (al primo avvio Numba compila il motore: qualche secondo)"):
        result, inputs, elapsed = run_cached(to_json(exp))
    st.session_state["last_run"] = (result, inputs, elapsed)
else:
    result, inputs, elapsed = st.session_state.get("last_run", (None, None, 0.0))
    if result is not None and result.neurons != neurons:
        result, inputs = None, None
preview = inputs if (live and inputs is not None) else prepare(exp)  # anteprima degli stimoli anche senza simulare

control = st.session_state.get("control")
if control is not None and (control.neurons != neurons):
    control = None

# ============================================================================ grafici
with tab_sim:
    with col_plots:
        section("① Protocollo di stimolo", "stimulus_plot")
        for channel in preview.channels:
            if channel.error:
                st.error(f"{channel.label}: {channel.error}")
        chart(stimulus_figure(preview.t, preview.channels, theme, gain, preview.temperature))

        if result is None:
            st.info("Premi **▶ Esegui simulazione** nella barra laterale.")
        else:
            section("② Potenziale di membrana V(t)", "voltage_plot")
            stimulated = [n for spec in specs for n in spec.get("targets", [])]
            default_visible = (neurons if len(neurons) <= MAX_SERIES
                               else list(dict.fromkeys(stimulated + neurons))[:MAX_SERIES])
            visible = state.multi("Neuroni visualizzati (max 8)", f"{sig}_visible", neurons, default_visible,
                                  max_selections=MAX_SERIES)
            colors = neuron_colors(neurons, visible, theme)
            chart(voltage_figure(result, visible, colors, theme, threshold=lif.v_thresh if model == "lif" else None))

            section("③ Raster plot e attività di popolazione", "raster_plot")
            bin_ms = state.select_slider("Bin PSTH (ms)", "psth_bin", [1, 2, 5, 10, 20, 50], 10)
            chart(raster_figure(result, colors if len(neurons) <= MAX_SERIES else {}, theme, bin_ms=bin_ms))

    if result is not None:
        st.divider()
        st.subheader("④ Connettività e frequenza di scarica")
        c1, c2, c3 = st.columns([0.22, 0.22, 0.56])
        if c1.button("📌 Usa come controllo", width="stretch", help=TIPS["btn_control"]):
            st.session_state["control"] = control = result
            st.session_state["control_label"] = f"{circuit.name} · {datetime.now():%H:%M:%S}"
        if control is not None and c2.button("✖ Rimuovi controllo", width="stretch"):
            st.session_state.pop("control", None)
            control = None
        if control is not None:
            c3.caption(f"Controllo: {st.session_state.get('control_label', '')} — barre grigie nel grafico delle "
                       "frequenze; confronto dettagliato nella scheda Analisi.")
        col_heat, col_rate = st.columns(2, gap="large")
        with col_heat:
            used_w, used_gap = inputs.weights, inputs.gap
            if used_gap.any():
                heat_chem, heat_gap = st.tabs(["Sinapsi chimiche", "Gap junction"])
                with heat_chem:
                    chart(connectivity_heatmap(used_w, neurons, theme))
                with heat_gap:
                    chart(gap_heatmap(used_gap, neurons, theme))
            else:
                chart(connectivity_heatmap(used_w, neurons, theme))
            explain("connectivity")
        with col_rate:
            chart(firing_rate_figure(result, theme, control))
            explain("rates")

        table = summary_table(result)
        with st.expander("📋 Tabella riassuntiva ed esportazione dati", expanded=False):
            st.dataframe(table, hide_index=True, width="stretch")

            # i CSV vengono generati solo al clic (callable), non a ogni aggiornamento della pagina
            def spikes_csv():
                rows = [(name, float(ts)) for name, times in zip(neurons, result.spike_times) for ts in times]
                return pd.DataFrame(rows, columns=["neurone", "tempo_ms"]).to_csv(index=False)

            def voltage_csv():
                return pd.DataFrame(result.v, columns=neurons).assign(t_ms=result.t).set_index("t_ms").to_csv()

            def weights_csv():
                return pd.DataFrame(inputs.weights, index=neurons, columns=neurons).to_csv()

            c1, c2, c3 = st.columns(3)
            for col, label, data, name in ((c1, "⬇️ Spike (CSV)", spikes_csv, "spike_times.csv"),
                                           (c2, "⬇️ V(t) (CSV)", voltage_csv, "membrane_potential.csv"),
                                           (c3, "⬇️ Matrice W (CSV)", weights_csv, "weights.csv")):
                col.download_button(label, data, name, "text/csv", on_click="ignore", width="stretch")
    else:
        table = None

# ============================================================================ altre schede
with tab_analysis:
    if result is None:
        st.info("Esegui una simulazione per analizzarla.")
    else:
        analysis_panel(result, control, sig, theme)

with tab_protocols:
    protocols_panel(exp, theme, sig)

with tab_guide:
    guide_panel(model)

# ============================================================================ riepilogo
if preview.notes:
    notes_slot.info("**Manipolazioni attive** — " + " · ".join(preview.notes))
if result is not None:
    with metrics_box:
        m1, m2, m3, m4, m5 = st.columns(5)
        n_exc, n_inh = int((inputs.weights > 0).sum()), int((inputs.weights < 0).sum())
        n_gap = int((np.triu(inputs.gap) > 0).sum())
        m1.metric("Neuroni", len(neurons))
        m2.metric("Sinapsi (E / I / el.)", f"{n_exc} / {n_inh} / {n_gap}")
        m3.metric("Spike totali", int(table["Spike"].sum()))
        m4.metric("Frequenza media", f"{table['Frequenza (Hz)'].mean():.1f} Hz")
        m5.metric("Tempo di calcolo", f"{elapsed * 1000:.0f} ms",
                  help=f"{t.size:,} passi di integrazione · motore: {BACKEND_LABEL[result.backend]}")

with experiment_slot:
    experiment_box(exp, circuit, table)

state.remember()  # i valori dei widget sopravvivono al passaggio nella pagina Dati
