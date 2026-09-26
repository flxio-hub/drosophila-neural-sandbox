"""Scheda "Protocolli": esperimenti automatici sull'esperimento configurato nella scheda Simulazione."""

import numpy as np
import streamlit as st

from ..config import STPParams
from ..experiment import Experiment
from ..protocols import (METRICS, current_value, fi_curve, hh_voltage_clamp, impedance_profile, paired_pulse,
                         parameter_sweep, repeated_trials, sweepable_parameters)
from ..stimuli import with_defaults
from ..viz import MAX_SERIES, Theme, neuron_colors
from ..viz.protocol_plots import (fi_figure, impedance_figure, iv_figure, ppr_figure, sweep_heatmap_figure,
                                  sweep_lines_figure, traces_figure, train_figure, trials_figure, vclamp_figure)
from . import state

PROTOCOLS = {
    "Curva f-I": "Gradini di corrente di ampiezza crescente su un neurone isolato con i parametri del modello "
                 "scelto: reobase, guadagno, adattamento.",
    "Esplorazione parametrica": "Ripete l'esperimento corrente variando uno o due parametri e misura la risposta "
                                "di ogni neurone.",
    "Trial ripetuti": "Stesso esperimento con semi casuali diversi: variabilità fra trial, fattore di Fano e "
                      "affidabilità temporale degli spike.",
    "Paired-pulse (STP)": "Due spike presinaptici a intervalli crescenti e un treno regolare: depressione e "
                          "facilitazione a breve termine.",
    "Impedenza (ZAP)": "Corrente chirp sottosoglia: |Z(f)| e risonanza della membrana.",
    "Voltage clamp (HH)": "L'esperimento di Hodgkin e Huxley: gradini di potenziale imposto e correnti di Na⁺ e K⁺.",
}


def _chart(fig):
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def _store(name: str, value) -> None:
    st.session_state[f"_protocol:{name}"] = value


def _stored(name: str):
    return st.session_state.get(f"_protocol:{name}")


def _download(table, filename: str, key: str) -> None:
    st.download_button("⬇️ Tabella (CSV)", lambda: table.to_csv(index=False), filename, "text/csv",
                       on_click="ignore", key=key)


def _has_randomness(exp: Experiment) -> bool:
    random_kinds = {"Rumore gaussiano", "Input sinaptico di Poisson"}
    for spec in exp.stimuli:
        full = with_defaults(spec)
        if full["kind"] in random_kinds or (full["kind"] == "Sensoriale" and full["profile"] == "Pennacchio turbolento"):
            return True
        if full["kind"] == "Equazione custom" and "noise(" in full["expression"]:
            return True
    return exp.sim.noise_sigma > 0


# ============================================================================ singoli protocolli

def _fi(exp: Experiment, theme: Theme) -> None:
    c1, c2, c3, c4 = st.columns(4)
    lo = state.number("Da (pA)", "proto_fi_lo", -100.0, 1000.0, 0.0, 5.0, where=c1)
    hi = state.number("A (pA)", "proto_fi_hi", -100.0, 2000.0, 80.0 if exp.model == "lif" else 250.0, 5.0, where=c2)
    step = state.number("Passo (pA)", "proto_fi_step", 0.5, 500.0, 5.0 if exp.model == "lif" else 10.0, 0.5, where=c3)
    step_ms = state.number("Gradino (ms)", "proto_fi_len", 50.0, 5000.0, 500.0, 50.0, where=c4)
    if st.button("▶ Misura la curva f-I", type="primary", key="proto_fi_run"):
        currents = np.arange(lo, hi + step / 2, step)
        if currents.size > 400:
            st.error("Troppi punti: aumenta il passo.")
        else:
            with st.spinner("Simulo tutti i gradini in parallelo…"):
                _store("fi", (exp.model, fi_curve(exp.model, exp.lif, exp.hh, currents, step_ms, exp.sim.dt)))
    stored = _stored("fi")
    if not stored:
        return
    model, curve = stored
    m1, m2, m3 = st.columns(3)
    m1.metric("Reobase (griglia)", f"{curve.rheobase:g} pA" if curve.rheobase is not None else "oltre l'intervallo")
    theory = None
    if model == "lif":
        p = exp.lif
        m2.metric("Reobase teorica LIF", f"{(p.v_thresh - p.v_rest) / p.r_m:.1f} pA")
        x = np.linspace(curve.table["Corrente (pA)"].min(), curve.table["Corrente (pA)"].max(), 200)
        v_inf = p.v_rest + p.r_m * x
        with np.errstate(divide="ignore", invalid="ignore"):
            period = p.t_ref + p.tau_m * np.log((v_inf - p.v_reset) / (v_inf - p.v_thresh))
            theory = (x, np.where(v_inf > p.v_thresh, 1000.0 / period, 0.0))
    # pendenza della retta di regressione sopra la reobase (escluso il salto iniziale da 0 Hz)
    firing = curve.table[curve.table["Spike"] > 0].iloc[1:]
    slope = (np.polyfit(firing["Corrente (pA)"], firing["Frequenza a regime (Hz)"], 1)[0]
             if len(firing) >= 2 else None)
    m3.metric("Pendenza f-I", f"{slope:.2f} Hz/pA" if slope is not None else "—",
              help="Guadagno a regime sopra la reobase (regressione lineare).")
    _chart(fi_figure(curve.table, theme, theory))
    with st.expander("Tracciati d'esempio"):
        _chart(traces_figure(curve.t, curve.examples, theme, curve.window))
    st.dataframe(curve.table.round(2), hide_index=True, width="stretch")
    _download(curve.table, "curva_fI.csv", "proto_fi_dl")


def _range_inputs(label: str, name: str, value: float | None, where_cols, max_points: int):
    base = value if value not in (None, 0.0) else 1.0
    lo_default, hi_default = (0.0, 2.0 * base) if base > 0 else (2.0 * base, 0.0)
    c1, c2, c3 = where_cols
    lo = state.number(f"{label}: da", f"{name}_lo", None, None, float(lo_default), where=c1, format="%.3g")
    hi = state.number(f"{label}: a", f"{name}_hi", None, None, float(hi_default), where=c2, format="%.3g")
    n = state.number("Punti", f"{name}_n", 2, max_points, min(9, max_points), where=c3)
    return np.linspace(lo, hi, int(n))


def _sweep(exp: Experiment, theme: Theme, sig: str) -> None:
    params = sweepable_parameters(exp)
    names = list(params)
    param = state.select("Parametro", f"{sig}_sw_p", names, names[0])
    values = _range_inputs("Valori", f"{sig}_sw_{names.index(param)}", current_value(exp, param), st.columns(3), 40)
    st.caption(f"Valore attuale: {current_value(exp, param):g}" if current_value(exp, param) is not None else "")
    two_d = state.check("Aggiungi un secondo parametro (mappa 2D)", f"{sig}_sw_2d", False)
    param2, values2 = None, None
    if two_d:
        others = [n for n in names if n != param]
        param2 = state.select("Secondo parametro", f"{sig}_sw_p2", others, others[0])
        values2 = _range_inputs("Valori", f"{sig}_sw2_{names.index(param2)}", current_value(exp, param2),
                                st.columns(3), 15)
    metric = state.select("Misura", "proto_sw_metric", list(METRICS))
    runs = len(values) * (len(values2) if two_d else 1)
    st.caption(f"{runs} simulazioni da {exp.sim.duration:g} ms.")
    if st.button("▶ Esegui l'esplorazione", type="primary", key="proto_sw_run"):
        bar = st.progress(0.0, text="Simulazioni in corso…")
        table = parameter_sweep(exp, param, values, metric, param2, values2,
                                progress=lambda f: bar.progress(f, text=f"Simulazioni in corso… {f:.0%}"))
        bar.empty()
        _store("sweep", (param, param2, metric, table))
    stored = _stored("sweep")
    if not stored:
        return
    param, param2, metric, table = stored
    neurons = list(dict.fromkeys(table["Neurone"]))
    if param2 is None:
        responsive = [n for n in neurons if table.loc[table["Neurone"] == n, metric].fillna(0).abs().sum() > 0]
        chosen = state.multi("Neuroni nel grafico (max 8)", "proto_sw_show", neurons, (responsive or neurons)[:4],
                             max_selections=MAX_SERIES)
        colors = neuron_colors(neurons, chosen, theme)
        _chart(sweep_lines_figure(table[table["Neurone"].isin(chosen)], param, metric, colors, theme))
    else:
        neuron = state.select("Neurone", "proto_sw_neuron", neurons)
        _chart(sweep_heatmap_figure(table, param, param2, metric, neuron, theme))
    with st.expander("Tabella dei risultati"):
        st.dataframe(table.round(3), hide_index=True, width="stretch")
    _download(table, "esplorazione_parametrica.csv", "proto_sw_dl")


def _trials(exp: Experiment, theme: Theme) -> None:
    if not _has_randomness(exp):
        st.warning("L'esperimento è deterministico: tutti i trial saranno identici (Fano = 0, affidabilità = 1). "
                   "Aggiungi rumore di membrana di fondo nella barra laterale o uno stimolo stocastico.")
    c1, c2 = st.columns(2)
    n_trials = state.slider("Numero di trial", "proto_tr_n", 2, 60, 20, where=c1)
    frozen = state.check("Stimolo congelato (\"frozen noise\")", "proto_tr_frozen", False, where=c2,
                         help="Gli stimoli casuali restano identici fra i trial; cambia solo il rumore di fondo. "
                              "Paradigma di Mainen & Sejnowski (1995).")
    if st.button("▶ Esegui i trial", type="primary", key="proto_tr_run"):
        bar = st.progress(0.0, text="Trial in corso…")
        result = repeated_trials(exp, n_trials, frozen,
                                 progress=lambda f: bar.progress(f, text=f"Trial in corso… {f:.0%}"))
        bar.empty()
        _store("trials", result)
    result = _stored("trials")
    if not result:
        return
    c1, c2 = st.columns([0.6, 0.4])
    active = [n for j, n in enumerate(result.neurons) if result.counts[:, j].sum()] or result.neurons
    name = state.select("Neurone", "proto_tr_neuron", result.neurons, active[0], where=c1)
    bin_ms = state.select_slider("Bin PSTH (ms)", "proto_tr_bin", [2, 5, 10, 20, 50], 10, where=c2)
    j = result.neurons.index(name)
    _chart(trials_figure(result.neuron_trials(j), result.psth(j, bin_ms), result.duration, name, theme))
    table = result.statistics()
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Fattore di Fano: varianza/media dei conteggi (1 = Poisson). Affidabilità (Schreiber 2003): "
               "correlazione media fra trial degli spike filtrati (1 = spike identici).")
    _download(table, "trial_statistiche.csv", "proto_tr_dl")


def _paired_pulse(exp: Experiment, theme: Theme) -> None:
    base = exp.stp
    nonzero = np.abs(exp.weights[exp.weights != 0])
    c1, c2, c3, c4 = st.columns(4)
    U = state.number("U", "proto_pp_U", 0.01, 1.0, float(base.U), 0.01, where=c1)
    tau_rec = state.number("τ_rec (ms)", "proto_pp_rec", 1.0, 5000.0, float(base.tau_rec), 10.0, where=c2)
    tau_fac = state.number("τ_fac (ms)", "proto_pp_fac", 0.0, 5000.0, float(base.tau_fac), 10.0, where=c3)
    weight = state.number("|W| (nS)", "proto_pp_w", 0.01, 50.0,
                          float(np.median(nonzero)) if nonzero.size else 1.0, 0.1, where=c4)
    c1, c2, c3 = st.columns(3)
    sign = state.radio("Sinapsi", "proto_pp_sign", ["Eccitatoria", "Inibitoria"], horizontal=True, where=c1)
    train_hz = state.number("Treno (Hz)", "proto_pp_hz", 1.0, 200.0, 20.0, 1.0, where=c2)
    n_train = state.number("Impulsi nel treno", "proto_pp_n", 2, 50, 10, where=c3)
    isis_text = state.text("Intervalli paired-pulse (ms)", "proto_pp_isis", "10, 20, 50, 100, 200, 500, 1000")
    try:
        isis = sorted({float(x) for x in isis_text.replace(";", ",").split(",") if x.strip()})
    except ValueError:
        st.error("Intervalli non validi: usa numeri separati da virgole.")
        return
    stp = STPParams(enabled=True, U=U, tau_rec=tau_rec, tau_fac=tau_fac)
    test = paired_pulse(weight if sign == "Eccitatoria" else -weight, stp, exp.syn, exp.lif, isis, train_hz,
                        int(n_train), dt=min(exp.sim.dt, 0.1))
    st.caption("Banco di prova isolato: spike presinaptici imposti e membrana postsinaptica passiva con τ_m e R_m "
               "del LIF. I parametri partono da quelli della barra laterale ma qui sono sempre attivi.")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        _chart(ppr_figure(test.pairs, theme))
        ppr = test.pairs["PPR (efficacia)"]
        kind = "depressione" if ppr.iloc[0] < 0.95 else "facilitazione" if ppr.iloc[0] > 1.05 else "nessun effetto"
        st.caption(f"All'intervallo più breve: PPR {ppr.iloc[0]:.2f} → {kind}.")
    with col_b:
        _chart(train_figure(test, theme))
    st.dataframe(test.pairs.round(3), hide_index=True, width="stretch")
    _download(test.pairs, "paired_pulse.csv", "proto_pp_dl")


def _impedance(exp: Experiment, theme: Theme) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    amplitude = state.number("Ampiezza (pA)", "proto_z_amp", 0.1, 200.0, 5.0 if exp.model == "lif" else 10.0, 0.5,
                             where=c1)
    f0 = state.number("f iniziale (Hz)", "proto_z_f0", 0.1, 50.0, 0.5, 0.1, where=c2)
    f1 = state.number("f finale (Hz)", "proto_z_f1", 5.0, 500.0, 150.0, 5.0, where=c3)
    duration = state.number("Durata (ms)", "proto_z_T", 500.0, 20_000.0, 4000.0, 500.0, where=c4)
    bias = state.number("Polarizzazione (pA)", "proto_z_bias", -200.0, 500.0, 0.0, 1.0, where=c5,
                        help="Corrente continua che sposta il potenziale di riposo: la risonanza dipende da V.")
    if st.button("▶ Misura l'impedenza", type="primary", key="proto_z_run"):
        dt = exp.sim.dt if exp.model == "lif" else min(exp.sim.dt, 0.025)
        with st.spinner("Simulo il chirp…"):
            _store("zap", (exp.model, impedance_profile(exp.model, exp.lif, exp.hh, amplitude, f0, f1, duration,
                                                        bias, dt)))
    stored = _stored("zap")
    if not stored:
        return
    model, profile = stored
    if profile.spikes:
        st.warning(f"Il neurone ha emesso {profile.spikes} spike: riduci l'ampiezza o la polarizzazione, "
                   "l'impedenza vale solo sottosoglia.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Frequenza di risonanza", f"{profile.resonance_hz:.1f} Hz")
    m2.metric("Fattore Q", f"{profile.q_factor:.2f}", help="|Z| al picco / |Z| alla frequenza più bassa; "
                                                           "Q ≈ 1: nessuna risonanza (filtro passa-basso).")
    m3.metric("|Z| a bassa frequenza", f"{profile.magnitude[0]:.2f} GΩ")
    theory = None
    if model == "lif":
        p = exp.lif
        theory = p.r_m / np.sqrt(1 + (2 * np.pi * profile.freqs * p.tau_m / 1000.0) ** 2)
    _chart(impedance_figure(profile, theme, theory))


def _voltage_clamp(exp: Experiment, theme: Theme) -> None:
    if exp.model != "hh":
        st.info("Il protocollo usa i parametri Hodgkin-Huxley della barra laterale (anche se il modello attivo è LIF).")
    c1, c2, c3, c4, c5 = st.columns(5)
    v_hold = state.number("V di mantenimento (mV)", "proto_vc_hold", -120.0, -40.0, -80.0, 5.0, where=c1)
    v_lo = state.number("Comandi da (mV)", "proto_vc_lo", -120.0, 80.0, -70.0, 5.0, where=c2)
    v_hi = state.number("a (mV)", "proto_vc_hi", -100.0, 100.0, 60.0, 5.0, where=c3)
    v_step = state.number("passo (mV)", "proto_vc_step", 1.0, 50.0, 10.0, 1.0, where=c4)
    step_ms = state.number("Durata (ms)", "proto_vc_len", 2.0, 100.0, 15.0, 1.0, where=c5)
    commands = np.arange(v_lo, v_hi + v_step / 2, v_step)
    if commands.size == 0 or commands.size > 60:
        st.error("Intervallo di comando non valido.")
        return
    vc = hh_voltage_clamp(exp.hh, v_hold, commands, step_ms)
    options = [float(c) for c in vc.commands]
    default = [options[int(i)] for i in np.linspace(0, len(options) - 1, min(5, len(options))).round()]
    shown = state.multi("Tracciati da mostrare (max 5)", "proto_vc_show", options, default, max_selections=5,
                        format_func=lambda v: f"{v:g} mV")
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        _chart(vclamp_figure(vc, shown, theme))
    with col_b:
        _chart(iv_figure(vc.iv, theme))
        st.caption("I_Na di picco si inverte vicino a E_Na; I_K cresce in modo sigmoide con la depolarizzazione "
                   "(attivazione di n⁴).")
    st.dataframe(vc.iv.round(1), hide_index=True, width="stretch")
    _download(vc.iv, "voltage_clamp_IV.csv", "proto_vc_dl")


def protocols_panel(exp: Experiment, theme: Theme, sig: str) -> None:
    choice = state.radio("Protocollo", "proto_choice", list(PROTOCOLS), horizontal=True)
    st.caption(PROTOCOLS[choice])
    st.divider()
    {"Curva f-I": lambda: _fi(exp, theme),
     "Esplorazione parametrica": lambda: _sweep(exp, theme, sig),
     "Trial ripetuti": lambda: _trials(exp, theme),
     "Paired-pulse (STP)": lambda: _paired_pulse(exp, theme),
     "Impedenza (ZAP)": lambda: _impedance(exp, theme),
     "Voltage clamp (HH)": lambda: _voltage_clamp(exp, theme)}[choice]()
