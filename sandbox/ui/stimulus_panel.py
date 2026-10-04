"""Pannello "Stimoli": ogni canale produce una specifica (dict) che ``stimuli.build_channel`` trasforma in segnali.

Le chiavi dei widget includono il tipo di stimolo: cambiando tipo non si ereditano
valori pensati per un altro stimolo (es. un'ampiezza in pA dentro un'intensità luminosa).
"""

import streamlit as st

from ..data import store
from ..stimuli import (CONDUCTANCE_KINDS, HELP_TEXT, KINDS, MODALITIES, OPSINS, PROFILES, RECEPTORS, ExpressionError,
                       evaluate_expression, time_vector, with_defaults)
from ..stimuli.waveforms import CHIRP_METHODS, PERIODIC_SHAPES
from . import state
from .look import explain
from .texts import KIND_HELP, TIPS

UPLOAD = "📤 Carica un file…"

KIND_CODES = {
    "Onda quadra": "sq", "Rampa": "rp", "Scala di gradini": "st", "Onda periodica": "pe", "Chirp (ZAP)": "ch",
    "Rumore gaussiano": "no", "Barra in movimento": "mb", "Equazione custom": "eq", "Da file CSV": "csv",
    "Sensoriale": "se", "Input sinaptico di Poisson": "po", "Optogenetica": "op",
}
MODALITY_CODES = {name: code for name, code in zip(MODALITIES, ("ph", "orn", "loom"))}
OPSIN_CODES = {name: code for name, code in zip(OPSINS, ("csc", "gta"))}


def _kind_label(kind: str) -> str:
    return f"{kind} · conduttanza" if kind in CONDUCTANCE_KINDS else kind


def _times(p, d, duration, on="t_on", off="t_off"):
    c1, c2 = st.columns(2)
    return {
        on: state.number("Inizio (ms)", f"{p}_ton", 0.0, None, float(d[on]), 10.0, where=c1),
        off: state.number("Fine (ms)", f"{p}_toff", 0.0, None, float(min(d[off], duration)), 10.0, where=c2),
    }


def _square(p, d, duration, neurons):
    amplitude = state.slider("Ampiezza (pA)", f"{p}_amp", -100.0, 300.0, float(d["amplitude"]), 1.0)
    c1, c2 = st.columns(2)
    frequency = state.slider("Frequenza (Hz, 0 = gradino)", f"{p}_f", 0.0, 200.0, float(d["frequency"]), 1.0, where=c1)
    width = state.slider("Durata impulso (ms)", f"{p}_w", 0.5, 100.0, float(d["width"]), 0.5, where=c2,
                         disabled=frequency == 0)
    return {"amplitude": amplitude, "frequency": frequency, "width": width, **_times(p, d, duration)}


def _ramp(p, d, duration, neurons):
    c1, c2 = st.columns(2)
    params = {"start": state.slider("Ampiezza iniziale (pA)", f"{p}_a0", -100.0, 300.0, float(d["start"]), 1.0, where=c1),
              "end": state.slider("Ampiezza finale (pA)", f"{p}_a1", -100.0, 300.0, float(d["end"]), 1.0, where=c2)}
    params.update(_times(p, d, duration))
    params["hold"] = state.check("Mantieni il valore finale dopo la rampa", f"{p}_hold", bool(d["hold"]))
    return params


def _staircase(p, d, duration, neurons):
    c1, c2 = st.columns(2)
    start = state.number("Primo gradino (pA)", f"{p}_start", -200.0, 500.0, float(d["start"]), 1.0, where=c1)
    step = state.number("Incremento (pA)", f"{p}_step", -200.0, 500.0, float(d["step"]), 1.0, where=c2)
    c1, c2, c3 = st.columns(3)
    n_steps = state.number("Gradini", f"{p}_n", 1, 40, int(d["n_steps"]), where=c1)
    step_ms = state.number("Durata (ms)", f"{p}_len", 1.0, 5000.0, float(d["step_ms"]), 5.0, where=c2)
    gap_ms = state.number("Pausa (ms)", f"{p}_gap", 0.0, 5000.0, float(d["gap_ms"]), 5.0, where=c3)
    t_on = state.number("Inizio (ms)", f"{p}_ton", 0.0, None, float(d["t_on"]), 10.0)
    end = t_on + n_steps * (step_ms + gap_ms)
    st.caption(f"Livelli: {', '.join(f'{start + k * step:g}' for k in range(min(n_steps, 12)))}"
               f"{'…' if n_steps > 12 else ''} pA · fine a {end:g} ms"
               + (" — oltre la durata della simulazione!" if end > duration else ""))
    return {"start": start, "step": step, "n_steps": n_steps, "step_ms": step_ms, "gap_ms": gap_ms, "t_on": t_on}


def _periodic(p, d, duration, neurons):
    shape = state.select("Forma", f"{p}_shape", PERIODIC_SHAPES, d["shape"])
    c1, c2 = st.columns(2)
    params = {
        "shape": shape,
        "amplitude": state.slider("Ampiezza (pA)", f"{p}_amp", 0.0, 200.0, float(d["amplitude"]), 1.0, where=c1),
        "dc": state.slider("Offset DC (pA)", f"{p}_dc", -100.0, 200.0, float(d["dc"]), 1.0, where=c2),
        "frequency": state.slider("Frequenza (Hz)", f"{p}_f", 0.5, 200.0, float(d["frequency"]), 0.5, where=c1),
        "phase": state.slider("Fase (°)", f"{p}_ph", 0.0, 360.0, float(d["phase"]), 5.0, where=c2),
    }
    return {**params, **_times(p, d, duration)}


def _chirp(p, d, duration, neurons):
    c1, c2 = st.columns(2)
    params = {
        "amplitude": state.slider("Ampiezza (pA)", f"{p}_amp", 0.0, 100.0, float(d["amplitude"]), 0.5, where=c1),
        "dc": state.slider("Offset DC (pA)", f"{p}_dc", -50.0, 100.0, float(d["dc"]), 0.5, where=c2),
        "f0": state.number("Frequenza iniziale (Hz)", f"{p}_f0", 0.1, 500.0, float(d["f0"]), 0.5, where=c1),
        "f1": state.number("Frequenza finale (Hz)", f"{p}_f1", 0.1, 500.0, float(d["f1"]), 1.0, where=c2),
        "method": state.radio("Andamento della frequenza", f"{p}_method", list(CHIRP_METHODS), d["method"],
                              horizontal=True),
    }
    params.update(_times(p, d, duration))
    st.caption("Per ricavare l'impedenza |Z(f)| usa il protocollo *Impedenza di membrana (ZAP)*.")
    return params


def _noise(p, d, duration, neurons):
    c1, c2 = st.columns(2)
    params = {
        "mean": state.slider("Media (pA)", f"{p}_mu", -50.0, 200.0, float(d["mean"]), 1.0, where=c1),
        "sigma": state.slider("Deviazione standard σ (pA)", f"{p}_sd", 0.0, 100.0, float(d["sigma"]), 0.5, where=c2),
        "tau_corr": state.slider("Tempo di correlazione τ (ms, 0 = bianco)", f"{p}_tau", 0.0, 50.0,
                                 float(d["tau_corr"]), 0.5),
    }
    params.update(_times(p, d, duration))
    params["independent"] = state.check("Realizzazione indipendente per ogni neurone", f"{p}_indep",
                                        bool(d["independent"]))
    return params


def _moving_bar(p, d, duration, neurons):
    direction = state.radio("Direzione", f"{p}_dir", ["→", "←"], d["direction"], horizontal=True)
    c1, c2 = st.columns(2)
    params = {
        "direction": direction,
        "amplitude": state.slider("Ampiezza (pA)", f"{p}_amp", 0.0, 300.0, float(d["amplitude"]), 1.0, where=c1),
        "pulse_ms": state.number("Durata su ogni posizione (ms)", f"{p}_pulse", 1.0, 1000.0, float(d["pulse_ms"]),
                                 1.0, where=c2),
        "interval_ms": state.number("Intervallo fra posizioni (ms)", f"{p}_int", 0.5, 1000.0, float(d["interval_ms"]),
                                    0.5, where=c1),
        "t_on": state.number("Inizio (ms)", f"{p}_ton", 0.0, None, float(d["t_on"]), 10.0, where=c2),
        "repeats": state.number("Passaggi", f"{p}_rep", 1, 50, int(d["repeats"]), where=c1),
        "repeat_ms": state.number("Periodo fra passaggi (ms)", f"{p}_repms", 1.0, 5000.0, float(d["repeat_ms"]),
                                  10.0, where=c2),
    }
    return params


def _expression(p, d, duration, neurons):
    expression = state.text("I(t) [pA] =", f"{p}_expr", d["expression"], area=True, height=80)
    st.caption(HELP_TEXT)
    try:
        evaluate_expression(expression, time_vector(min(duration, 1000.0), 0.5), seed=0)
    except ExpressionError as exc:
        st.error(f"Equazione non valida — {exc}")
    return {"expression": expression}


def _csv(p, d, duration, neurons):
    """Forma d'onda da file: caricata qui o scelta dalla libreria (pagina Dati). I campioni finiscono comunque
    nella specifica, così l'esperimento esportato in JSON resta autosufficiente."""
    memo = state.key(f"_samples:{p}")  # sopravvive al cambio di pagina (non all'importazione di un esperimento)
    samples = st.session_state.get(memo) or state.default(f"{p}_samples", None)
    choice = state.select("Forma d'onda", f"{p}_lib", [UPLOAD, *store.list_stimuli()])
    if choice == UPLOAD:
        upload = st.file_uploader("File .csv (tempo in ms, valore in pA)", type=["csv", "txt"],
                                  key=state.key(f"{p}_file"), help=TIPS["upload_waveform"])
        if upload is not None:
            try:
                t, v = store.parse_waveform(upload.getvalue())
                samples = {"t": t, "v": v, "source": upload.name}
                if st.button("💾 Salva nella libreria", key=state.key(f"{p}_save"), help=TIPS["save_to_library"]):
                    st.toast(f"Salvato nella libreria come «{store.save_stimulus(upload.name, upload.getvalue())}».")
            except Exception as exc:
                st.error(f"CSV non leggibile: {exc}")
    else:
        try:
            samples = store.load_stimulus(choice)
        except Exception as exc:
            st.error(f"Stimolo «{choice}» non leggibile: {exc}")
    st.session_state[memo] = samples
    if samples:
        state.record(f"{p}_samples", samples)
        st.caption(f"{samples.get('source', 'dati importati')}: {len(samples['t'])} campioni tra "
                   f"{min(samples['t']):g} e {max(samples['t']):g} ms.")
    c1, c2 = st.columns(2)
    return {"samples_t": samples["t"] if samples else [], "samples_v": samples["v"] if samples else [],
            "source": samples.get("source", "") if samples else "",
            "offset_ms": state.number("Ritardo (ms)", f"{p}_off", -10_000.0, 10_000.0, float(d["offset_ms"]), 10.0,
                                      where=c1),
            "scale": state.number("Scala (×)", f"{p}_scale", -100.0, 100.0, float(d["scale"]), 0.1, where=c2)}


def _sensory(p, d, duration, neurons):
    modality = state.select("Modalità sensoriale", f"{p}_mod", list(MODALITIES), d["modality"])
    info = MODALITIES[modality]
    q = f"{p}_{MODALITY_CODES[modality]}"  # ogni modalità ha i propri widget (e i propri default)
    default_profile = d["profile"] if d["modality"] == modality else info["default_profile"]
    profile = state.select("Profilo dello stimolo", f"{q}_prof", PROFILES, default_profile)
    params = {"modality": modality, "profile": profile}
    if profile == "Impulso":
        c1, c2 = st.columns(2)
        params["amplitude"] = state.slider("Intensità", f"{q}_amp", 0.0, 1.0, float(d["amplitude"]), 0.01, where=c1)
        params["background"] = state.slider("Fondo", f"{q}_bg", 0.0, 1.0, float(d["background"]), 0.01, where=c2)
        params.update(_times(q, d, duration))
    elif profile == "Gradiente (rampa)":
        c1, c2 = st.columns(2)
        params["level_start"] = state.slider("Livello iniziale", f"{q}_s0", 0.0, 1.0, float(d["level_start"]), 0.01,
                                             where=c1)
        params["level_end"] = state.slider("Livello finale", f"{q}_s1", 0.0, 1.0, float(d["level_end"]), 0.01, where=c2)
        params.update(_times(q, d, duration))
    elif profile == "Sfarfallio sinusoidale":
        c1, c2, c3 = st.columns(3)
        params["mean"] = state.slider("Media", f"{q}_m", 0.0, 1.0, float(d["mean"]), 0.01, where=c1)
        params["contrast"] = state.slider("Contrasto", f"{q}_c", 0.0, 1.0, float(d["contrast"]), 0.01, where=c2)
        params["frequency"] = state.slider("Hz", f"{q}_f", 0.5, 50.0, float(d["frequency"]), 0.5, where=c3)
        params.update(_times(q, d, duration))
    elif profile == "Pennacchio turbolento":
        c1, c2, c3 = st.columns(3)
        params["rate"] = state.slider("Sbuffi/s", f"{q}_r", 0.5, 30.0, float(d["rate"]), 0.5, where=c1)
        params["puff_ms"] = state.slider("Durata (ms)", f"{q}_w", 2.0, 100.0, float(d["puff_ms"]), 1.0, where=c2)
        params["peak"] = state.slider("Picco", f"{q}_pk", 0.0, 1.0, float(d["peak"]), 0.01, where=c3)
        params.update(_times(q, d, duration))
    else:  # Looming
        c1, c2 = st.columns(2)
        params["t_collision"] = state.number("Istante di collisione (ms)", f"{q}_tc", 0.0, None,
                                             float(d["t_collision"]), 10.0, where=c1)
        params["l_over_v"] = state.slider("l/v (ms)", f"{q}_lv", 5.0, 120.0, float(d["l_over_v"]), 1.0, where=c2)
    base = info["params"] if d["modality"] != modality else None
    t = {name: (getattr(base, name) if base is not None else d[name])
         for name in ("half_sat", "hill_n", "gain_pA", "tau_fast", "tau_slow", "adaptation")}
    with st.expander("Parametri di trasduzione"):
        c1, c2, c3 = st.columns(3)
        params.update(
            half_sat=state.number("K (semi-sat.)", f"{q}_K", 0.01, 5.0, float(t["half_sat"]), 0.01, where=c1),
            hill_n=state.number("n (Hill)", f"{q}_n", 0.5, 6.0, float(t["hill_n"]), 0.1, where=c2),
            gain_pA=state.number("Guadagno (pA)", f"{q}_g", 0.0, 500.0, float(t["gain_pA"]), 5.0, where=c3),
            tau_fast=state.number("τ veloce (ms)", f"{q}_tf", 0.5, 200.0, float(t["tau_fast"]), 0.5, where=c1),
            tau_slow=state.number("τ adattamento (ms)", f"{q}_ts", 10.0, 5000.0, float(t["tau_slow"]), 10.0, where=c2),
            adaptation=state.number("Adattamento k", f"{q}_k", 0.0, 1.0, float(t["adaptation"]), 0.05, where=c3),
        )
    return params


def _poisson(p, d, duration, neurons):
    receptor = state.radio("Tipo di sinapsi", f"{p}_rec", RECEPTORS, d["receptor"])
    c1, c2, c3 = st.columns(3)
    params = {
        "receptor": receptor,
        "rate": state.number("Frequenza totale (Hz)", f"{p}_rate", 1.0, 5000.0, float(d["rate"]), 10.0, where=c1),
        "g_peak": state.number("g per evento (nS)", f"{p}_g", 0.01, 20.0, float(d["g_peak"]), 0.05, where=c2),
        "tau": state.number("τ (ms)", f"{p}_tau", 0.5, 100.0, float(d["tau"]), 0.5, where=c3),
    }
    params.update(_times(p, d, duration))
    mean_g = params["rate"] / 1000.0 * params["g_peak"] * params["tau"]
    st.caption(f"Conduttanza media ≈ {mean_g:.2f} nS per bersaglio (treni indipendenti).")
    return params


def _optogenetics(p, d, duration, neurons):
    opsin = state.select("Opsina", f"{p}_opsin", list(OPSINS), d["opsin"])
    info = OPSINS[opsin]
    q = f"{p}_{OPSIN_CODES[opsin]}"
    c1, c2 = st.columns(2)
    params = {
        "opsin": opsin,
        "intensity": state.slider("Intensità luminosa (norm.)", f"{q}_int", 0.0, 1.0, float(d["intensity"]), 0.01,
                                  where=c1),
        "g_max": state.number("ḡ opsina (nS)", f"{q}_g", 0.0, 50.0, float(d["g_max"] or info.g_max), 0.5, where=c2),
        "frequency": state.slider("Impulsi (Hz, 0 = luce continua)", f"{q}_f", 0.0, 100.0, float(d["frequency"]),
                                  1.0, where=c1),
        "pulse_ms": state.number("Durata impulso (ms)", f"{q}_pw", 0.5, 500.0, float(d["pulse_ms"]), 0.5, where=c2),
    }
    params.update(_times(q, d, duration))
    st.caption(f"Luce {info.color} · E_rev {info.e_rev:g} mV · τ_on {info.tau_on:g} ms, τ_off {info.tau_off:g} ms "
               "(valori indicativi).")
    return params


PARAM_UI = {
    "Onda quadra": _square, "Rampa": _ramp, "Scala di gradini": _staircase, "Onda periodica": _periodic,
    "Chirp (ZAP)": _chirp, "Rumore gaussiano": _noise, "Barra in movimento": _moving_bar,
    "Equazione custom": _expression, "Da file CSV": _csv, "Sensoriale": _sensory,
    "Input sinaptico di Poisson": _poisson, "Optogenetica": _optogenetics,
}


def channel_ui(k: int, prefix: str, neurons: list[str], default: dict, duration: float) -> dict:
    """Controlli di un canale; restituisce la sua specifica."""
    base = with_defaults(default)
    c1, c2 = st.columns([0.45, 0.55])
    kind = state.select("Tipo", f"{prefix}_kind", KINDS, base["kind"], where=c1, format_func=_kind_label)
    default_targets = [n for n in base.get("targets", neurons[:1]) if n in neurons] or neurons[:1]
    targets = state.multi("Neuroni bersaglio", f"{prefix}_targets", neurons, default_targets, where=c2)
    st.caption(KIND_HELP[kind])
    d = base if kind == base["kind"] else with_defaults({"kind": kind})
    params = PARAM_UI[kind](f"{prefix}_{KIND_CODES[kind]}", d, duration, neurons)
    return {"kind": kind, "targets": targets, **params}


def stimulus_panel(neurons: list[str], sig: str, defaults: list[dict], duration: float,
                   model: str) -> tuple[list[dict], float]:
    """Tutti i canali di stimolo. Restituisce (specifiche, guadagno globale delle correnti)."""
    explain("stimuli")
    c1, c2 = st.columns(2)
    n_channels = state.number("Canali di stimolo", f"{sig}_nch", 0, 8, len(defaults), where=c1)
    gain = state.number("Guadagno correnti (×)", f"gain_{model}", 0.0, 20.0, 1.0 if model == "lif" else 3.0, 0.1,
                        where=c2)
    specs = []
    for k in range(int(n_channels)):
        default = defaults[k] if k < len(defaults) else {"kind": "Onda quadra", "targets": neurons[:1]}
        with st.expander(f"Canale S{k + 1}", expanded=k < 2):
            specs.append(channel_ui(k, f"{sig}_ch{k}", neurons, default, duration))
    if not specs:
        st.caption("Nessuno stimolo: il circuito riceve solo rumore di fondo e manipolazioni (es. dTrpA1).")
    return specs, gain
