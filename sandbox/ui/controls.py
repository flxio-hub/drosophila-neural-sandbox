"""Controlli della barra laterale: circuito, modello neuronale, sinapsi, simulazione, salvataggio."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st

from .. import experiment as experiment_io
from ..config import HHParams, LIFParams, SimParams, STPParams, SynapseParams
from ..network import PRESETS, Circuit, edges_to_circuit, fetch_flywire_edges, fetch_hemibrain_edges, load_edge_csv
from ..network.topologies import feedforward_cascade, mushroom_body, random_ei_network
from . import state

CONNECTOME = "📂 Connettoma reale (CSV · FlyWire · Hemibrain)"
IMPORTED_CIRCUIT = "📥 Circuito importato (JSON)"
EXAMPLE_CSV = Path(__file__).resolve().parents[2] / "data" / "example_mushroom_body.csv"


def signature(*parts) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:10]


@st.cache_data(show_spinner=False)
def _load_csv_cached(data: bytes, syn_to_nS: float, min_synapses: int, max_neurons: int):
    return load_edge_csv(io.BytesIO(data), syn_to_nS=syn_to_nS, min_synapses=min_synapses,
                         max_neurons=max_neurons)


def _parse_ids(text: str) -> list[str]:
    return [tok for tok in text.replace(",", " ").split() if tok.strip()]


# ============================================================================ circuito

def circuit_controls() -> tuple[Circuit | None, str, list[str]]:
    """Selettore di topologia. Restituisce (circuito, firma per gli editor delle matrici, avvisi)."""
    st.header("🧬 Circuito")
    options = [*PRESETS, CONNECTOME]
    imported = st.session_state.get("_imported_circuit")
    if imported:
        options.append(IMPORTED_CIRCUIT)
    preset = state.select("Topologia", "preset", options)

    if preset == "Cascata feed-forward":
        n = state.slider("Numero di stadi", "cascade_n", 2, 12, 5)
        w = state.slider("Peso sinaptico (nS)", "cascade_w", 0.0, 10.0, 3.0, 0.1)
        return feedforward_cascade(n, w), signature(preset, n, w), []

    if preset == "Rete ricorrente E/I casuale":
        n = state.slider("Neuroni", "rand_n", 5, 60, 20)
        p = state.slider("Probabilità di connessione", "rand_p", 0.0, 1.0, 0.2, 0.01)
        frac = state.slider("Frazione eccitatoria", "rand_frac", 0.0, 1.0, 0.8, 0.05)
        c1, c2 = st.columns(2)
        w_e = state.number("w_E (nS)", "rand_we", 0.0, 20.0, 1.0, 0.1, where=c1)
        w_i = state.number("|w_I| (nS)", "rand_wi", 0.0, 50.0, 3.0, 0.1, where=c2)
        seed = state.number("Seme della rete", "rand_seed", 0, 10_000, 0)
        return random_ei_network(n, p, frac, w_e, w_i, seed), signature(preset, n, p, frac, w_e, w_i, seed), []

    if preset == "Corpo fungo: codifica sparsa":
        n_kc = state.slider("Cellule di Kenyon", "mb_nkc", 10, 60, 30)
        seed = state.number("Seme della connettività PN → KC", "mb_seed", 0, 10_000, 1)
        return mushroom_body(n_kc=n_kc, seed=seed), signature(preset, n_kc, seed), []

    if preset == IMPORTED_CIRCUIT:
        data = imported
        circuit = Circuit(name=data.get("name", "Circuito importato"), neurons=list(data["neurons"]),
                          weights=np.array(data["weights"], dtype=float),
                          gap=np.array(data["gap"], dtype=float) if data.get("gap") is not None else None,
                          description=data.get("description", ""))
        return circuit, signature("imported", json.dumps(data, sort_keys=True)[:2000]), []

    if preset != CONNECTOME:
        return PRESETS[preset](), signature(preset), []

    return _connectome_controls()


def _connectome_controls() -> tuple[Circuit | None, str, list[str]]:
    source = state.radio("Sorgente", "cx_source", ["Esempio incluso (corpo fungo, sintetico)", "File CSV",
                                                  "FlyWire live (CAVEclient)", "Hemibrain live (neuPrint)"])
    syn_to_nS = state.slider("Conduttanza per sinapsi (nS)", "cx_syn_to_nS", 0.01, 1.0, 0.15, 0.01,
                             help="Il numero di sinapsi tra due neuroni viene moltiplicato per questo valore.")
    c1, c2 = st.columns(2)
    min_syn = state.number("Soglia sinapsi", "cx_min_syn", 1, 100, 1, where=c1,
                           help="Scarta connessioni più deboli (FlyWire usa spesso ≥ 5).")
    max_neurons = state.number("Max neuroni", "cx_max_neurons", 2, 150, 60, where=c2)
    opts = dict(syn_to_nS=syn_to_nS, min_synapses=int(min_syn), max_neurons=int(max_neurons))

    try:
        if source.startswith("Esempio"):
            circuit, warnings = _load_csv_cached(EXAMPLE_CSV.read_bytes(), **opts)
            return circuit, signature("example", *opts.values()), warnings

        if source == "File CSV":
            st.caption("Colonne riconosciute: `pre_root_id`/`bodyId_pre`/`pre`, `post_root_id`/`bodyId_post`/`post`, "
                       "`syn_count`/`weight`, opzionali `nt_type`, `pre_name`, `post_name` "
                       "(es. `connections.csv` scaricato da FlyWire Codex).")
            upload = st.file_uploader("Lista di archi (.csv)", type=["csv"], key="cx_upload")
            if upload is None:
                st.info("Carica un file CSV per costruire il circuito.")
                return None, "", []
            data = upload.getvalue()
            circuit, warnings = _load_csv_cached(data, **opts)
            return circuit, signature(upload.name, len(data), *opts.values()), warnings

        is_flywire = source.startswith("FlyWire")
        ids_text = state.text("Root ID FlyWire" if is_flywire else "Body ID Hemibrain", "cx_ids", "", area=True,
                              placeholder="720575940… 720575940…" if is_flywire else "5813105172 1071121755")
        token = "" if is_flywire else st.text_input("Token neuPrint", type="password", key="cx_token",
                                                    help="Dal tuo profilo su neuprint.janelia.org (non viene salvato)")
        if st.button("⬇️ Scarica connettività", width="stretch"):
            ids = _parse_ids(ids_text)
            if len(ids) < 2:
                st.error("Servono almeno 2 ID.")
            else:
                with st.spinner("Interrogo il server…"):
                    if is_flywire:
                        st.session_state["live_edges"] = (source, fetch_flywire_edges(ids), None, ids)
                    else:
                        edges, labels = fetch_hemibrain_edges(ids, token=token)
                        st.session_state["live_edges"] = (source, edges, labels, ids)
        stored = st.session_state.get("live_edges")
        if not stored or stored[0] != source:
            st.info("Inserisci gli ID e scarica la connettività.")
            return None, "", []
        _, edges, labels, ids = stored
        circuit, warnings = edges_to_circuit(edges, neuron_ids=ids, labels=labels, name=source, **opts)
        return circuit, signature(source, tuple(ids), *opts.values()), warnings

    except Exception as exc:  # errori di rete, token, formato del file…
        st.error(f"Impossibile costruire il circuito: {exc}")
        return None, "", []


# ============================================================================ modello e sinapsi

def model_controls() -> tuple[str, LIFParams, HHParams]:
    st.header("🧠 Modello neuronale")
    model_label = state.radio("Modello", "model", ["LIF adattivo", "Hodgkin-Huxley"], horizontal=True)
    lif, hh = LIFParams(), HHParams()
    if model_label == "LIF adattivo":
        lif = LIFParams(
            v_rest=state.slider("V_rest (mV)", "lif_v_rest", -80.0, -40.0, lif.v_rest, 0.5),
            v_thresh=state.slider("V_thresh (mV)", "lif_v_thresh", -60.0, -20.0, lif.v_thresh, 0.5),
            v_reset=state.slider("V_reset (mV)", "lif_v_reset", -85.0, -40.0, lif.v_reset, 0.5),
            tau_m=state.slider("τ_m (ms)", "lif_tau_m", 1.0, 100.0, lif.tau_m, 0.5),
            r_m=state.slider("R_m (GΩ)", "lif_r_m", 0.1, 5.0, lif.r_m, 0.05),
            t_ref=state.slider("Refrattarietà (ms)", "lif_t_ref", 0.0, 10.0, lif.t_ref, 0.1),
            theta_jump=state.slider("Adattamento soglia Δθ (mV)", "lif_theta_jump", 0.0, 10.0, lif.theta_jump, 0.1,
                                    help="Aumento della soglia a ogni spike; 0 = LIF classico."),
            tau_theta=state.slider("τ_θ (ms)", "lif_tau_theta", 10.0, 1000.0, lif.tau_theta, 5.0),
        )
        rheobase = (lif.v_thresh - lif.v_rest) / lif.r_m
        st.caption(f"C_m = {lif.tau_m / lif.r_m:.1f} pF · reobase ≈ {rheobase:.1f} pA")
        if lif.v_thresh <= lif.v_rest:
            st.warning("V_thresh ≤ V_rest: i neuroni scaricheranno anche senza input.")
    else:
        with st.expander("Parametri Hodgkin-Huxley", expanded=True):
            hh = HHParams(
                area_um2=state.slider("Area di membrana (µm²)", "hh_area", 100.0, 5000.0, hh.area_um2, 50.0),
                g_na=state.slider("ḡ_Na (mS/cm²)", "hh_g_na", 0.0, 200.0, hh.g_na, 1.0),
                g_k=state.slider("ḡ_K (mS/cm²)", "hh_g_k", 0.0, 80.0, hh.g_k, 0.5),
                g_l=state.slider("g_L (mS/cm²)", "hh_g_l", 0.0, 1.0, hh.g_l, 0.01),
                e_na=state.slider("E_Na (mV)", "hh_e_na", 20.0, 80.0, hh.e_na, 0.5),
                e_k=state.slider("E_K (mV)", "hh_e_k", -100.0, -60.0, hh.e_k, 0.5),
                e_l=state.slider("E_L (mV)", "hh_e_l", -80.0, -40.0, hh.e_l, 0.1),
            )
        st.caption(f"C_m = {hh.c_m * hh.area_um2 * 1e-2:.1f} pF · reobase ≈ {6.5 * hh.area_um2 * 1e-2:.0f} pA "
                   "(cinetiche classiche di Hodgkin-Huxley)")
    return ("lif" if model_label == "LIF adattivo" else "hh"), lif, hh


def synapse_controls(neurons: list[str], sig: str) -> tuple[SynapseParams, STPParams, list[str] | None]:
    st.header("🔗 Sinapsi")
    d = SynapseParams()
    c1, c2 = st.columns(2)
    syn = SynapseParams(
        tau_exc=state.slider("τ_exc (ms)", "syn_tau_exc", 0.5, 50.0, d.tau_exc, 0.5, where=c1),
        tau_inh=state.slider("τ_inh (ms)", "syn_tau_inh", 0.5, 100.0, d.tau_inh, 0.5, where=c2),
        e_exc=state.slider("E_exc (mV)", "syn_e_exc", -20.0, 20.0, d.e_exc, 1.0, where=c1),
        e_inh=state.slider("E_inh (mV)", "syn_e_inh", -100.0, -50.0, d.e_inh, 1.0, where=c2),
        delay=state.slider("Ritardo sinaptico (ms)", "syn_delay", 0.0, 20.0, d.delay, 0.1),
    )
    with st.expander("Plasticità a breve termine (STP)"):
        s = STPParams()
        enabled = state.toggle("Attiva depressione/facilitazione", "stp_enabled", False)
        stp = STPParams(
            enabled=enabled,
            U=state.slider("U (frazione di rilascio)", "stp_U", 0.01, 1.0, s.U, 0.01, disabled=not enabled,
                           help="U alto → forte depressione; U basso con τ_fac > 0 → facilitazione."),
            tau_rec=state.slider("τ_rec recupero (ms)", "stp_tau_rec", 10.0, 2000.0, s.tau_rec, 10.0,
                                 disabled=not enabled),
            tau_fac=state.slider("τ_fac facilitazione (ms)", "stp_tau_fac", 0.0, 2000.0, s.tau_fac, 10.0,
                                 disabled=not enabled, help="0 = solo depressione."),
        )
        targets = state.multi("Neuroni presinaptici con STP (vuoto = tutti)", f"{sig}_stp_targets", neurons,
                              disabled=not enabled)
        st.caption("Esempio reale: ORN → PN è fortemente depressiva (Kazama & Wilson 2008). "
                   "Il protocollo *Paired-pulse* mostra l'effetto dei parametri.")
    return syn, stp, (targets or None)


def simulation_controls(model: str) -> SimParams:
    st.header("⏱️ Simulazione")
    if model == "lif":
        dt = state.select_slider("dt (ms)", "dt_lif", [0.01, 0.025, 0.05, 0.1, 0.2, 0.5], 0.1)
    else:
        dt = state.select_slider("dt (ms)", "dt_hh", [0.005, 0.01, 0.025, 0.05], 0.025,
                                 help="Hodgkin-Huxley richiede passi piccoli per risolvere il potenziale d'azione.")
    duration = state.slider("Durata (ms)", f"duration_{model}", 50, 3000, 500, 10)
    seed = state.number("Seme stocastico", "seed", 0, 100_000, 0, help="Rumore e pennacchi d'odore riproducibili.")
    with st.expander("Rumore di membrana di fondo"):
        c1, c2 = st.columns(2)
        sigma = state.number("σ (pA)", "noise_sigma", 0.0, 100.0, 0.0, 0.5, where=c1,
                             help="Rumore indipendente su ogni neurone: rende i trial variabili.")
        tau = state.number("τ (ms)", "noise_tau", 0.5, 100.0, 5.0, 0.5, where=c2)
    return SimParams(duration=float(duration), dt=float(dt), seed=int(seed), noise_sigma=float(sigma),
                     noise_tau=float(tau))


# ============================================================================ salvataggio

def _on_import() -> None:
    upload = st.session_state.get("import_file")
    if upload is None:
        return
    try:
        data = json.loads(upload.getvalue())
        exp = experiment_io.from_dict(data)
        circuit = data.get("circuit") or {}
        st.session_state["_imported_circuit"] = {
            "name": circuit.get("name", exp.name), "description": circuit.get("description", ""),
            "neurons": exp.neurons, "weights": exp.weights.tolist(), "gap": exp.gap.tolist(),
        }
        widgets = dict(data.get("widgets", {}))
        widgets["preset"] = IMPORTED_CIRCUIT
        state.load(widgets)
        st.session_state["_import_message"] = ("success", f"Esperimento «{exp.name}» caricato "
                                                          f"({len(exp.neurons)} neuroni, {len(exp.stimuli)} stimoli).")
    except Exception as exc:
        st.session_state["_import_message"] = ("error", f"File non valido: {exc}")


def _on_reset() -> None:
    state.load({"preset": state.registry().get("preset", next(iter(PRESETS)))})


def experiment_box(exp, circuit: Circuit, summary) -> None:
    """Esporta/importa l'esperimento completo in JSON e ripristina i valori di default."""
    with st.expander("💾 Salva / carica esperimento"):
        message = st.session_state.pop("_import_message", None)
        if message:
            getattr(st, message[0])(message[1])

        widgets = state.registry()

        def export():
            return experiment_io.to_json(
                exp, widgets=widgets,
                circuit={"name": circuit.name, "description": circuit.description},
                # passando da to_json i NaN diventano null (JSON valido anche fuori da Python)
                summary=json.loads(summary.to_json(orient="records")) if summary is not None else None,
            )

        st.download_button("⬇️ Esporta esperimento (JSON)", export, "esperimento_drosophila.json",
                           "application/json", on_click="ignore", width="stretch",
                           help="Circuito, parametri, stimoli, manipolazioni e riepilogo dei risultati.")
        st.file_uploader("Carica un esperimento (.json)", type=["json"], key="import_file", on_change=_on_import)
        st.button("↺ Ripristina i valori di default", on_click=_on_reset, width="stretch")
        st.caption("Il file JSON è leggibile anche da Python: `sandbox.experiment.from_json(...)`.")
