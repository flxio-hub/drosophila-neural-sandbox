"""Pannello "Manipolazioni": strumenti genetici, termogenetici e farmacologici."""

import streamlit as st

from ..manipulations import DEFAULT_MANIPULATIONS as D
from . import state


def manipulation_panel(neurons: list[str], sig: str) -> dict:
    p = f"{sig}_man"
    st.caption("Confronta sempre con il controllo: premi **📌 Usa come controllo** prima di manipolare.")

    st.markdown("**Genetiche** · espresse nei neuroni scelti (GAL4/UAS)")
    spec = {
        "ablated": state.multi("🔪 Ablazione — il neurone viene rimosso dal circuito", f"{p}_ablated", neurons),
        "tnt": state.multi("🚫 TNT — rilascio sinaptico abolito", f"{p}_tnt", neurons),
        "kir": state.multi("🔇 Kir2.1 — silenziamento con una conduttanza K⁺", f"{p}_kir", neurons),
    }
    if spec["kir"]:
        c1, c2 = st.columns(2)
        spec["kir_g"] = state.number("g Kir2.1 (nS)", f"{p}_kir_g", 0.0, 50.0, D["kir_g"], 0.5, where=c1)
        spec["e_k"] = state.number("E_K (mV)", f"{p}_ek", -110.0, -50.0, D["e_k"], 1.0, where=c2)

    st.markdown("**Termogenetiche** · attivate dal protocollo di temperatura")
    spec["trpa1"] = state.multi("🔥 dTrpA1 — canale cationico attivato dal caldo", f"{p}_trpa1", neurons)
    spec["shibire"] = state.multi("🧊 shibire^ts — rilascio bloccato a temperatura restrittiva", f"{p}_shi", neurons)
    if spec["trpa1"] or spec["shibire"]:
        c1, c2, c3 = st.columns(3)
        spec["temp_baseline"] = state.number("T permissiva (°C)", f"{p}_tb", 15.0, 30.0, D["temp_baseline"], 0.5,
                                             where=c1)
        spec["temp_target"] = state.number("T restrittiva (°C)", f"{p}_tt", 20.0, 40.0, D["temp_target"], 0.5,
                                           where=c2)
        spec["temp_ramp"] = state.number("Rampa (ms)", f"{p}_tr", 1.0, 2000.0, D["temp_ramp"], 5.0, where=c3)
        c1, c2 = st.columns(2)
        spec["temp_on"] = state.number("Caldo da (ms)", f"{p}_ton", 0.0, None, D["temp_on"], 10.0, where=c1)
        spec["temp_off"] = state.number("Caldo fino a (ms)", f"{p}_toff", 0.0, None, D["temp_off"], 10.0, where=c2)
        c1, c2, c3 = st.columns(3)
        spec["trpa1_gmax"] = state.number("ḡ dTrpA1 (nS)", f"{p}_tg", 0.0, 50.0, D["trpa1_gmax"], 0.5, where=c1)
        spec["trpa1_t_half"] = state.number("T½ dTrpA1 (°C)", f"{p}_th", 20.0, 35.0, D["trpa1_t_half"], 0.5, where=c2)
        spec["shibire_t_half"] = state.number("T½ shibire (°C)", f"{p}_sh", 20.0, 35.0, D["shibire_t_half"], 0.5,
                                              where=c3)
        st.caption("In vivo servono secondi o minuti; qui l'effetto segue subito la temperatura. "
                   "La temperatura compare nel grafico ①.")

    st.markdown("**Farmacologiche** · applicate a tutto il circuito")
    spec["ptx"] = state.slider("Picrotossina: blocco delle sinapsi inibitorie (%)", f"{p}_ptx", 0, 100, 0, 5) / 100
    spec["mec"] = state.slider("Mecamilamina: blocco dei nAChR eccitatori (%)", f"{p}_mec", 0, 100, 0, 5) / 100
    spec["gap_block"] = state.slider("Blocco delle gap junction (come nei mutanti shakB) (%)", f"{p}_gap",
                                     0, 100, 0, 5) / 100
    active = [name for name in ("ablated", "tnt", "kir", "trpa1", "shibire") if spec.get(name)]
    active += [name for name in ("ptx", "mec", "gap_block") if spec[name] > 0]
    if active:
        st.success(f"{len(active)} manipolazioni attive.")
    return spec
