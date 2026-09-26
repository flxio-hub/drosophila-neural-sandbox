"""Editor delle matrici di connettività: sinapsi chimiche W[pre, post] e gap junction (simmetriche)."""

import numpy as np
import pandas as pd
import streamlit as st

from ..network import Circuit
from . import state


def _editor(matrix: np.ndarray, neurons: list[str], name: str, step: float) -> np.ndarray:
    df = pd.DataFrame(matrix, index=neurons, columns=neurons)
    edited = st.data_editor(
        df, key=state.key(name), width="stretch",
        column_config={n: st.column_config.NumberColumn(n, format="%.2f", step=step) for n in neurons},
    )
    return edited.fillna(0.0).to_numpy(dtype=float)


def weight_editor(circuit: Circuit, sig: str) -> tuple[np.ndarray, float]:
    st.caption("Righe = neurone **presinaptico**, colonne = **postsinaptico**. Valori in nS: positivi eccitatori "
               "(ACh), negativi inibitori (GABA/Glu). |W| è la conduttanza di picco evocata da un singolo spike.")
    weights = _editor(circuit.weights, circuit.neurons, f"W_{sig}", 0.1)
    scale = state.slider("Moltiplicatore globale dei pesi chimici", "w_scale", 0.0, 3.0, 1.0, 0.05)
    n_exc, n_inh = int((weights > 0).sum()), int((weights < 0).sum())
    st.caption(f"{n_exc} sinapsi eccitatorie · {n_inh} inibitorie · "
               f"densità {100 * (n_exc + n_inh) / max(len(circuit.neurons) ** 2, 1):.0f}%")
    return weights, scale


def gap_editor(circuit: Circuit, sig: str) -> tuple[np.ndarray, float]:
    st.caption("Sinapsi elettriche (innexine, es. ShakB) in nS. Sono simmetriche: per ogni coppia vale il valore "
               "più grande fra (i, j) e (j, i). Con il LIF lo spike attraversa la giunzione come "
               "\"spikelet\" (carica = g · (V_picco − V) · durata dello spike).")
    raw = _editor(circuit.gap, circuit.neurons, f"G_{sig}", 0.5)
    gap = np.maximum(np.clip(raw, 0.0, None), np.clip(raw, 0.0, None).T)
    np.fill_diagonal(gap, 0.0)
    if (raw < 0).any():
        st.warning("Le conduttanze negative sono state portate a 0.")
    scale = state.slider("Moltiplicatore globale delle gap junction", "gap_scale", 0.0, 3.0, 1.0, 0.05)
    st.caption(f"{int((np.triu(gap) > 0).sum())} coppie accoppiate elettricamente.")
    return gap, scale
