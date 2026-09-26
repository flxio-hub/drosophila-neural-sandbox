"""Stimoli sensoriali simulati: profilo fisico (luce, odore, looming) → corrente di trasduzione.

Pipeline (fenomenologica, ma con le proprietà qualitative note in Drosophila):

    segnale s(t) ≥ 0
      → compressione di Hill       f = sⁿ / (sⁿ + Kⁿ)          (curva dose-risposta)
      → filtro veloce              a_fast = LP(f, τ_fast)       (cinetica di trasduzione)
      → adattamento sottrattivo    I = g · (a_fast − k · LP(f, τ_slow))

Con k > 0 la risposta è fasico-tonica: picco all'onset, plateau adattato e
sottoelongazione (iperpolarizzazione) all'offset, come negli ORN (Nagel &
Wilson 2011) e nei fotorecettori R1–R6, che in Drosophila si DEPOLARIZZANO
alla luce (fototrasduzione via PLC/TRP).
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import lfilter


@dataclass
class TransductionParams:
    half_sat: float     # K: intensità/concentrazione di semi-saturazione (unità normalizzate)
    hill_n: float       # coefficiente di Hill
    tau_fast: float     # ms
    tau_slow: float     # ms
    adaptation: float   # k: frazione della risposta rimossa dall'adattamento (0–1)
    gain_pA: float      # corrente massima iniettata nel neurone bersaglio (pA)


# Preset per modalità sensoriale: parametri di trasduzione + profilo di stimolo suggerito.
MODALITIES = {
    "Fotorecettore (R1–R6)": {
        "params": TransductionParams(half_sat=0.3, hill_n=1.0, tau_fast=4.0, tau_slow=150.0, adaptation=0.6, gain_pA=60.0),
        "signal_label": "Intensità luminosa (norm.)",
        "default_profile": "Impulso",
    },
    "Neurone olfattivo (ORN)": {
        "params": TransductionParams(half_sat=0.2, hill_n=1.5, tau_fast=25.0, tau_slow=400.0, adaptation=0.5, gain_pA=80.0),
        "signal_label": "Concentrazione odore (norm.)",
        "default_profile": "Impulso",
    },
    "Rilevatore di looming (LC4 / LPLC2)": {
        "params": TransductionParams(half_sat=0.25, hill_n=3.0, tau_fast=10.0, tau_slow=300.0, adaptation=0.3, gain_pA=60.0),
        "signal_label": "Dimensione angolare θ/180°",
        "default_profile": "Looming",
    },
}

PROFILES = ("Impulso", "Gradiente (rampa)", "Sfarfallio sinusoidale", "Pennacchio turbolento", "Looming")


# ----------------------------------------------------------------------------- profili fisici

def profile_pulse(t, amplitude, t_on, t_off, background=0.0):
    return background + amplitude * ((t >= t_on) & (t < t_off))


def profile_gradient(t, level_start, level_end, t_on, t_off):
    """Gradiente temporale lineare (es. la mosca che risale un gradiente d'odore)."""
    frac = np.clip((t - t_on) / max(t_off - t_on, 1e-9), 0.0, 1.0)
    level = level_start + (level_end - level_start) * frac
    return np.where(t < t_on, 0.0, level)


def profile_flicker(t, mean, contrast, frequency_hz, t_on=0.0, t_off=np.inf):
    """Luce sfarfallante: mean · (1 + contrast · sin(2πft))."""
    window = (t >= t_on) & (t < t_off)
    return window * mean * (1.0 + contrast * np.sin(2 * np.pi * frequency_hz * (t - t_on) / 1000.0))


def profile_plume(t, rate_hz, puff_ms, amplitude, t_on=0.0, t_off=np.inf, seed=None):
    """Pennacchio d'odore intermittente: sbuffi gaussiani con arrivi di Poisson e ampiezza variabile."""
    rng = np.random.default_rng(seed)
    t_end = min(t_off, t[-1]) if t.size else 0.0
    n_puffs = rng.poisson(rate_hz * max(t_end - t_on, 0.0) / 1000.0)
    centers = rng.uniform(t_on, t_end, n_puffs)
    heights = amplitude * rng.uniform(0.3, 1.0, n_puffs)
    signal = np.zeros_like(t)
    for c, h in zip(centers, heights):
        signal += h * np.exp(-0.5 * ((t - c) / puff_ms) ** 2)
    return signal


def profile_looming(t, t_collision, l_over_v):
    """Disco in avvicinamento a velocità costante (paradigma di von Reyn et al. 2014).

    Dimensione angolare θ(t) = 2·atan(l/v / (t_c − t)), normalizzata a 180°;
    dopo la "collisione" il disco resta a piena dimensione.
    """
    remaining = t_collision - t
    theta = np.where(remaining > 0, 2.0 * np.arctan(l_over_v / np.maximum(remaining, 1e-9)), np.pi)
    return theta / np.pi


# ----------------------------------------------------------------------------- trasduzione

def _lowpass(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """Filtro passa-basso del primo ordine (IIR esatto), inizializzato a regime sul primo campione."""
    if tau <= 0:
        return x.copy()
    a = np.exp(-dt / tau)
    y, _ = lfilter([1.0 - a], [1.0, -a], x, zi=[a * x[0]])
    return y


def transduce(signal: np.ndarray, dt: float, p: TransductionParams) -> np.ndarray:
    """Converte il segnale sensoriale in corrente (pA) iniettata nel neurone bersaglio."""
    if signal.size == 0:
        return signal.copy()
    s = np.clip(signal, 0.0, None)
    s_n = s ** p.hill_n
    activation = s_n / (s_n + max(p.half_sat, 1e-9) ** p.hill_n)
    fast = _lowpass(activation, dt, p.tau_fast)
    slow = _lowpass(activation, dt, p.tau_slow)
    return p.gain_pA * (fast - p.adaptation * slow)


def default_params(modality: str) -> TransductionParams:
    return replace(MODALITIES[modality]["params"])
