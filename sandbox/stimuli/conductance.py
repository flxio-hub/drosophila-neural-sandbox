"""Stimoli a conduttanza: g(t) con un potenziale di inversione, invece di una corrente imposta.

A differenza di una corrente, una conduttanza cambia anche la resistenza della
membrana (effetto "shunting") e il suo effetto dipende da V: un canale che
inverte vicino al riposo può inibire senza iperpolarizzare.

* Input sinaptico di Poisson — bombardamento di fondo da molti neuroni non modellati.
* Optogenetica — CsChrimson (attivazione, luce rossa) e GtACR1 (inibizione, luce verde),
  i due strumenti più usati per manipolare neuroni di Drosophila con la luce.

I parametri cinetici delle opsine sono indicativi (ordini di grandezza da
Klapoetke et al. 2014 e Govorunova et al. 2015), non misure nei neuroni di mosca.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


def poisson_conductance(t, rate_hz, g_peak, tau_ms, n_trains=1, t_on=0.0, t_off=np.inf, seed=None):
    """Treni di Poisson indipendenti filtrati con un kernel esponenziale: g(t) = Σ_k g_peak·exp(−(t − t_k)/τ).

    Restituisce una matrice (passi, n_trains) in nS.
    """
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0] if t.size > 1 else 1.0
    window = ((t >= t_on) & (t < t_off))[:, None]
    events = rng.poisson(rate_hz * dt / 1000.0, size=(t.size, int(n_trains))) * window
    a = np.exp(-dt / tau_ms) if tau_ms > 0 else 0.0
    return lfilter([g_peak], [1.0, -a], events.astype(float), axis=0)


@dataclass(frozen=True)
class OpsinParams:
    e_rev: float      # potenziale di inversione (mV)
    tau_on: float     # costante di attivazione (ms)
    tau_off: float    # costante di chiusura dopo lo spegnimento della luce (ms)
    half_sat: float   # intensità di semi-attivazione (unità normalizzate)
    g_max: float      # conduttanza massima di default (nS)
    color: str        # lunghezza d'onda di eccitazione


OPSINS = {
    "CsChrimson (attivazione)": OpsinParams(e_rev=0.0, tau_on=3.0, tau_off=20.0, half_sat=0.25, g_max=5.0,
                                            color="rossa, 617 nm"),
    "GtACR1 (inibizione)": OpsinParams(e_rev=-65.0, tau_on=3.0, tau_off=50.0, half_sat=0.25, g_max=8.0,
                                       color="verde, 530 nm"),
}


def light_pulses(t, intensity, t_on, t_off, frequency_hz=0.0, pulse_ms=10.0):
    """Protocollo luminoso normalizzato (0–1): luce continua o impulsi a frequenza data."""
    window = (t >= t_on) & (t < t_off)
    if frequency_hz > 0:
        period = 1000.0 / frequency_hz
        window &= ((t - t_on) % period) < min(pulse_ms, period)
    return intensity * window


def opsin_conductance(light, dt, opsin: OpsinParams, g_max=None):
    """Conduttanza dell'opsina: attivazione a regime a∞ = L/(L + K), cinetica del primo ordine
    con τ_on quando il canale si apre e τ_off quando si chiude."""
    g_max = opsin.g_max if g_max is None else g_max
    a_inf = light / (light + opsin.half_sat)
    k_on, k_off = 1.0 - np.exp(-dt / opsin.tau_on), 1.0 - np.exp(-dt / opsin.tau_off)
    a = np.empty_like(a_inf)
    state = 0.0
    for k, target in enumerate(a_inf):  # τ dipende dalla direzione: serve un ciclo esplicito
        state += (target - state) * (k_on if target > state else k_off)
        a[k] = state
    return g_max * a
