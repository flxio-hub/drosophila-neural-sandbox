"""Modello di Hodgkin-Huxley (Na⁺, K⁺, leak) con sinapsi a conduttanza.

Cinetiche classiche dell'assone gigante di calamaro (convenzione V_rest ≈ -65 mV):
sono un modello generico di neurone a potenziali d'azione, non una
ricostruzione di canali specifici di Drosophila (per quelli vedi p.es. i
modelli dei motoneuroni di Günay et al. 2015).

Integrazione:
* gate m, h, n → schema di Rush–Larsen (esponenziale esatto a V fissato),
  con x∞(V) ed exp(−dt/τ_x(V)) precalcolati in tabelle da 0.01 mV
  (la stessa strategia delle TABLE di NEURON): costo per passo indipendente
  dalla complessità delle funzioni α/β;
* V → Eulero esponenziale con tutte le conduttanze congelate nel passo.
Entrambi sono stabili; per spike ben risolti usa dt ≤ 0.025 ms.
"""

import numpy as np

from ..config import HHParams, SynapseParams

V_MIN, V_MAX, V_RES = -120.0, 80.0, 0.01  # intervallo e risoluzione delle tabelle (mV)


def _vtrap(x, y):
    """x / (exp(x/y) − 1) con il limite corretto (= y) per x → 0."""
    ratio = x / y
    small = np.abs(ratio) < 1e-6
    return np.where(small, y * (1.0 - ratio / 2.0), x / np.expm1(np.where(small, 1.0, ratio)))


def rates(v):
    """Costanti di velocità α, β (1/ms) per i gate m, h, n."""
    alpha_m = 0.1 * _vtrap(-(v + 40.0), 10.0)
    beta_m = 4.0 * np.exp(-(v + 65.0) / 18.0)
    alpha_h = 0.07 * np.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))
    alpha_n = 0.01 * _vtrap(-(v + 55.0), 10.0)
    beta_n = 0.125 * np.exp(-(v + 65.0) / 80.0)
    return (alpha_m, beta_m), (alpha_h, beta_h), (alpha_n, beta_n)


def gate_tables(dt: float):
    """Tabelle (3, n_V) di x∞ e del fattore di decadimento exp(−dt/τ) per m, h, n."""
    v = np.arange(V_MIN, V_MAX + V_RES / 2, V_RES)
    x_inf, decay = [], []
    for a, b in rates(v):
        x_inf.append(a / (a + b))
        decay.append(np.exp(-dt * (a + b)))
    return np.array(x_inf), np.array(decay)


class HodgkinHuxley:
    name = "Hodgkin-Huxley"

    def __init__(self, params: HHParams, n_neurons: int, syn: SynapseParams, dt: float):
        self.p = params
        scale = params.area_um2 * 1e-2  # mS/cm² → nS  e  µF/cm² → pF
        self.g_na = params.g_na * scale
        self.g_k = params.g_k * scale
        self.g_l = params.g_l * scale
        self.i_leak = self.g_l * params.e_l
        self.dt_over_c = dt / (params.c_m * scale)
        self.e_exc, self.e_inh = syn.e_exc, syn.e_inh
        self.c_m = params.c_m * scale
        self.x_inf, self.x_decay = gate_tables(dt)
        self.v = np.full(n_neurons, params.v_init)
        # gate [m, h, n] inizializzati allo stato stazionario del potenziale iniziale
        self.gates = self.x_inf[:, self._table_index(self.v)]

    @staticmethod
    def _table_index(v):
        # np.minimum/np.maximum invece di np.clip: su array piccoli clip è ~5× più lento
        return ((np.minimum(np.maximum(v, V_MIN), V_MAX) - V_MIN) * (1.0 / V_RES) + 0.5).astype(np.intp)

    def step(self, i_ext, g_exc, g_inh, g_extra=0.0, num_extra=0.0):
        """Avanza di dt. ``g_extra``/``num_extra``: Σg e Σg·E delle conduttanze extra (stimoli, gap junction)."""
        p = self.p
        idx = self._table_index(self.v)
        x_inf = self.x_inf[:, idx]
        self.gates = x_inf + (self.gates - x_inf) * self.x_decay[:, idx]
        m, h, n = self.gates

        g_na = self.g_na * m ** 3 * h
        g_k = self.g_k * n ** 4
        g_tot = g_na + g_k + self.g_l + g_exc + g_inh + g_extra
        v_inf = (g_na * p.e_na + g_k * p.e_k + self.i_leak
                 + g_exc * self.e_exc + g_inh * self.e_inh + num_extra + i_ext) / g_tot
        v_new = v_inf + (self.v - v_inf) * np.exp(-self.dt_over_c * g_tot)

        spiked = (self.v < p.spike_detect) & (v_new >= p.spike_detect)
        self.v = v_new
        return spiked, v_new
