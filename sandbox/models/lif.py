"""Leaky Integrate-and-Fire con soglia adattiva e sinapsi a conduttanza.

    C dV/dt = −g_L (V − V_rest) − g_e (V − E_e) − g_i (V − E_i)
              − Σ_k g_k (V − E_k) + I_ext                 (conduttanze extra: stimoli, gap junction)
    con  g_L = 1 / R_m,  C = τ_m / R_m

Spike quando V ≥ V_thresh + θ; allora V → V_reset, θ → θ + Δθ, refrattarietà t_ref.
θ decade verso 0 con costante τ_θ (adattamento di soglia / frequenza).

Integrazione: Eulero esponenziale con conduttanze congelate nel passo, cioè
V(t+dt) = V∞ + (V − V∞)·exp(−dt·g_tot/C). È esatta per input costante e
incondizionatamente stabile anche con conduttanze sinaptiche molto grandi.

Le conduttanze extra arrivano già riassunte in due termini per neurone:
``g_extra`` = Σ g_k  e  ``num_extra`` = Σ g_k·E_k  (per le gap junction E_k = V del vicino).
"""

import numpy as np

from ..config import LIFParams, SynapseParams


class AdaptiveLIF:
    name = "LIF adattivo"

    def __init__(self, params: LIFParams, n_neurons: int, syn: SynapseParams, dt: float):
        self.p = params
        self.syn = syn
        self.dt = dt
        self.g_l = 1.0 / params.r_m                  # nS
        self.c_m = params.tau_m / params.r_m         # pF
        self.i_leak = self.g_l * params.v_rest       # g_L·V_rest (pA), costante
        self.dt_over_c = dt / self.c_m               # 1/nS
        self.theta_decay = np.exp(-dt / params.tau_theta) if params.tau_theta > 0 else 0.0
        self.v = np.full(n_neurons, params.v_rest)
        self.theta = np.zeros(n_neurons)             # soglia aggiuntiva adattiva (mV)
        self.refractory_left = np.zeros(n_neurons)   # ms di refrattarietà residua

    def step(self, i_ext, g_exc, g_inh, g_extra=0.0, num_extra=0.0):
        """Avanza di dt. Restituisce (maschera degli spike, V da registrare)."""
        p, syn = self.p, self.syn
        g_tot = self.g_l + g_exc + g_inh + g_extra
        v_inf = (self.i_leak + g_exc * syn.e_exc + g_inh * syn.e_inh + num_extra + i_ext) / g_tot
        v = v_inf + (self.v - v_inf) * np.exp(-self.dt_over_c * g_tot)

        refractory = self.refractory_left > 0
        v[refractory] = p.v_reset
        self.refractory_left -= self.dt
        self.theta *= self.theta_decay

        spiked = (v >= p.v_thresh + self.theta) & ~refractory
        self.v = v
        if not spiked.any():
            return spiked, v

        v_record = v.copy()
        v_record[spiked] = p.v_peak  # lo spike viene "disegnato" come picco nel tracciato
        v[spiked] = p.v_reset
        self.theta[spiked] += p.theta_jump
        self.refractory_left[spiked] = p.t_ref
        return spiked, v_record

    def spikelets(self, spiked: np.ndarray, gap: np.ndarray) -> None:
        """Il LIF non ha una forma d'onda dello spike: la carica che un potenziale d'azione
        spingerebbe attraverso le gap junction, g·(V_peak − V)·durata, arriva ai vicini
        come un salto di potenziale ("spikelet")."""
        coupling = gap[:, spiked].sum(axis=1)
        coupling[spiked] = 0.0
        self.v += coupling * (self.p.v_peak - self.v) * self.p.spike_width / self.c_m
