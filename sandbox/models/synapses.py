"""Sinapsi chimiche a conduttanza con decadimento esponenziale, ritardo e plasticità a breve termine.

Convenzione della matrice: ``W[pre, post]`` in nS (righe = presinaptico,
colonne = postsinaptico, come nelle tabelle di connettività FlyWire).
    W > 0 → sinapsi eccitatoria (inverte a E_exc)
    W < 0 → sinapsi inibitoria  (inverte a E_inh)

Ogni spike presinaptico j, dopo il ritardo, aggiunge efficacia_j · |W[j, i]|
alla conduttanza del neurone postsinaptico i, che poi decade con τ_exc o τ_inh:
    dg_exc/dt = −g_exc/τ_exc + Σ_j W⁺[j, i] · Σ_spike e_j · δ(t − t_j − ritardo)

L'efficacia e_j vale 1, salvo:
* plasticità a breve termine (Tsodyks-Markram): e = u·x/U;
* guadagno di rilascio < 1 (blocco sinaptico: TNT, shibire^ts).
Così |W| resta la conduttanza di picco evocata da un singolo spike "a riposo".

Poiché tutte le sinapsi dello stesso tipo decadono con la stessa costante di
tempo, basta seguire la conduttanza totale di ogni neurone postsinaptico: il
prodotto vettore-matrice serve solo nei passi in cui arrivano spike.
"""

import numpy as np

from ..config import STPParams, SynapseParams


def delay_steps(delay: float, dt: float) -> int:
    """Lunghezza della linea di ritardo. La conduttanza aggiornata agisce dal passo
    successivo, che conta già come un dt: il buffer copre il resto del ritardo."""
    return max(int(round(delay / dt)) - 1, 0)


class ShortTermPlasticity:
    """Risorse x e utilizzo u per ogni neurone presinaptico (modello di Tsodyks-Markram)."""

    def __init__(self, params: STPParams, mask: np.ndarray, dt: float):
        n = mask.size
        self.U = params.U
        self.mask = mask.astype(bool)
        self.x = np.ones(n)
        self.u = np.full(n, params.U)
        self.rec_decay = np.exp(-dt / params.tau_rec) if params.tau_rec > 0 else 0.0
        self.fac_decay = np.exp(-dt / params.tau_fac) if params.tau_fac > 0 else 0.0

    def efficacy(self, spiked: np.ndarray) -> np.ndarray:
        """Recupero di un passo, poi rilascio per i neuroni che hanno appena scaricato."""
        self.x = 1.0 - (1.0 - self.x) * self.rec_decay
        self.u = self.U + (self.u - self.U) * self.fac_decay
        efficacy = spiked.astype(float)
        plastic = spiked & self.mask
        if plastic.any():
            release = self.u[plastic] * self.x[plastic]
            efficacy[plastic] = release / self.U
            self.x[plastic] -= release
            self.u[plastic] += self.U * (1.0 - self.u[plastic])
        return efficacy


class ExponentialSynapses:
    def __init__(self, weights: np.ndarray, params: SynapseParams, dt: float,
                 stp: STPParams | None = None, stp_mask: np.ndarray | None = None):
        weights = np.asarray(weights, dtype=float)
        n = weights.shape[0]
        self.w_exc = np.clip(weights, 0.0, None)
        self.w_inh = np.clip(-weights, 0.0, None)
        self.g_exc = np.zeros(n)  # conduttanza eccitatoria su ogni neurone postsinaptico (nS)
        self.g_inh = np.zeros(n)
        self.decay_exc = np.exp(-dt / params.tau_exc)
        self.decay_inh = np.exp(-dt / params.tau_inh)
        self._buffer = np.zeros((delay_steps(params.delay, dt) + 1, n))  # efficacie in transito
        self._ptr = 0
        mask = np.ones(n, dtype=bool) if stp_mask is None else np.asarray(stp_mask, dtype=bool)
        self.stp = ShortTermPlasticity(stp, mask, dt) if stp is not None and stp.enabled and mask.any() else None

    def update(self, spiked: np.ndarray, release_gain: np.ndarray | None = None) -> None:
        """Registra gli spike appena emessi, fa decadere le conduttanze e consegna gli spike in arrivo."""
        efficacy = self.stp.efficacy(spiked) if self.stp is not None else spiked.astype(float)
        if release_gain is not None:
            efficacy = efficacy * release_gain
        size = len(self._buffer)
        self._buffer[self._ptr] = efficacy
        self._ptr = (self._ptr + 1) % size
        arriving = self._buffer[self._ptr]  # lo slot più vecchio: spike emessi `delay_steps` passi fa (o ora, se 0)
        self.g_exc *= self.decay_exc
        self.g_inh *= self.decay_inh
        if arriving.any():
            self.g_exc += arriving @ self.w_exc
            self.g_inh += arriving @ self.w_inh
