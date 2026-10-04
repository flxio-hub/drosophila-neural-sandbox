"""Motore di simulazione: integra la rete passo per passo.

Ordine delle operazioni a ogni passo k:
    1. termini extra: conduttanze esterne (stimoli a conduttanza) e gap junction,
       calcolate con i potenziali all'inizio del passo
    2. aggiornamento dei neuroni con I_ext[k] e le conduttanze sinaptiche correnti → spike
       (LIF: gli spike raggiungono i vicini elettrici come "spikelet")
    3. rilascio (plasticità a breve termine × guadagno di rilascio), linea di ritardo,
       decadimento e consegna delle conduttanze (effetto dal passo k+1)

Due backend con gli stessi schemi numerici:
    "numpy" — classi di ``sandbox.models``, implementazione di riferimento
    "numba" — kernel JIT di ``fast.py`` (usato automaticamente se Numba è installato)

``Stepper`` fa avanzare la rete a blocchi conservando lo stato, per l'anello chiuso
(stimoli che dipendono da ciò che la rete ha appena fatto); ``simulate`` è uno
Stepper con un solo blocco.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..config import HHParams, LIFParams, STPParams, SynapseParams
from ..models import AdaptiveLIF, ExponentialSynapses, HodgkinHuxley
from ..models.hodgkin_huxley import V_MAX, V_MIN, V_RES, gate_tables
from ..models.synapses import delay_steps
from . import fast

MODELS = {"LIF adattivo": "lif", "Hodgkin-Huxley": "hh"}
DEFAULT_BACKEND = "numba" if fast.AVAILABLE else "numpy"


@dataclass
class SimulationResult:
    t: np.ndarray                  # (steps,) ms
    v: np.ndarray                  # (steps, n) mV
    i_ext: np.ndarray              # (steps, n) pA
    spike_times: list[np.ndarray]  # tempi di spike per neurone (ms)
    neurons: list[str]
    dt: float
    backend: str = "numpy"
    g_exc: np.ndarray | None = None  # (steps, n) conduttanza sinaptica eccitatoria (nS, float32)
    g_inh: np.ndarray | None = None  # (steps, n) conduttanza sinaptica inibitoria (nS, float32)
    g_ext: np.ndarray | None = None  # conduttanza degli stimoli esterni (nS), se presente
    ge_ext: np.ndarray | None = None  # Σ g·E_rev degli stimoli esterni (pA), se presente
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.t.size * self.dt

    @property
    def spike_counts(self) -> np.ndarray:
        return np.array([s.size for s in self.spike_times])


def _as_input(array, steps, n, fill):
    """Array (passi, n) contiguo, oppure un segnaposto (1, n) costante se l'input è assente."""
    if array is None:
        return np.full((1, n), fill)
    array = np.ascontiguousarray(array, dtype=float)
    if array.shape != (steps, n):
        raise ValueError(f"Input con forma {array.shape}, attesa {(steps, n)}.")
    return array


def simulate(weights: np.ndarray, i_ext: np.ndarray, neurons: list[str], dt: float, *,
             model: str = "lif", lif: LIFParams | None = None, hh: HHParams | None = None,
             syn: SynapseParams | None = None, gap: np.ndarray | None = None,
             g_ext: np.ndarray | None = None, ge_ext: np.ndarray | None = None,
             release_gain: np.ndarray | None = None, stp: STPParams | None = None,
             stp_mask: np.ndarray | None = None, backend: str = "auto") -> SimulationResult:
    """Simula la rete.

    weights      : W[pre, post] in nS (segno = tipo di sinapsi chimica)
    i_ext        : corrente esterna (passi, n) in pA
    gap          : conduttanze delle gap junction (n, n) in nS; gap[i, j] porta corrente da j a i
    g_ext, ge_ext: conduttanze esterne (passi, n): Σg in nS e Σg·E_rev in pA (optogenetica, Poisson…)
    release_gain : moltiplicatore del rilascio presinaptico (passi, n); 0 = sinapsi bloccate
    stp, stp_mask: plasticità a breve termine e neuroni presinaptici a cui si applica
    backend      : "auto", "numpy" o "numba"
    """
    stepper = Stepper(weights, neurons, dt, model=model, lif=lif, hh=hh, syn=syn, gap=gap, stp=stp,
                      stp_mask=stp_mask, backend=backend)
    return stepper.advance(i_ext, g_ext=g_ext, ge_ext=ge_ext, release_gain=release_gain)


class Stepper:
    """Rete che avanza a blocchi conservando lo stato fra una chiamata di ``advance`` e l'altra.

    Serve all'anello chiuso (una mosca virtuale in un mondo 3D): lo stimolo del blocco
    successivo dipende da ciò che la rete ha fatto nel precedente, quindi non si conosce in
    anticipo. Far avanzare la rete a blocchi dà esattamente lo stesso risultato di un'unica
    chiamata di ``simulate`` (tests/test_stepper.py). Parametri come in ``simulate``.
    """

    def __init__(self, weights: np.ndarray, neurons: list[str], dt: float, *, model: str = "lif",
                 lif: LIFParams | None = None, hh: HHParams | None = None, syn: SynapseParams | None = None,
                 gap: np.ndarray | None = None, stp: STPParams | None = None,
                 stp_mask: np.ndarray | None = None, backend: str = "auto"):
        self.syn, self.lif, self.hh = syn or SynapseParams(), lif or LIFParams(), hh or HHParams()
        self.stp = stp or STPParams()
        weights = np.ascontiguousarray(weights, dtype=float)
        n = weights.shape[0] if weights.ndim == 2 else -1
        if weights.shape != (n, n):
            raise ValueError("La matrice W deve essere n × n.")
        if model not in ("lif", "hh"):
            raise ValueError(f"Modello sconosciuto: {model}")
        if self.stp.enabled and not 0.0 < self.stp.U <= 1.0:
            raise ValueError("U della plasticità a breve termine deve essere in (0, 1].")
        gap = np.zeros((n, n)) if gap is None else np.ascontiguousarray(gap, dtype=float)
        if gap.shape != (n, n):
            raise ValueError("La matrice delle gap junction deve essere n × n.")
        if (gap < 0).any():
            raise ValueError("Le conduttanze delle gap junction devono essere ≥ 0.")
        backend = DEFAULT_BACKEND if backend == "auto" else backend
        if backend == "numba" and not fast.AVAILABLE:
            raise RuntimeError("Backend Numba richiesto ma Numba non è installato.")

        self.neurons, self.n, self.dt, self.model, self.backend = list(neurons), n, dt, model, backend
        self.steps_done = 0
        self._gap, self._gap_on = gap, bool(gap.any())
        self._stp_mask = np.ones(n, dtype=bool) if stp_mask is None else np.asarray(stp_mask, dtype=bool)
        self._stp_on = bool(self.stp.enabled and self._stp_mask.any())
        if backend == "numba":
            self._init_numba(weights)
        else:
            self._init_numpy(weights)

    @property
    def time(self) -> float:
        """Tempo simulato finora (ms)."""
        return self.steps_done * self.dt

    def advance(self, i_ext: np.ndarray, g_ext: np.ndarray | None = None, ge_ext: np.ndarray | None = None,
                release_gain: np.ndarray | None = None) -> SimulationResult:
        """Avanza di ``len(i_ext)`` passi; tutti gli input hanno forma (passi del blocco, n).
        Restituisce il risultato del solo blocco, con i tempi contati dall'inizio della simulazione."""
        i_ext = np.ascontiguousarray(i_ext, dtype=float)
        if i_ext.ndim != 2 or i_ext.shape[1] != self.n:
            raise ValueError("La matrice W e la corrente esterna hanno un numero di neuroni diverso.")
        if (g_ext is None) != (ge_ext is None):
            raise ValueError("g_ext e ge_ext vanno forniti insieme.")
        steps, n = i_ext.shape
        inputs = (_as_input(g_ext, steps, n, 0.0), _as_input(ge_ext, steps, n, 0.0),
                  _as_input(release_gain, steps, n, 1.0))
        run = self._advance_numba if self.backend == "numba" else self._advance_numpy
        v_trace, raster, g_e_trace, g_i_trace = run(i_ext, *inputs)

        t = (self.steps_done + np.arange(steps)) * self.dt
        self.steps_done += steps
        spike_times = [t[raster[:, i]] for i in range(n)]
        return SimulationResult(t=t, v=v_trace, i_ext=i_ext, spike_times=spike_times, neurons=list(self.neurons),
                                dt=self.dt, backend=self.backend, g_exc=g_e_trace, g_inh=g_i_trace,
                                g_ext=g_ext, ge_ext=ge_ext,
                                meta={"model": self.model, "e_exc": self.syn.e_exc, "e_inh": self.syn.e_inh})

    def set_weights(self, weights: np.ndarray) -> None:
        """Cambia i pesi sinaptici fra un blocco e l'altro (plasticità a lungo termine calcolata fuori dalla rete,
        per esempio l'apprendimento del corpo a fungo); lo stato della rete resta quello che era. Gli spike già in
        viaggio nella linea di ritardo arrivano con i pesi nuovi."""
        weights = np.ascontiguousarray(weights, dtype=float)
        if weights.shape != (self.n, self.n):
            raise ValueError("La matrice W deve essere n × n.")
        w_exc, w_inh = np.ascontiguousarray(np.clip(weights, 0.0, None)), np.ascontiguousarray(np.clip(-weights, 0.0, None))
        if self.backend == "numba":
            self._w_exc, self._w_inh = w_exc, w_inh
        else:
            self._synapses.w_exc, self._synapses.w_inh = w_exc, w_inh

    # ------------------------------------------------------------------------- backend NumPy

    def _init_numpy(self, weights):
        n, dt, syn = self.n, self.dt, self.syn
        self._cell = AdaptiveLIF(self.lif, n, syn, dt) if self.model == "lif" else HodgkinHuxley(self.hh, n, syn, dt)
        self._synapses = ExponentialSynapses(weights, syn, dt, self.stp if self._stp_on else None, self._stp_mask)
        self._gap_rowsum = self._gap.sum(axis=1)

    def _advance_numpy(self, i_ext, g_ext, ge_ext, release_gain):
        cell, synapses, gap, gap_on = self._cell, self._synapses, self._gap, self._gap_on
        steps, n = i_ext.shape
        ext_varies, gain_varies = g_ext.shape[0] == steps, release_gain.shape[0] == steps

        v_trace = np.empty((steps, n))
        g_e_trace = np.empty((steps, n), dtype=np.float32)
        g_i_trace = np.empty((steps, n), dtype=np.float32)
        raster = np.zeros((steps, n), dtype=bool)
        for k in range(steps):
            k_ext = k if ext_varies else 0
            g_extra, num_extra = g_ext[k_ext], ge_ext[k_ext]
            if gap_on:
                g_extra = g_extra + self._gap_rowsum
                num_extra = num_extra + gap @ cell.v
            g_e_trace[k], g_i_trace[k] = synapses.g_exc, synapses.g_inh
            spiked, v_trace[k] = cell.step(i_ext[k], synapses.g_exc, synapses.g_inh, g_extra, num_extra)
            if gap_on and self.model == "lif" and spiked.any():
                cell.spikelets(spiked, gap)
            raster[k] = spiked
            synapses.update(spiked, release_gain[k if gain_varies else 0])
        return v_trace, raster, g_e_trace, g_i_trace

    # ------------------------------------------------------------------------- backend Numba

    def _init_numba(self, weights):
        n, dt, stp = self.n, self.dt, self.stp
        self._w_exc = np.ascontiguousarray(np.clip(weights, 0.0, None))
        self._w_inh = np.ascontiguousarray(np.clip(-weights, 0.0, None))
        self._rec_decay = np.exp(-dt / stp.tau_rec) if stp.tau_rec > 0 else 0.0
        self._fac_decay = np.exp(-dt / stp.tau_fac) if stp.tau_fac > 0 else 0.0
        if self.model == "lif":
            cell = (np.full(n, self.lif.v_rest), np.zeros(n), np.zeros(n))  # V, soglia adattiva θ, refrattarietà
        else:
            self._x_inf, self._x_decay = _cached_gate_tables(dt)
            v = np.full(n, self.hh.v_init)
            gates = np.ascontiguousarray(self._x_inf[:, HodgkinHuxley._table_index(v)])  # m, h, n a regime
            cell = (v, gates)
        # g_exc, g_inh, risorse x e utilizzo u della STP, linea di ritardo e suo puntatore
        synapses = (np.zeros(n), np.zeros(n), np.ones(n), np.full(n, float(stp.U)),
                    np.zeros((delay_steps(self.syn.delay, dt) + 1, n)), np.zeros(1, dtype=np.int64))
        self._state = cell + synapses

    def _advance_numba(self, i_ext, g_ext, ge_ext, release_gain):
        dt, syn = self.dt, self.syn
        head = (self._w_exc, self._w_inh, self._gap, self._gap_on, i_ext, g_ext, ge_ext, release_gain,
                self._stp_on, self._stp_mask, float(self.stp.U), self._rec_decay, self._fac_decay, dt)
        tail = (np.exp(-dt / syn.tau_exc), np.exp(-dt / syn.tau_inh), syn.e_exc, syn.e_inh, *self._state)
        if self.model == "lif":
            lif = self.lif
            return fast.lif_kernel(*head, lif.v_rest, lif.v_thresh, lif.v_reset, lif.tau_m, lif.r_m, lif.t_ref,
                                   lif.theta_jump, lif.tau_theta, lif.v_peak, lif.spike_width, *tail)
        hh = self.hh
        scale = hh.area_um2 * 1e-2  # mS/cm² → nS  e  µF/cm² → pF
        return fast.hh_kernel(*head, self._x_inf, self._x_decay, V_MIN, V_MAX, 1.0 / V_RES,
                              hh.g_na * scale, hh.g_k * scale, hh.g_l * scale, hh.e_na, hh.e_k, hh.e_l,
                              hh.c_m * scale, hh.spike_detect, *tail)


@lru_cache(maxsize=8)
def _cached_gate_tables(dt: float):
    return gate_tables(dt)
