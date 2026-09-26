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
    syn, lif, hh = syn or SynapseParams(), lif or LIFParams(), hh or HHParams()
    stp = stp or STPParams()
    weights = np.ascontiguousarray(weights, dtype=float)
    i_ext = np.ascontiguousarray(i_ext, dtype=float)
    steps, n = i_ext.shape
    if weights.shape != (n, n):
        raise ValueError("La matrice W e la corrente esterna hanno un numero di neuroni diverso.")
    if model not in ("lif", "hh"):
        raise ValueError(f"Modello sconosciuto: {model}")
    if (g_ext is None) != (ge_ext is None):
        raise ValueError("g_ext e ge_ext vanno forniti insieme.")
    if stp.enabled and not 0.0 < stp.U <= 1.0:
        raise ValueError("U della plasticità a breve termine deve essere in (0, 1].")

    gap = np.zeros((n, n)) if gap is None else np.ascontiguousarray(gap, dtype=float)
    if gap.shape != (n, n):
        raise ValueError("La matrice delle gap junction deve essere n × n.")
    if (gap < 0).any():
        raise ValueError("Le conduttanze delle gap junction devono essere ≥ 0.")
    mask = np.ones(n, dtype=bool) if stp_mask is None else np.asarray(stp_mask, dtype=bool)
    inputs = dict(
        gap=gap, gap_on=bool(gap.any()),
        g_ext=_as_input(g_ext, steps, n, 0.0), ge_ext=_as_input(ge_ext, steps, n, 0.0),
        release_gain=_as_input(release_gain, steps, n, 1.0),
        stp=stp, stp_mask=mask, stp_on=bool(stp.enabled and mask.any()),
    )

    backend = DEFAULT_BACKEND if backend == "auto" else backend
    if backend == "numba" and not fast.AVAILABLE:
        raise RuntimeError("Backend Numba richiesto ma Numba non è installato.")
    run = _run_numba if backend == "numba" else _run_numpy
    v_trace, raster, g_e_trace, g_i_trace = run(weights, i_ext, dt, model, lif, hh, syn, **inputs)

    t = np.arange(steps) * dt
    spike_times = [t[raster[:, i]] for i in range(n)]
    return SimulationResult(t=t, v=v_trace, i_ext=i_ext, spike_times=spike_times, neurons=list(neurons),
                            dt=dt, backend=backend, g_exc=g_e_trace, g_inh=g_i_trace,
                            g_ext=g_ext, ge_ext=ge_ext,
                            meta={"model": model, "e_exc": syn.e_exc, "e_inh": syn.e_inh})


def _run_numpy(weights, i_ext, dt, model, lif, hh, syn, *, gap, gap_on, g_ext, ge_ext, release_gain,
               stp, stp_mask, stp_on):
    steps, n = i_ext.shape
    cell = AdaptiveLIF(lif, n, syn, dt) if model == "lif" else HodgkinHuxley(hh, n, syn, dt)
    synapses = ExponentialSynapses(weights, syn, dt, stp if stp_on else None, stp_mask)
    gap_rowsum = gap.sum(axis=1)
    ext_varies, gain_varies = g_ext.shape[0] == steps, release_gain.shape[0] == steps

    v_trace = np.empty((steps, n))
    g_e_trace = np.empty((steps, n), dtype=np.float32)
    g_i_trace = np.empty((steps, n), dtype=np.float32)
    raster = np.zeros((steps, n), dtype=bool)
    for k in range(steps):
        k_ext = k if ext_varies else 0
        g_extra, num_extra = g_ext[k_ext], ge_ext[k_ext]
        if gap_on:
            g_extra = g_extra + gap_rowsum
            num_extra = num_extra + gap @ cell.v
        g_e_trace[k], g_i_trace[k] = synapses.g_exc, synapses.g_inh
        spiked, v_trace[k] = cell.step(i_ext[k], synapses.g_exc, synapses.g_inh, g_extra, num_extra)
        if gap_on and model == "lif" and spiked.any():
            cell.spikelets(spiked, gap)
        raster[k] = spiked
        synapses.update(spiked, release_gain[k if gain_varies else 0])
    return v_trace, raster, g_e_trace, g_i_trace


@lru_cache(maxsize=8)
def _cached_gate_tables(dt: float):
    return gate_tables(dt)


def _run_numba(weights, i_ext, dt, model, lif, hh, syn, *, gap, gap_on, g_ext, ge_ext, release_gain,
               stp, stp_mask, stp_on):
    w_exc = np.ascontiguousarray(np.clip(weights, 0.0, None))
    w_inh = np.ascontiguousarray(np.clip(-weights, 0.0, None))
    rec_decay = np.exp(-dt / stp.tau_rec) if stp.tau_rec > 0 else 0.0
    fac_decay = np.exp(-dt / stp.tau_fac) if stp.tau_fac > 0 else 0.0
    head = (w_exc, w_inh, gap, gap_on, i_ext, g_ext, ge_ext, release_gain,
            stp_on, stp_mask, float(stp.U), rec_decay, fac_decay, dt)
    tail = (np.exp(-dt / syn.tau_exc), np.exp(-dt / syn.tau_inh), syn.e_exc, syn.e_inh,
            delay_steps(syn.delay, dt))
    if model == "lif":
        return fast.lif_kernel(*head, lif.v_rest, lif.v_thresh, lif.v_reset, lif.tau_m, lif.r_m, lif.t_ref,
                               lif.theta_jump, lif.tau_theta, lif.v_peak, lif.spike_width, *tail)
    scale = hh.area_um2 * 1e-2  # mS/cm² → nS  e  µF/cm² → pF
    x_inf, x_decay = _cached_gate_tables(dt)
    return fast.hh_kernel(*head, x_inf, x_decay, V_MIN, V_MAX, 1.0 / V_RES,
                          hh.g_na * scale, hh.g_k * scale, hh.g_l * scale, hh.e_na, hh.e_k, hh.e_l,
                          hh.c_m * scale, hh.v_init, hh.spike_detect, *tail)
