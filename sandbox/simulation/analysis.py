"""Analisi dei treni di spike e dei segnali di membrana."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

from .engine import SimulationResult


def firing_rates(result: SimulationResult) -> np.ndarray:
    """Frequenza di scarica media (Hz) di ogni neurone sull'intera simulazione."""
    return result.spike_counts / (result.duration / 1000.0)


def isi_cv(spike_times: np.ndarray) -> float:
    """Coefficiente di variazione degli intervalli inter-spike (≈1 Poisson, ≈0 regolare)."""
    if spike_times.size < 3:
        return float("nan")
    isi = np.diff(spike_times)
    return float(isi.std() / isi.mean())


def population_rate(result: SimulationResult, bin_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """PSTH di popolazione: (centri dei bin in ms, frequenza media per neurone in Hz)."""
    edges = np.arange(0.0, result.duration + bin_ms, bin_ms)
    all_spikes = np.concatenate(result.spike_times) if result.spike_times else np.array([])
    counts, _ = np.histogram(all_spikes, bins=edges)
    rate = counts / (len(result.neurons) * bin_ms / 1000.0)
    return edges[:-1] + bin_ms / 2, rate


def summary_table(result: SimulationResult) -> pd.DataFrame:
    rates = firing_rates(result)
    rows = []
    for name, spikes, rate, v in zip(result.neurons, result.spike_times, rates, result.v.T):
        rows.append({
            "Neurone": name,
            "Spike": int(spikes.size),
            "Frequenza (Hz)": round(float(rate), 2),
            "Primo spike (ms)": round(float(spikes[0]), 2) if spikes.size else None,
            "CV ISI": round(isi_cv(spikes), 3),
            "V media (mV)": round(float(v.mean()), 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- singolo neurone

def isi_histogram(spike_times: np.ndarray, bin_ms: float = 2.0, max_ms: float | None = None):
    """Istogramma degli intervalli inter-spike: (centri dei bin, conteggi)."""
    isi = np.diff(spike_times)
    if isi.size == 0:
        return np.array([]), np.array([])
    top = max_ms or float(isi.max()) + bin_ms
    edges = np.arange(0.0, top + bin_ms, bin_ms)
    counts, _ = np.histogram(isi, bins=edges)
    return edges[:-1] + bin_ms / 2, counts


def instantaneous_rate(spike_times: np.ndarray, t: np.ndarray, sigma_ms: float = 10.0) -> np.ndarray:
    """Frequenza istantanea (Hz): treno di spike convoluto con un kernel gaussiano di area 1."""
    dt = t[1] - t[0]
    idx = np.clip(np.round(spike_times / dt).astype(int), 0, t.size - 1)
    train = np.bincount(idx, minlength=t.size).astype(float) * (1000.0 / dt)
    return gaussian_filter1d(train, sigma_ms / dt, mode="constant")


def cross_correlogram(reference: np.ndarray, target: np.ndarray, window_ms: float = 50.0, bin_ms: float = 1.0):
    """Cross-correlogramma: frequenza di ``target`` (Hz) in funzione del ritardo da ogni spike di ``reference``.

    Picco a ritardo positivo → target tende a scaricare DOPO reference (es. connessione eccitatoria).
    """
    edges = np.arange(-window_ms, window_ms + bin_ms / 2, bin_ms)
    counts = np.zeros(edges.size - 1)
    if reference.size and target.size:
        target = np.sort(target)
        lo = np.searchsorted(target, reference - window_ms)
        hi = np.searchsorted(target, reference + window_ms)
        lags = np.concatenate([target[a:b] - r for r, a, b in zip(reference, lo, hi)])
        counts, _ = np.histogram(lags, bins=edges)
    rate = counts / (max(reference.size, 1) * bin_ms / 1000.0)
    return edges[:-1] + bin_ms / 2, rate


def spike_triggered_average(stimulus: np.ndarray, dt: float, spike_times: np.ndarray, window_ms: float = 50.0):
    """Media dello stimolo nei ``window_ms`` che precedono ogni spike (correlazione inversa).

    Restituisce (ritardi da −window a 0 ms, STA, numero di spike usati).
    """
    n_lag = int(round(window_ms / dt))
    idx = np.round(spike_times / dt).astype(int)
    idx = idx[(idx >= n_lag) & (idx < stimulus.size)]
    lags = np.arange(-n_lag, 1) * dt
    if idx.size == 0:
        return lags, np.full(lags.size, np.nan), 0
    segments = stimulus[idx[:, None] + np.arange(-n_lag, 1)[None, :]]
    return lags, segments.mean(axis=0), int(idx.size)


def power_spectrum(signal: np.ndarray, dt_ms: float, max_hz: float = 200.0):
    """Densità spettrale (metodo di Welch) di un segnale campionato ogni dt_ms. Frequenze in Hz."""
    fs = 1000.0 / dt_ms
    nperseg = min(signal.size, max(16, int(fs)))  # finestre da ~1 s (o l'intero segnale)
    freqs, psd = welch(signal - signal.mean(), fs=fs, nperseg=nperseg)
    keep = (freqs > 0) & (freqs <= max_hz)
    return freqs[keep], psd[keep]


def synaptic_currents(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """Correnti sinaptiche (pA): I = g·(E − V), positive = depolarizzanti."""
    e_exc, e_inh = result.meta.get("e_exc", 0.0), result.meta.get("e_inh", -75.0)
    return result.g_exc * (e_exc - result.v), result.g_inh * (e_inh - result.v)


# ----------------------------------------------------------------------------- confronti e trial

def compare_rates(control: SimulationResult, current: SimulationResult) -> pd.DataFrame:
    """Frequenze di controllo e attuali, neurone per neurone."""
    rc, rn = firing_rates(control), firing_rates(current)
    return pd.DataFrame({
        "Neurone": current.neurons,
        "Controllo (Hz)": np.round(rc, 2),
        "Attuale (Hz)": np.round(rn, 2),
        "Δ (Hz)": np.round(rn - rc, 2),
        "Variazione %": np.round(np.where(rc > 0, 100 * (rn - rc) / np.where(rc > 0, rc, 1), np.nan), 1),
    })


def fano_factor(counts: np.ndarray) -> float:
    """Varianza/media dei conteggi fra trial (1 per un processo di Poisson, 0 se deterministico)."""
    counts = np.asarray(counts, dtype=float)
    return float(counts.var() / counts.mean()) if counts.mean() > 0 else float("nan")


def spike_reliability(trials: list[np.ndarray], t: np.ndarray, sigma_ms: float = 3.0) -> float:
    """Affidabilità di Schreiber et al. (2003): correlazione media fra coppie di trial dei treni
    di spike filtrati con una gaussiana. 1 = spike identici in ogni trial."""
    filtered = [instantaneous_rate(s, t, sigma_ms) for s in trials if s.size]
    if len(filtered) < 2:
        return float("nan")
    x = np.array(filtered)
    norms = np.linalg.norm(x, axis=1)
    corr = (x @ x.T) / np.outer(norms, norms)
    upper = corr[np.triu_indices(len(filtered), k=1)]
    return float(upper.mean())


def trial_psth(trials: list[np.ndarray], duration: float, bin_ms: float = 10.0):
    """PSTH medio fra trial (Hz): (centri dei bin, frequenza, errore standard)."""
    edges = np.arange(0.0, duration + bin_ms, bin_ms)
    counts = np.array([np.histogram(s, bins=edges)[0] for s in trials], dtype=float)
    rates = counts / (bin_ms / 1000.0)
    sem = rates.std(axis=0, ddof=1) / np.sqrt(len(trials)) if len(trials) > 1 else np.zeros(edges.size - 1)
    return edges[:-1] + bin_ms / 2, rates.mean(axis=0), sem
