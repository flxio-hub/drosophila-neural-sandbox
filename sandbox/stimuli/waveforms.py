"""Forme d'onda preimpostate per l'iniezione di corrente I(t).

Tutte le funzioni ricevono il vettore dei tempi ``t`` (ms) e restituiscono un
array della stessa lunghezza in pA. ``t_on``/``t_off`` delimitano la finestra
in cui lo stimolo è attivo.
"""

import numpy as np
from scipy.signal import chirp as scipy_chirp
from scipy.signal import lfilter, sawtooth

PERIODIC_SHAPES = ("Sinusoide", "Triangolare", "Dente di sega", "Quadra")
CHIRP_METHODS = {"Lineare": "linear", "Logaritmico": "logarithmic"}


def time_vector(duration: float, dt: float) -> np.ndarray:
    """Vettore dei tempi [0, duration) con passo dt (evita l'accumulo d'errore di arange)."""
    n_steps = int(round(duration / dt))
    return np.arange(n_steps) * dt


def _window(t: np.ndarray, t_on: float, t_off: float) -> np.ndarray:
    return (t >= t_on) & (t < t_off)


def square_pulse(t, amplitude, t_on, t_off, frequency_hz=0.0, pulse_width=5.0, baseline=0.0):
    """Treno di impulsi quadrati.

    frequency_hz = 0 produce un singolo gradino di corrente da t_on a t_off;
    altrimenti impulsi larghi ``pulse_width`` ms ripetuti con la frequenza data.
    """
    window = _window(t, t_on, t_off)
    if frequency_hz > 0:
        period = 1000.0 / frequency_hz
        width = min(pulse_width, period)
        window &= ((t - t_on) % period) < width
    return baseline + amplitude * window


def ramp(t, start_amp, end_amp, t_on, t_off, hold=True):
    """Rampa lineare da start_amp a end_amp; dopo t_off mantiene il valore finale se hold."""
    frac = np.clip((t - t_on) / max(t_off - t_on, 1e-9), 0.0, 1.0)
    current = start_amp + (end_amp - start_amp) * frac
    current[t < t_on] = 0.0
    if not hold:
        current[t >= t_off] = 0.0
    return current


def sine(t, amplitude, frequency_hz, t_on=0.0, t_off=np.inf, phase_deg=0.0, dc=0.0):
    """Sinusoide (frequenza in Hz, t in ms) con componente continua dc."""
    phase = 2 * np.pi * frequency_hz * (t - t_on) / 1000.0 + np.deg2rad(phase_deg)
    return _window(t, t_on, t_off) * (dc + amplitude * np.sin(phase))


def periodic(t, amplitude, frequency_hz, shape="Sinusoide", t_on=0.0, t_off=np.inf, phase_deg=0.0, dc=0.0):
    """Onda periodica: sinusoide, triangolare, dente di sega o quadra (duty 50%), tra −A e +A attorno a dc."""
    if shape == "Sinusoide":
        return sine(t, amplitude, frequency_hz, t_on, t_off, phase_deg, dc)
    phase = 2 * np.pi * frequency_hz * (t - t_on) / 1000.0 + np.deg2rad(phase_deg)
    if shape == "Triangolare":
        wave = sawtooth(phase + np.pi / 2, width=0.5)  # parte da 0 salendo, come la sinusoide
    elif shape == "Dente di sega":
        wave = sawtooth(phase + np.pi, width=1.0)      # rampa lenta in salita e ritorno brusco
    elif shape == "Quadra":
        wave = np.where(np.sin(phase) >= 0, 1.0, -1.0)
    else:
        raise ValueError(f"Forma sconosciuta: {shape}")
    return _window(t, t_on, t_off) * (dc + amplitude * wave)


def staircase(t, start_amp, step_amp, n_steps, step_ms, t_on=0.0, gap_ms=0.0):
    """Scala di gradini: start, start+Δ, start+2Δ… ognuno lungo step_ms, separati da pause di gap_ms.

    È il protocollo classico per misurare la curva frequenza-corrente in una sola registrazione.
    """
    current = np.zeros_like(t, dtype=float)
    period = step_ms + gap_ms
    for k in range(int(n_steps)):
        start = t_on + k * period
        current[(t >= start) & (t < start + step_ms)] = start_amp + k * step_amp
    return current


def chirp(t, amplitude, f0_hz, f1_hz, t_on=0.0, t_off=np.inf, method="Lineare", dc=0.0):
    """Chirp / ZAP: sinusoide con frequenza che va da f0 a f1 tra t_on e t_off.

    Usato per misurare l'impedenza di membrana e le risonanze sottosoglia (Hutcheon & Yarom 2000).
    """
    t_end = min(t_off, t[-1] + (t[1] - t[0]) if t.size > 1 else t_off)
    span = max(t_end - t_on, 1e-9) / 1000.0
    rel = np.clip((t - t_on) / 1000.0, 0.0, None)
    scipy_method = CHIRP_METHODS.get(method, method)
    if scipy_method == "logarithmic" and (f0_hz <= 0 or f1_hz <= 0 or f0_hz == f1_hz):
        scipy_method = "linear"  # il log richiede frequenze positive e diverse
    wave = scipy_chirp(rel, f0=f0_hz, t1=span, f1=f1_hz, method=scipy_method, phi=-90.0)
    return _window(t, t_on, t_end) * (dc + amplitude * wave)


def moving_bar(t, n_positions, amplitude, pulse_ms, interval_ms, t_on=0.0, direction="→",
               repeats=1, repeat_ms=200.0):
    """Barra in movimento su posizioni ordinate: ogni posizione riceve un impulso ``interval_ms`` dopo la precedente.

    Restituisce una matrice (passi, n_posizioni). direction "←" inverte l'ordine di attraversamento.
    Velocità apparente = 1 posizione ogni ``interval_ms``.
    """
    out = np.zeros((t.size, int(n_positions)))
    order = range(int(n_positions)) if direction == "→" else range(int(n_positions) - 1, -1, -1)
    for r in range(int(repeats)):
        for s, pos in enumerate(order):
            start = t_on + r * repeat_ms + s * interval_ms
            out[(t >= start) & (t < start + pulse_ms), pos] = amplitude
    return out


def from_samples(t, sample_t, sample_values, offset_ms=0.0, scale=1.0):
    """Riproduce una forma d'onda campionata (es. una corrente registrata) interpolandola su t; 0 fuori range."""
    sample_t = np.asarray(sample_t, dtype=float)
    sample_values = np.asarray(sample_values, dtype=float)
    if sample_t.size == 0:
        return np.zeros_like(t, dtype=float)
    order = np.argsort(sample_t)
    return scale * np.interp(t - offset_ms, sample_t[order], sample_values[order], left=0.0, right=0.0)


def gaussian_noise(t, mean, sigma, tau_corr=0.0, t_on=0.0, t_off=np.inf, seed=None):
    """Rumore gaussiano.

    tau_corr = 0  → campioni indipendenti a ogni passo (rumore bianco discreto:
                    il suo effetto sulla membrana dipende da dt).
    tau_corr > 0  → processo di Ornstein-Uhlenbeck con deviazione standard
                    stazionaria ``sigma``: statistica indipendente da dt, più
                    vicino al rumore sinaptico di fondo reale.
    """
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(t.size)
    if tau_corr <= 0 or t.size < 2:
        noise = sigma * xi
    else:
        dt = t[1] - t[0]
        a = np.exp(-dt / tau_corr)
        # Aggiornamento esatto OU: x[k] = a·x[k-1] + σ·sqrt(1-a²)·ξ[k], partendo dalla distribuzione stazionaria
        x0 = sigma * rng.standard_normal()
        noise, _ = lfilter([sigma * np.sqrt(1 - a * a)], [1.0, -a], xi, zi=[a * x0])
    return _window(t, t_on, t_off) * (mean + noise)
