"""Protocolli sperimentali automatici: le "prove da banco" dell'elettrofisiologia.

* ``fi_curve``            curva frequenza-corrente, reobase, adattamento
* ``parameter_sweep``     esplorazione parametrica 1D/2D sull'esperimento corrente
* ``repeated_trials``     trial ripetuti: variabilità, fattore di Fano, affidabilità degli spike
* ``paired_pulse``        plasticità a breve termine: rapporto paired-pulse e treni
* ``impedance_profile``   impedenza di membrana con uno stimolo chirp (ZAP), risonanza
* ``hh_voltage_clamp``    voltage clamp di Hodgkin-Huxley: correnti Na⁺/K⁺ e curve I-V

Tutte le funzioni sono pure (niente Streamlit) e restituiscono tabelle/array.
"""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import HHParams, LIFParams, STPParams, SynapseParams
from .experiment import Experiment, run_experiment
from .models.hodgkin_huxley import rates
from .models.synapses import delay_steps
from .simulation import (fano_factor, firing_rates, isi_cv, simulate, spike_reliability, trial_psth)
from .stimuli import DEFAULTS, chirp, time_vector, with_defaults

# ============================================================================ curva f-I


@dataclass
class FICurve:
    table: pd.DataFrame
    rheobase: float | None       # pA: prima corrente della griglia con almeno uno spike
    t: np.ndarray
    examples: dict               # corrente (pA) → V(t)
    window: tuple[float, float]  # inizio e fine del gradino (ms)


def fi_curve(model: str, lif: LIFParams, hh: HHParams, currents, step_ms: float = 500.0,
             dt: float = 0.1, backend: str = "auto") -> FICurve:
    """Un gradino di corrente per ogni ampiezza. Tutte le ampiezze vengono simulate insieme come
    neuroni indipendenti (W = 0): una sola simulazione per l'intera curva."""
    currents = np.asarray(currents, dtype=float)
    pre = 50.0
    t = time_vector(pre + step_ms + 50.0, dt)
    in_step = (t >= pre) & (t < pre + step_ms)
    n = currents.size
    names = [f"{c:g} pA" for c in currents]
    result = simulate(np.zeros((n, n)), in_step[:, None] * currents[None, :], names, dt,
                      model=model, lif=lif, hh=hh, backend=backend)
    rows = []
    for current, spikes in zip(currents, result.spike_times):
        s = spikes[(spikes >= pre) & (spikes < pre + step_ms)]
        steady = s[s >= pre + step_ms / 2]
        initial = 1000.0 / (s[1] - s[0]) if s.size >= 2 else np.nan
        steady_rate = steady.size / (step_ms / 2) * 1000.0
        rows.append({
            "Corrente (pA)": current,
            "Spike": int(s.size),
            "Frequenza media (Hz)": s.size / step_ms * 1000.0,
            "Frequenza iniziale (Hz)": initial,
            "Frequenza a regime (Hz)": steady_rate,
            "Latenza (ms)": s[0] - pre if s.size else np.nan,
            "Indice di adattamento": 1.0 - steady_rate / initial if s.size >= 2 and initial > 0 else np.nan,
        })
    table = pd.DataFrame(rows)
    firing = table.index[table["Spike"] > 0]
    rheobase = float(currents[firing[0]]) if len(firing) else None
    picks = sorted({int(firing[0]) if len(firing) else 0, n // 2, n - 1})
    examples = {float(currents[j]): result.v[:, j] for j in picks}
    return FICurve(table=table, rheobase=rheobase, t=t, examples=examples, window=(pre, pre + step_ms))


# ============================================================================ esplorazione parametrica

def _section_setter(section: str, attr: str) -> Callable[[Experiment, float], Experiment]:
    return lambda exp, value: replace(exp, **{section: replace(getattr(exp, section), **{attr: value})})


GLOBAL_PARAMETERS: dict[str, Callable[[Experiment, float], Experiment]] = {
    "Guadagno stimoli (×)": lambda exp, v: replace(exp, stim_gain=v),
    "Moltiplicatore pesi chimici (×)": lambda exp, v: replace(exp, weight_scale=v),
    "Moltiplicatore gap junction (×)": lambda exp, v: replace(exp, gap_scale=v),
    "Rumore di fondo σ (pA)": _section_setter("sim", "noise_sigma"),
    "Sinapsi: τ_exc (ms)": _section_setter("syn", "tau_exc"),
    "Sinapsi: τ_inh (ms)": _section_setter("syn", "tau_inh"),
    "Sinapsi: E_inh (mV)": _section_setter("syn", "e_inh"),
    "Sinapsi: ritardo (ms)": _section_setter("syn", "delay"),
    "STP: U": _section_setter("stp", "U"),
    "STP: τ_rec (ms)": _section_setter("stp", "tau_rec"),
    "STP: τ_fac (ms)": _section_setter("stp", "tau_fac"),
    "LIF: V_thresh (mV)": _section_setter("lif", "v_thresh"),
    "LIF: τ_m (ms)": _section_setter("lif", "tau_m"),
    "LIF: R_m (GΩ)": _section_setter("lif", "r_m"),
    "LIF: Δθ (mV)": _section_setter("lif", "theta_jump"),
    "LIF: τ_θ (ms)": _section_setter("lif", "tau_theta"),
    "HH: ḡ_Na (mS/cm²)": _section_setter("hh", "g_na"),
    "HH: ḡ_K (mS/cm²)": _section_setter("hh", "g_k"),
    "HH: area (µm²)": _section_setter("hh", "area_um2"),
}


def _stimulus_setter(k: int, name: str):
    def setter(exp: Experiment, value: float) -> Experiment:
        stimuli = [dict(spec) for spec in exp.stimuli]
        stimuli[k][name] = value
        return replace(exp, stimuli=stimuli)
    return setter


def _weight_setter(i: int, j: int):
    def setter(exp: Experiment, value: float) -> Experiment:
        weights = exp.weights.copy()
        weights[i, j] = value
        return replace(exp, weights=weights)
    return setter


def sweepable_parameters(exp: Experiment) -> dict[str, Callable[[Experiment, float], Experiment]]:
    """Parametri variabili: globali, numerici di ogni canale di stimolo, pesi delle connessioni esistenti."""
    params = dict(GLOBAL_PARAMETERS)
    for k, spec in enumerate(exp.stimuli):
        full = with_defaults(spec)
        for name, value in full.items():
            if name in DEFAULTS[full["kind"]] or name in ("half_sat", "hill_n", "gain_pA", "tau_fast",
                                                          "tau_slow", "adaptation"):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    params[f"S{k + 1} ({full['kind']}): {name}"] = _stimulus_setter(k, name)
    for i, j in zip(*np.nonzero(exp.weights)):
        params[f"W: {exp.neurons[i]} → {exp.neurons[j]} (nS)"] = _weight_setter(int(i), int(j))
    return params


def current_value(exp: Experiment, name: str) -> float | None:
    """Valore attuale di un parametro variabile (per proporre un intervallo sensato)."""
    probes = {"Guadagno stimoli (×)": exp.stim_gain, "Moltiplicatore pesi chimici (×)": exp.weight_scale,
              "Moltiplicatore gap junction (×)": exp.gap_scale, "Rumore di fondo σ (pA)": exp.sim.noise_sigma}
    if name in probes:
        return float(probes[name])
    for prefix, section in (("Sinapsi: ", exp.syn), ("STP: ", exp.stp), ("LIF: ", exp.lif), ("HH: ", exp.hh)):
        if name.startswith(prefix):
            attr = {"τ_exc (ms)": "tau_exc", "τ_inh (ms)": "tau_inh", "E_inh (mV)": "e_inh", "ritardo (ms)": "delay",
                    "U": "U", "τ_rec (ms)": "tau_rec", "τ_fac (ms)": "tau_fac", "V_thresh (mV)": "v_thresh",
                    "τ_m (ms)": "tau_m", "R_m (GΩ)": "r_m", "Δθ (mV)": "theta_jump", "τ_θ (ms)": "tau_theta",
                    "ḡ_Na (mS/cm²)": "g_na", "ḡ_K (mS/cm²)": "g_k", "area (µm²)": "area_um2"}.get(name[len(prefix):])
            return float(getattr(section, attr)) if attr else None
    if name.startswith("W: "):
        pre, post = name[3:].rsplit(" (", 1)[0].split(" → ")
        return float(exp.weights[exp.neurons.index(pre), exp.neurons.index(post)])
    if name.startswith("S") and ": " in name:
        k = int(name[1:name.index(" ")]) - 1
        return float(with_defaults(exp.stimuli[k])[name.rsplit(": ", 1)[1]])
    return None


def _first_spike(result):
    return np.array([s[0] if s.size else np.nan for s in result.spike_times])


METRICS: dict[str, Callable] = {
    "Frequenza media (Hz)": firing_rates,
    "Numero di spike": lambda r: r.spike_counts.astype(float),
    "Latenza primo spike (ms)": _first_spike,
    "CV ISI": lambda r: np.array([isi_cv(s) for s in r.spike_times]),
    "V media (mV)": lambda r: r.v.mean(axis=0),
    "V massimo (mV)": lambda r: r.v.max(axis=0),
}


def parameter_sweep(exp: Experiment, param: str, values, metric: str, param2: str | None = None,
                    values2=None, backend: str = "auto", progress: Callable[[float], None] | None = None,
                    ) -> pd.DataFrame:
    """Esegue l'esperimento per ogni valore (o coppia di valori) e misura ``metric`` su ogni neurone.

    Restituisce una tabella "lunga": una riga per (valore, [valore2,] neurone).
    """
    setters = sweepable_parameters(exp)
    grid = [(v, None) for v in values] if param2 is None else [(v, w) for v in values for w in values2]
    rows = []
    for count, (v, w) in enumerate(grid, start=1):
        variant = setters[param](exp, float(v))
        if param2 is not None:
            variant = setters[param2](variant, float(w))
        result, _ = run_experiment(variant, backend=backend)
        for name, value in zip(exp.neurons, METRICS[metric](result)):
            row = {param: float(v), "Neurone": name, metric: float(value)}
            if param2 is not None:
                row[param2] = float(w)
            rows.append(row)
        if progress:
            progress(count / len(grid))
    return pd.DataFrame(rows)


# ============================================================================ trial ripetuti


@dataclass
class TrialsResult:
    neurons: list[str]
    t: np.ndarray
    duration: float
    spikes: list[list[np.ndarray]]  # [trial][neurone]
    frozen: bool

    @property
    def counts(self) -> np.ndarray:
        return np.array([[s.size for s in trial] for trial in self.spikes])

    def neuron_trials(self, j: int) -> list[np.ndarray]:
        return [trial[j] for trial in self.spikes]

    def psth(self, j: int, bin_ms: float = 10.0):
        return trial_psth(self.neuron_trials(j), self.duration, bin_ms)

    def statistics(self, sigma_ms: float = 3.0) -> pd.DataFrame:
        counts = self.counts
        rates = counts / (self.duration / 1000.0)
        return pd.DataFrame({
            "Neurone": self.neurons,
            "Frequenza media (Hz)": np.round(rates.mean(axis=0), 2),
            "Dev. std fra trial (Hz)": np.round(rates.std(axis=0, ddof=1) if len(counts) > 1 else 0.0, 2),
            "Fattore di Fano": [round(fano_factor(counts[:, j]), 3) for j in range(len(self.neurons))],
            "Affidabilità": [round(spike_reliability(self.neuron_trials(j), self.t, sigma_ms), 3)
                             for j in range(len(self.neurons))],
        })


def repeated_trials(exp: Experiment, n_trials: int, freeze_stimulus: bool = False, backend: str = "auto",
                    progress: Callable[[float], None] | None = None) -> TrialsResult:
    spikes = []
    for trial in range(n_trials):
        result, _ = run_experiment(exp, trial=trial, backend=backend, freeze_stimulus=freeze_stimulus)
        spikes.append(result.spike_times)
        if progress:
            progress((trial + 1) / n_trials)
    return TrialsResult(neurons=list(exp.neurons), t=exp.t, duration=exp.sim.duration, spikes=spikes,
                        frozen=freeze_stimulus)


# ============================================================================ plasticità a breve termine


def stp_efficacies(spike_steps, stp: STPParams, dt: float) -> np.ndarray:
    """Efficacia di ogni spike (u·x/U) con la stessa ricorsione per passi del motore, in forma chiusa."""
    rec = np.exp(-dt / stp.tau_rec) if stp.tau_rec > 0 else 0.0
    fac = np.exp(-dt / stp.tau_fac) if stp.tau_fac > 0 else 0.0
    x, u, last = 1.0, stp.U, None
    out = []
    for step in spike_steps:
        m = step + 1 if last is None else step - last  # passi di recupero dall'ultimo spike (incluso questo)
        x = 1.0 - (1.0 - x) * rec ** m
        u = stp.U + (u - stp.U) * fac ** m
        if not stp.enabled:
            out.append(1.0)
        else:
            release = u * x
            out.append(release / stp.U)
            x -= release
            u += stp.U * (1.0 - u)
        last = step
    return np.array(out)


@dataclass
class SynapseTest:
    pairs: pd.DataFrame      # un intervallo per riga: efficacie e PSP del 1° e 2° spike
    train: pd.DataFrame      # treno: efficacia e PSP di ogni impulso, normalizzati al primo
    t: np.ndarray            # tracciato d'esempio del treno
    v: np.ndarray
    g: np.ndarray
    spike_times: np.ndarray


def _psp_response(spike_times, weight, stp, syn, lif, dt, duration):
    """Conduttanza evocata dagli spike e risposta di una membrana postsinaptica passiva."""
    t = time_vector(duration, dt)
    steps = np.round(np.asarray(spike_times) / dt).astype(int)
    eff = stp_efficacies(steps, stp, dt)
    tau = syn.tau_exc if weight >= 0 else syn.tau_inh
    e_rev = syn.e_exc if weight >= 0 else syn.e_inh
    decay = np.exp(-dt / tau)
    g = np.zeros(t.size)
    arrivals = steps + delay_steps(syn.delay, dt) + 1  # come nel motore: effetto dal passo successivo alla consegna
    for a, e in zip(arrivals, eff):
        if a < t.size:
            g[a:] += abs(weight) * e * decay ** np.arange(t.size - a)
    passive = replace(lif, v_thresh=1e9, theta_jump=0.0)
    r = simulate(np.zeros((1, 1)), np.zeros((t.size, 1)), ["post"], dt, lif=passive,
                 g_ext=g[:, None], ge_ext=(g * e_rev)[:, None])
    v = r.v[:, 0]
    bounds = list(arrivals) + [t.size]
    amps = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a = min(a, t.size - 1)
        segment = v[a:max(b, a + 1)]
        base = v[a - 1] if a > 0 else lif.v_rest
        amps.append((segment.max() if weight >= 0 else segment.min()) - base)
    return t, v, g, eff, np.array(amps)


def paired_pulse(weight: float, stp: STPParams, syn: SynapseParams, lif: LIFParams, isis,
                 train_hz: float = 20.0, n_train: int = 10, dt: float = 0.1) -> SynapseTest:
    """Rapporto paired-pulse (2° / 1°) a vari intervalli e risposta a un treno regolare.

    Banco di prova isolato: spike presinaptici imposti, sinapsi con la plasticità scelta e
    membrana postsinaptica passiva (τ_m e R_m del LIF), così che l'ampiezza dei PSP rifletta
    solo la sinapsi.
    """
    t0 = 20.0
    rows = []
    for isi in isis:
        _, _, _, eff, amps = _psp_response([t0, t0 + isi], weight, stp, syn, lif, dt, t0 + isi + 150.0)
        rows.append({"Intervallo (ms)": float(isi), "Efficacia 1": eff[0], "Efficacia 2": eff[1],
                     "PPR (efficacia)": eff[1] / eff[0], "PSP 1 (mV)": amps[0], "PSP 2 (mV)": amps[1],
                     "PPR (PSP)": amps[1] / amps[0] if amps[0] else np.nan})
    interval = 1000.0 / train_hz
    times = t0 + interval * np.arange(n_train)
    t, v, g, eff, amps = _psp_response(times, weight, stp, syn, lif, dt, times[-1] + 150.0)
    train = pd.DataFrame({"Impulso": np.arange(1, n_train + 1), "Tempo (ms)": times,
                          "Efficacia (norm.)": eff / eff[0],
                          "PSP (mV)": amps, "PSP (norm.)": amps / amps[0] if amps[0] else np.nan})
    return SynapseTest(pairs=pd.DataFrame(rows), train=train, t=t, v=v, g=g, spike_times=times)


# ============================================================================ impedenza (ZAP)


@dataclass
class ImpedanceProfile:
    freqs: np.ndarray        # Hz
    magnitude: np.ndarray    # |Z| in GΩ (= mV/pA)
    phase: np.ndarray        # gradi (negativo = V in ritardo su I)
    resonance_hz: float
    q_factor: float          # |Z|max / |Z| alla frequenza più bassa
    t: np.ndarray
    current: np.ndarray
    v: np.ndarray
    spikes: int


def _smooth(y, width):
    if width <= 1:
        return y
    kernel = np.ones(width) / width
    return np.convolve(np.pad(y, width // 2, mode="edge"), kernel, mode="valid")[: y.size]


def impedance_profile(model: str, lif: LIFParams, hh: HHParams, amplitude: float = 5.0, f0: float = 0.5,
                      f1: float = 100.0, duration: float = 4000.0, bias: float = 0.0, dt: float = 0.1,
                      backend: str = "auto") -> ImpedanceProfile:
    """Z(f) = FFT(V) / FFT(I) con una corrente chirp sottosoglia (più eventuale polarizzazione DC).

    Un LIF passivo è un filtro passa-basso (|Z| = R/√(1 + (2πfτ)²)); Hodgkin-Huxley mostra una
    risonanza sottosoglia dovuta alla cinetica di Na⁺ e K⁺.
    """
    settle = 300.0  # la polarizzazione DC si stabilizza prima del chirp
    t = time_vector(settle + duration, dt)
    current = bias + chirp(t, amplitude, f0, f1, settle, settle + duration, "Lineare")
    r = simulate(np.zeros((1, 1)), current[:, None], ["cellula"], dt, model=model, lif=lif, hh=hh, backend=backend)
    keep = t >= settle
    v_seg, i_seg = r.v[keep, 0], current[keep]
    v_f = np.fft.rfft(v_seg - v_seg.mean())
    i_f = np.fft.rfft(i_seg - i_seg.mean())
    freqs = np.fft.rfftfreq(v_seg.size, dt / 1000.0)
    band = (freqs >= max(f0, freqs[1])) & (freqs <= f1) & (np.abs(i_f) > 1e-9 * np.abs(i_f).max())
    z = v_f[band] / i_f[band]
    width = max(1, int(band.sum() // 150))
    magnitude = _smooth(np.abs(z), width)
    phase = np.degrees(np.unwrap(np.angle(z)))
    peak = int(np.argmax(magnitude))
    return ImpedanceProfile(freqs=freqs[band], magnitude=magnitude, phase=_smooth(phase, width),
                            resonance_hz=float(freqs[band][peak]), q_factor=float(magnitude[peak] / magnitude[0]),
                            t=t, current=current, v=r.v[:, 0], spikes=int(r.spike_counts[0]))


# ============================================================================ voltage clamp (HH)


@dataclass
class VoltageClamp:
    t: np.ndarray                 # ms
    commands: np.ndarray          # potenziali di comando (mV)
    i_na: np.ndarray              # (n_comandi, passi) pA, negativo = entrante
    i_k: np.ndarray
    i_leak: np.ndarray
    g_na: np.ndarray              # nS
    g_k: np.ndarray
    iv: pd.DataFrame


def hh_voltage_clamp(hh: HHParams, v_hold: float = -80.0, commands=None, step_ms: float = 15.0,
                     pre_ms: float = 2.0, post_ms: float = 5.0, dt: float = 0.01) -> VoltageClamp:
    """Esperimento di Hodgkin & Huxley (1952): gradini di potenziale imposto.

    A V costante le equazioni dei gate hanno soluzione esatta
    x(t) = x∞(V) + (x₀ − x∞(V))·exp(−t/τ_x(V)), quindi non serve integrare.
    """
    commands = np.arange(-70.0, 61.0, 10.0) if commands is None else np.asarray(commands, dtype=float)
    scale = hh.area_um2 * 1e-2
    g_na_max, g_k_max, g_l = hh.g_na * scale, hh.g_k * scale, hh.g_l * scale
    t = np.arange(0.0, pre_ms + step_ms + post_ms, dt)

    def steady(v):
        return np.array([a / (a + b) for a, b in rates(np.array([v]))]).ravel()

    def relax(x0, v, duration):
        ab = rates(np.array([v]))
        x_inf = np.array([a / (a + b) for a, b in ab]).ravel()
        tau = np.array([1.0 / (a + b) for a, b in ab]).ravel()
        return x_inf[:, None] + (x0 - x_inf)[:, None] * np.exp(-duration[None, :] / tau[:, None])

    hold_gates = steady(v_hold)
    phases = [(t < pre_ms), (t >= pre_ms) & (t < pre_ms + step_ms), (t >= pre_ms + step_ms)]
    i_na, i_k, i_l, g_na_all, g_k_all, rows = [], [], [], [], [], []
    for v_cmd in commands:
        gates = np.empty((3, t.size))
        gates[:, phases[0]] = hold_gates[:, None]
        step_t = t[phases[1]] - pre_ms
        gates[:, phases[1]] = relax(hold_gates, v_cmd, step_t)
        end = relax(hold_gates, v_cmd, np.array([step_ms]))[:, 0]
        gates[:, phases[2]] = relax(end, v_hold, t[phases[2]] - pre_ms - step_ms)
        v = np.where(phases[1], v_cmd, v_hold)
        m, h, n = gates
        g_na, g_k = g_na_max * m ** 3 * h, g_k_max * n ** 4
        ina, ik = g_na * (v - hh.e_na), g_k * (v - hh.e_k)
        i_na.append(ina), i_k.append(ik), i_l.append(g_l * (v - hh.e_l))
        g_na_all.append(g_na), g_k_all.append(g_k)
        during = phases[1]
        rows.append({"V comando (mV)": v_cmd,
                     "I_Na di picco (pA)": float(ina[during][np.argmax(np.abs(ina[during]))]),
                     "I_K a regime (pA)": float(ik[during][-1]),
                     "g_Na di picco (nS)": float(g_na[during].max()),
                     "g_K a regime (nS)": float(g_k[during][-1])})
    return VoltageClamp(t=t, commands=commands, i_na=np.array(i_na), i_k=np.array(i_k), i_leak=np.array(i_l),
                        g_na=np.array(g_na_all), g_k=np.array(g_k_all), iv=pd.DataFrame(rows))
