"""Da specifica (dict serializzabile in JSON) a segnali di stimolo.

Un canale è descritto da un dict piatto, per esempio::

    {"kind": "Onda quadra", "targets": ["N1"], "amplitude": 30, "t_on": 50, "t_off": 400}

I parametri mancanti prendono i valori di ``DEFAULTS[kind]``. Separare la
specifica dalla generazione permette di rigenerare gli stimoli con semi diversi
(trial ripetuti), di variarli nei protocolli e di salvarli in un file JSON.
"""

from dataclasses import dataclass

import numpy as np

from . import sensory
from .conductance import OPSINS, light_pulses, opsin_conductance, poisson_conductance
from .expression import ExpressionError, evaluate_expression
from .waveforms import chirp, from_samples, gaussian_noise, moving_bar, periodic, ramp, square_pulse, staircase

DEFAULT_EXPRESSION = "20 + 20 * np.sin(2 * np.pi * 0.01 * t) * (t > 50)"
RECEPTORS = ("Eccitatoria (ACh → E_exc)", "Inibitoria (GABA/Glu → E_inh)")

CURRENT_KINDS = ("Onda quadra", "Rampa", "Scala di gradini", "Onda periodica", "Chirp (ZAP)",
                 "Rumore gaussiano", "Barra in movimento", "Equazione custom", "Da file CSV", "Sensoriale")
CONDUCTANCE_KINDS = ("Input sinaptico di Poisson", "Optogenetica")
KINDS = CURRENT_KINDS + CONDUCTANCE_KINDS

DEFAULTS = {
    "Onda quadra": dict(amplitude=30.0, frequency=0.0, width=5.0, t_on=50.0, t_off=400.0, baseline=0.0),
    "Rampa": dict(start=0.0, end=60.0, t_on=50.0, t_off=400.0, hold=False),
    "Scala di gradini": dict(start=5.0, step=5.0, n_steps=6, step_ms=60.0, gap_ms=20.0, t_on=20.0),
    "Onda periodica": dict(shape="Sinusoide", amplitude=20.0, dc=20.0, frequency=10.0, phase=0.0,
                           t_on=50.0, t_off=400.0),
    "Chirp (ZAP)": dict(amplitude=10.0, dc=0.0, f0=1.0, f1=100.0, method="Lineare", t_on=0.0, t_off=1e9),
    "Rumore gaussiano": dict(mean=20.0, sigma=10.0, tau_corr=5.0, t_on=0.0, t_off=1e9, independent=True),
    "Barra in movimento": dict(amplitude=40.0, pulse_ms=20.0, interval_ms=15.0, t_on=50.0, direction="→",
                               repeats=1, repeat_ms=200.0),
    "Equazione custom": dict(expression=DEFAULT_EXPRESSION),
    "Da file CSV": dict(samples_t=[], samples_v=[], offset_ms=0.0, scale=1.0, source=""),
    "Sensoriale": dict(modality="Neurone olfattivo (ORN)", profile="Impulso", amplitude=0.6, background=0.0,
                       t_on=100.0, t_off=400.0, level_start=0.0, level_end=0.8, mean=0.5, contrast=0.8,
                       frequency=5.0, rate=8.0, puff_ms=15.0, peak=0.8, t_collision=350.0, l_over_v=40.0),
    "Input sinaptico di Poisson": dict(rate=200.0, g_peak=0.5, tau=5.0, receptor=RECEPTORS[0],
                                       t_on=0.0, t_off=1e9),
    "Optogenetica": dict(opsin="CsChrimson (attivazione)", intensity=0.8, t_on=100.0, t_off=300.0,
                         frequency=0.0, pulse_ms=10.0, g_max=None),
}
TRANSDUCTION_FIELDS = ("half_sat", "hill_n", "gain_pA", "tau_fast", "tau_slow", "adaptation")


@dataclass
class ChannelOutput:
    label: str
    kind: str
    targets: list[str]
    unit: str                              # "pA" (corrente) o "nS" (conduttanza)
    trace: np.ndarray                      # forma d'onda mostrata nel grafico (1ª realizzazione)
    per_target: np.ndarray | None = None   # (passi, n_bersagli) se i bersagli ricevono segnali diversi
    e_rev: float | None = None             # potenziale di inversione (solo conduttanze)
    signal: np.ndarray | None = None       # segnale fisico normalizzato (luce, odore, looming)
    signal_label: str | None = None
    error: str | None = None

    def for_target(self, j: int) -> np.ndarray:
        return self.per_target[:, j] if self.per_target is not None else self.trace


ALIASES = {"Sinusoide": "Onda periodica"}  # nomi usati dalle versioni precedenti


def with_defaults(spec: dict) -> dict:
    kind = ALIASES.get(spec.get("kind", "Onda quadra"), spec.get("kind", "Onda quadra"))
    params = {**DEFAULTS[kind], **spec, "kind": kind}
    if kind == "Sensoriale":
        base = sensory.MODALITIES[params["modality"]]["params"]
        for name in TRANSDUCTION_FIELDS:
            params.setdefault(name, getattr(base, name))
    return params


def _per_target(matrix: np.ndarray):
    return (matrix[:, 0] if matrix.shape[1] else np.zeros(matrix.shape[0])), (matrix if matrix.shape[1] > 1 else None)


def build_channel(spec: dict, t: np.ndarray, dt: float, seed: int, *, index: int = 0,
                  e_exc: float = 0.0, e_inh: float = -75.0) -> ChannelOutput:
    """Genera i segnali di un canale. Gli errori (es. equazione non valida) finiscono in ``error``."""
    p = with_defaults(spec)
    kind, targets = p["kind"], list(p.get("targets", []))
    n_targets = len(targets)
    short = ", ".join(targets[:3]) + ("…" if n_targets > 3 else "") if targets else "nessun bersaglio"
    out = ChannelOutput(label=f"S{index + 1} → {short}", kind=kind, targets=targets, unit="pA",
                        trace=np.zeros_like(t))

    if kind == "Onda quadra":
        out.trace = square_pulse(t, p["amplitude"], p["t_on"], p["t_off"], p["frequency"], p["width"], p["baseline"])
    elif kind == "Rampa":
        out.trace = ramp(t, p["start"], p["end"], p["t_on"], p["t_off"], hold=p["hold"])
    elif kind == "Scala di gradini":
        out.trace = staircase(t, p["start"], p["step"], p["n_steps"], p["step_ms"], p["t_on"], p["gap_ms"])
    elif kind == "Onda periodica":
        out.trace = periodic(t, p["amplitude"], p["frequency"], p["shape"], p["t_on"], p["t_off"], p["phase"], p["dc"])
    elif kind == "Chirp (ZAP)":
        out.trace = chirp(t, p["amplitude"], p["f0"], p["f1"], p["t_on"], p["t_off"], p["method"], p["dc"])
    elif kind == "Rumore gaussiano":
        n_real = max(n_targets, 1) if p["independent"] else 1
        noise = np.column_stack([gaussian_noise(t, p["mean"], p["sigma"], p["tau_corr"], p["t_on"], p["t_off"],
                                                seed=seed + j) for j in range(n_real)])
        out.trace, out.per_target = _per_target(noise)
        if out.per_target is not None:
            out.label += " (1ª realizzazione)"
    elif kind == "Barra in movimento":
        bar = moving_bar(t, max(n_targets, 1), p["amplitude"], p["pulse_ms"], p["interval_ms"], p["t_on"],
                         p["direction"], p["repeats"], p["repeat_ms"])
        out.trace, out.per_target = bar.max(axis=1), (bar if n_targets > 1 else None)
        out.label = f"S{index + 1} barra {p['direction']} ({short})"
    elif kind == "Equazione custom":
        try:
            out.trace = evaluate_expression(p["expression"], t, seed=seed)
        except ExpressionError as exc:
            out.error = f"Equazione non valida — {exc}"
    elif kind == "Da file CSV":
        out.trace = from_samples(t, p["samples_t"], p["samples_v"], p["offset_ms"], p["scale"])
    elif kind == "Sensoriale":
        out.signal = _sensory_signal(p, t, seed)
        params = sensory.TransductionParams(**{name: float(p[name]) for name in TRANSDUCTION_FIELDS})
        out.trace = sensory.transduce(out.signal, dt, params)
        out.signal_label = sensory.MODALITIES[p["modality"]]["signal_label"]
    elif kind == "Input sinaptico di Poisson":
        g = poisson_conductance(t, p["rate"], p["g_peak"], p["tau"], max(n_targets, 1), p["t_on"], p["t_off"], seed)
        out.trace, out.per_target = _per_target(g)
        out.unit, out.e_rev = "nS", (e_exc if p["receptor"] == RECEPTORS[0] else e_inh)
        if out.per_target is not None:
            out.label += " (1ª realizzazione)"
    elif kind == "Optogenetica":
        opsin = OPSINS[p["opsin"]]
        out.signal = light_pulses(t, p["intensity"], p["t_on"], p["t_off"], p["frequency"], p["pulse_ms"])
        out.signal_label = f"Luce {opsin.color} (norm.)"
        out.trace = opsin_conductance(out.signal, dt, opsin, p["g_max"])
        out.unit, out.e_rev = "nS", opsin.e_rev
        out.label = f"S{index + 1} {p['opsin'].split()[0]} → {short}"
    else:
        raise ValueError(f"Tipo di stimolo sconosciuto: {kind}")
    out.trace = np.asarray(out.trace, dtype=float)
    return out


def _sensory_signal(p: dict, t: np.ndarray, seed: int) -> np.ndarray:
    profile = p["profile"]
    if profile == "Impulso":
        return sensory.profile_pulse(t, p["amplitude"], p["t_on"], p["t_off"], p["background"])
    if profile == "Gradiente (rampa)":
        return sensory.profile_gradient(t, p["level_start"], p["level_end"], p["t_on"], p["t_off"])
    if profile == "Sfarfallio sinusoidale":
        return sensory.profile_flicker(t, p["mean"], p["contrast"], p["frequency"], p["t_on"], p["t_off"])
    if profile == "Pennacchio turbolento":
        return sensory.profile_plume(t, p["rate"], p["puff_ms"], p["peak"], p["t_on"], p["t_off"], seed=seed)
    if profile == "Looming":
        return sensory.profile_looming(t, p["t_collision"], p["l_over_v"])
    raise ValueError(f"Profilo sensoriale sconosciuto: {profile}")
