"""Un esperimento completo: circuito + modello + stimoli + manipolazioni, riproducibile e serializzabile.

``Experiment`` contiene solo dati semplici (parametri, matrici, specifiche degli
stimoli come dict), quindi può essere salvato in JSON, ricaricato, variato dai
protocolli (``dataclasses.replace``) e rieseguito con semi diversi per i trial.

    exp = Experiment(neurons=["A", "B"], weights=W, stimuli=[{"kind": "Onda quadra", "targets": ["A"]}])
    result, inputs = run_experiment(exp)
"""

import json
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from .config import HHParams, LIFParams, SimParams, STPParams, SynapseParams
from .manipulations import apply_manipulations
from .simulation import SimulationResult, simulate
from .simulation.drive import NetworkDrive
from .stimuli import ChannelOutput, build_channel, gaussian_noise, time_vector

FORMAT = "drosophila-sandbox/experiment-v1"
NOISE_SEED_OFFSET = 500_009
TRIAL_SEED_STRIDE = 7_919


@dataclass
class Experiment:
    neurons: list[str]
    weights: np.ndarray                      # W[pre, post] (nS), prima di manipolazioni e moltiplicatori
    gap: np.ndarray | None = None            # gap junction (nS), simmetrica
    model: str = "lif"
    lif: LIFParams = field(default_factory=LIFParams)
    hh: HHParams = field(default_factory=HHParams)
    syn: SynapseParams = field(default_factory=SynapseParams)
    stp: STPParams = field(default_factory=STPParams)
    stp_targets: list[str] | None = None     # neuroni presinaptici con STP (None = tutti)
    sim: SimParams = field(default_factory=SimParams)
    stimuli: list[dict] = field(default_factory=list)
    manipulations: dict = field(default_factory=dict)
    stim_gain: float = 1.0                   # scala tutte le correnti di stimolo
    weight_scale: float = 1.0                # scala le sinapsi chimiche
    gap_scale: float = 1.0                   # scala le gap junction
    name: str = "Esperimento"

    def __post_init__(self):
        n = len(self.neurons)
        self.weights = np.asarray(self.weights, dtype=float).reshape(n, n)
        self.gap = np.zeros((n, n)) if self.gap is None else np.asarray(self.gap, dtype=float).reshape(n, n)

    @property
    def t(self) -> np.ndarray:
        return time_vector(self.sim.duration, self.sim.dt)


@dataclass
class PreparedInputs:
    """Tutto ciò che entra nel motore, già con manipolazioni e moltiplicatori applicati."""
    t: np.ndarray
    weights: np.ndarray
    gap: np.ndarray
    drive: NetworkDrive
    channels: list[ChannelOutput]
    temperature: np.ndarray | None
    notes: list[str]
    stp_mask: np.ndarray


def trial_seed(exp: Experiment, trial: int = 0) -> int:
    return exp.sim.seed + TRIAL_SEED_STRIDE * trial


def prepare(exp: Experiment, trial: int = 0, freeze_stimulus: bool = False) -> PreparedInputs:
    """Costruisce stimoli, rumore di fondo e manipolazioni.

    ``trial`` cambia i semi delle sorgenti casuali. Con ``freeze_stimulus`` gli stimoli
    stocastici restano identici fra i trial ("frozen noise") e varia solo il rumore di
    fondo della membrana: il paradigma di Mainen & Sejnowski (1995) sull'affidabilità.
    """
    t, n = exp.t, len(exp.neurons)
    index = {name: i for i, name in enumerate(exp.neurons)}
    drive = NetworkDrive.zeros(t.size, n)
    base_seed = trial_seed(exp, trial)
    stim_seed = trial_seed(exp, 0 if freeze_stimulus else trial)

    channels = []
    for k, spec in enumerate(exp.stimuli):
        spec = {**spec, "targets": [name for name in spec.get("targets", []) if name in index]}
        out = build_channel(spec, t, exp.sim.dt, stim_seed + 1000 * k, index=k,
                            e_exc=exp.syn.e_exc, e_inh=exp.syn.e_inh)
        channels.append(out)
        if out.error:
            continue
        for j, name in enumerate(out.targets):
            if out.unit == "nS":
                drive.add_conductance(index[name], out.for_target(j), out.e_rev)
            else:
                drive.add_current(index[name], exp.stim_gain * out.for_target(j))

    if exp.sim.noise_sigma > 0:
        for j in range(n):
            drive.add_current(j, gaussian_noise(t, 0.0, exp.sim.noise_sigma, exp.sim.noise_tau,
                                                seed=base_seed + NOISE_SEED_OFFSET + j))

    weights, gap, temperature, notes = apply_manipulations(
        exp.manipulations, exp.neurons, exp.weights * exp.weight_scale, exp.gap * exp.gap_scale, drive, t)
    stp_mask = (np.ones(n, dtype=bool) if exp.stp_targets is None
                else np.array([name in set(exp.stp_targets) for name in exp.neurons]))
    return PreparedInputs(t=t, weights=weights, gap=gap, drive=drive, channels=channels,
                          temperature=temperature, notes=notes, stp_mask=stp_mask)


def simulate_prepared(exp: Experiment, inputs: PreparedInputs, backend: str = "auto") -> SimulationResult:
    return simulate(inputs.weights, inputs.drive.i_ext, exp.neurons, exp.sim.dt, model=exp.model,
                    lif=exp.lif, hh=exp.hh, syn=exp.syn, gap=inputs.gap, stp=exp.stp,
                    stp_mask=inputs.stp_mask, backend=backend, **inputs.drive.simulate_kwargs())


def run_experiment(exp: Experiment, trial: int = 0, backend: str = "auto", freeze_stimulus: bool = False):
    """Prepara e simula. Restituisce (risultato, input preparati)."""
    inputs = prepare(exp, trial, freeze_stimulus)
    return simulate_prepared(exp, inputs, backend), inputs


# ----------------------------------------------------------------------------- JSON

def to_dict(exp: Experiment) -> dict:
    data = {}
    for f in fields(exp):
        value = getattr(exp, f.name)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif hasattr(value, "__dataclass_fields__"):
            value = asdict(value)
        data[f.name] = value
    return {"format": FORMAT, **data}


_PARAM_TYPES = {"lif": LIFParams, "hh": HHParams, "syn": SynapseParams, "stp": STPParams, "sim": SimParams}


def from_dict(data: dict) -> Experiment:
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError(f"Formato non riconosciuto: {data.get('format')}")
    kwargs = {}
    known = {f.name for f in fields(Experiment)}
    for name, value in data.items():
        if name not in known:
            continue
        if name in _PARAM_TYPES:
            cls = _PARAM_TYPES[name]
            allowed = {f.name for f in fields(cls)}
            value = cls(**{k: v for k, v in (value or {}).items() if k in allowed})
        kwargs[name] = value
    return Experiment(**kwargs)


def to_json(exp: Experiment, **extra) -> str:
    return json.dumps({**to_dict(exp), **extra}, ensure_ascii=False, indent=1)


def from_json(text: str) -> Experiment:
    return from_dict(json.loads(text))
