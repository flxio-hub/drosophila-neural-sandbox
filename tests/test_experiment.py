"""Nuovi stimoli, builder delle specifiche, manipolazioni ed esperimenti serializzabili."""

import numpy as np
import pytest

from sandbox.config import SimParams, STPParams
from sandbox.experiment import Experiment, from_json, prepare, run_experiment, to_json
from sandbox.manipulations import shibire_block, temperature_trace, trpa1_activation
from sandbox.network import PRESETS
from sandbox.stimuli import (DEFAULTS, KINDS, OPSINS, build_channel, chirp, light_pulses, moving_bar,
                             opsin_conductance, periodic, poisson_conductance, staircase, time_vector)

T = time_vector(500.0, 0.1)


# ----------------------------------------------------------------------------- forme d'onda

def test_staircase_levels():
    i = staircase(T, 5.0, 5.0, 4, 50.0, t_on=20.0, gap_ms=10.0)
    levels = [i[np.searchsorted(T, 20.0 + k * 60.0 + 25.0)] for k in range(4)]
    assert levels == [5.0, 10.0, 15.0, 20.0] and i[np.searchsorted(T, 75.0)] == 0.0


@pytest.mark.parametrize("shape", ["Sinusoide", "Triangolare", "Dente di sega", "Quadra"])
def test_periodic_shapes_span_amplitude(shape):
    w = periodic(T, 10.0, 20.0, shape)
    assert w.max() == pytest.approx(10.0, abs=0.2) and w.min() == pytest.approx(-10.0, abs=0.2)
    assert abs(w[0]) < 1e-9 or shape == "Quadra"


def test_chirp_frequency_increases():
    t = time_vector(1000.0, 0.1)
    w = chirp(t, 1.0, 2.0, 50.0, 0.0, 1000.0)
    rising = t[1:][(w[:-1] < 0) & (w[1:] >= 0)]           # attraversamenti dello zero verso l'alto
    first, last = (rising < 200.0).sum(), (rising >= 800.0).sum()
    # f(t) = 2 + 48·t[s]: ~1.4 cicli nei primi 200 ms, ~9 negli ultimi 200 ms
    assert first in (1, 2) and last in (8, 9, 10)


def test_moving_bar_order_and_direction():
    right = moving_bar(T, 3, 1.0, 10.0, 20.0, t_on=50.0, direction="→")
    left = moving_bar(T, 3, 1.0, 10.0, 20.0, t_on=50.0, direction="←")
    onset = lambda m: [T[np.argmax(m[:, j] > 0)] for j in range(3)]
    assert onset(right) == pytest.approx([50.0, 70.0, 90.0])
    assert onset(left) == pytest.approx([90.0, 70.0, 50.0])


def test_poisson_conductance_mean():
    t = time_vector(20000.0, 0.1)
    g = poisson_conductance(t, 100.0, 0.5, 5.0, n_trains=2, seed=1)
    # media del processo shot-noise discreto: λ·g_peak / (1 − a), con λ = eventi per passo ≈ rate·g_peak·τ
    events_per_step, a = 100.0 * 0.1 / 1000.0, np.exp(-0.1 / 5.0)
    assert g.shape == (t.size, 2)
    assert g.mean() == pytest.approx(events_per_step * 0.5 / (1 - a), rel=0.1)


def test_opsin_kinetics_rise_and_decay():
    light = light_pulses(T, 1.0, 100.0, 200.0)
    opsin = OPSINS["CsChrimson (attivazione)"]
    g = opsin_conductance(light, 0.1, opsin, g_max=5.0)
    plateau = 5.0 * 1.0 / (1.0 + opsin.half_sat)
    assert g[np.searchsorted(T, 199.0)] == pytest.approx(plateau, rel=0.01)
    after = g[np.searchsorted(T, 200.0 + opsin.tau_off)]
    assert after == pytest.approx(plateau * np.exp(-1.0), rel=0.05)


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_builds(kind):
    spec = {"kind": kind, "targets": ["A", "B"]}
    if kind == "Da file CSV":
        spec.update(samples_t=[0.0, 100.0, 200.0], samples_v=[0.0, 20.0, 0.0])
    out = build_channel(spec, T, 0.1, seed=0)
    assert out.error is None and out.trace.shape == T.shape and np.all(np.isfinite(out.trace))
    assert set(DEFAULTS[kind]) <= set({**DEFAULTS[kind], **spec})
    if kind in ("Input sinaptico di Poisson", "Optogenetica"):
        assert out.unit == "nS" and out.e_rev is not None and out.trace.min() >= 0


def test_invalid_expression_is_reported_not_raised():
    out = build_channel({"kind": "Equazione custom", "targets": ["A"], "expression": "import os"}, T, 0.1, 0)
    assert out.error and not out.trace.any()


# ----------------------------------------------------------------------------- manipolazioni

def test_temperature_protocol_and_thermogenetics():
    spec = {"temp_baseline": 22.0, "temp_target": 32.0, "temp_on": 100.0, "temp_off": 300.0, "temp_ramp": 20.0}
    temp = temperature_trace(T, spec)
    assert temp[0] == 22.0 and temp[np.searchsorted(T, 200.0)] == 32.0 and temp[-1] == pytest.approx(22.0)
    assert trpa1_activation(22.0, 27.0) < 0.01 < 0.99 < trpa1_activation(32.0, 27.0)
    assert shibire_block(22.0, 29.5) < 0.01 < 0.99 < shibire_block(32.0, 29.5)


def _experiment(**kwargs):
    circuit = PRESETS["Cascata feed-forward"]()
    base = dict(neurons=circuit.neurons, weights=circuit.weights, stimuli=circuit.default_stimuli)
    return Experiment(**{**base, **kwargs})


def test_baseline_cascade_propagates():
    result, _ = run_experiment(_experiment())
    assert (result.spike_counts > 0).all()


@pytest.mark.parametrize("manipulation", [
    {"tnt": ["N2"]}, {"ablated": ["N2"]}, {"kir": ["N2"], "kir_g": 10.0},
    {"mec": 1.0}, {"shibire": ["N2"], "temp_on": 0.0, "temp_ramp": 1.0},
])
def test_manipulations_stop_propagation_after_n2(manipulation):
    result, _ = run_experiment(_experiment(manipulations=manipulation))
    assert result.spike_counts[0] > 0 and result.spike_counts[3:].sum() == 0


def test_trpa1_activates_unstimulated_neuron():
    exp = _experiment(stimuli=[], manipulations={"trpa1": ["N3"], "trpa1_gmax": 5.0, "temp_on": 100.0})
    result, inputs = run_experiment(exp)
    assert inputs.temperature is not None
    assert result.spike_counts[2] > 0 and result.spike_counts[:2].sum() == 0
    assert result.spike_times[2][0] > 100.0


def test_picrotoxin_releases_inhibition():
    circuit = PRESETS["Feedback inibitorio (lobo antennale)"]()
    exp = Experiment(neurons=circuit.neurons, weights=circuit.weights, stimuli=circuit.default_stimuli)
    control, _ = run_experiment(exp)
    ptx, _ = run_experiment(Experiment(**{**exp.__dict__, "manipulations": {"ptx": 1.0}}))
    pn_b = circuit.neurons.index("PN_B")
    assert ptx.spike_counts[pn_b] > control.spike_counts[pn_b]


# ----------------------------------------------------------------------------- esperimenti

def test_trials_change_noise_but_are_reproducible():
    exp = _experiment(sim=SimParams(noise_sigma=5.0))
    a0, a1 = prepare(exp, 0).drive.i_ext, prepare(exp, 1).drive.i_ext
    assert not np.allclose(a0, a1)
    np.testing.assert_array_equal(a0, prepare(exp, 0).drive.i_ext)


def test_json_roundtrip_reproduces_result():
    exp = _experiment(stp=STPParams(enabled=True), stp_targets=["N1"], manipulations={"ptx": 0.5},
                      gap=np.eye(5, k=1) + np.eye(5, k=-1))
    again = from_json(to_json(exp, widgets={"ignored": True}))
    r1, _ = run_experiment(exp)
    r2, _ = run_experiment(again)
    np.testing.assert_array_equal(r1.v, r2.v)
    assert again.stp.enabled and again.stp_targets == ["N1"] and again.manipulations == {"ptx": 0.5}
