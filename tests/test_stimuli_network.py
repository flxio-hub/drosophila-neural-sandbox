from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sandbox.network import PRESETS, edges_to_circuit, load_edge_csv
from sandbox.stimuli import MODALITIES, gaussian_noise, ramp, sine, square_pulse, time_vector, transduce
from sandbox.stimuli.sensory import profile_looming, profile_pulse

T = time_vector(1000.0, 0.1)
EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example_mushroom_body.csv"


def test_square_pulse_train_frequency():
    i = square_pulse(T, 10.0, 0.0, 1000.0, frequency_hz=20.0, pulse_width=5.0)
    onsets = np.flatnonzero(np.diff(i) > 0)
    assert len(onsets) + 1 == 20  # il primo impulso inizia a t = 0


def test_ramp_and_sine_shapes():
    r = ramp(T, 0.0, 100.0, 100.0, 600.0)
    assert r[T < 100].max() == 0 and r[np.searchsorted(T, 350.0)] == pytest.approx(50.0, abs=0.1)
    s = sine(T, 10.0, 10.0)
    assert s.max() == pytest.approx(10.0, abs=0.01) and s.mean() == pytest.approx(0.0, abs=0.1)


def test_ou_noise_has_requested_statistics():
    x = gaussian_noise(time_vector(20000.0, 0.1), 5.0, 3.0, tau_corr=5.0, seed=1)
    assert x.mean() == pytest.approx(5.0, abs=0.3) and x.std() == pytest.approx(3.0, rel=0.1)


def test_orn_response_is_phasic_tonic_with_offset_undershoot():
    p = MODALITIES["Neurone olfattivo (ORN)"]["params"]
    current = transduce(profile_pulse(T, 0.6, 100.0, 600.0), 0.1, p)
    peak = current[(T > 100) & (T < 300)].max()
    plateau = current[(T > 550) & (T < 600)].mean()
    assert peak > 1.3 * plateau > 0
    assert current[(T > 600) & (T < 800)].min() < 0  # "off-inhibition" dopo lo stimolo


def test_looming_angle_grows_until_collision():
    theta = profile_looming(T, 500.0, 40.0)
    before = theta[T < 500]
    assert np.all(np.diff(before) >= 0) and theta[T >= 500].min() == 1.0


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_consistent(name):
    circuit = PRESETS[name]()
    assert circuit.weights.shape == (circuit.n, circuit.n)
    for stim in circuit.default_stimuli:
        assert set(stim["targets"]) <= set(circuit.neurons)


def test_random_network_respects_dale_law():
    circuit = PRESETS["Rete ricorrente E/I casuale"]()
    for row in circuit.weights:
        assert (row >= 0).all() or (row <= 0).all()


def test_edges_are_aggregated_and_signed():
    edges = pd.DataFrame({
        "pre_root_id": [1, 1, 2], "post_root_id": [2, 2, 1],
        "syn_count": [4, 6, 10], "nt_type": ["ACH", "ACH", "GABA"],
    })
    circuit, _ = edges_to_circuit(edges, syn_to_nS=0.1)
    assert circuit.neurons == ["1", "2"]
    np.testing.assert_allclose(circuit.weights, [[0.0, 1.0], [-1.0, 0.0]])


def test_injection_supports_shared_and_independent_channels():
    from sandbox.experiment import Experiment, prepare

    noise = {"kind": "Rumore gaussiano", "mean": 0.0, "sigma": 5.0, "tau_corr": 0.0}
    exp = Experiment(neurons=["A", "B", "C"], weights=np.zeros((3, 3)), stim_gain=2.0, stimuli=[
        {"kind": "Onda quadra", "targets": ["A", "B"], "amplitude": 10.0, "t_on": 0.0, "t_off": 1e9},
        {**noise, "targets": ["B", "C"], "independent": False},
        {**noise, "targets": ["A", "C"], "independent": True},
    ])
    i_ext = prepare(exp).drive.i_ext
    shared = prepare(Experiment(neurons=exp.neurons, weights=exp.weights, stimuli=[exp.stimuli[1]])).drive.i_ext
    np.testing.assert_allclose(shared[:, 1], shared[:, 2])            # stesso rumore sui due bersagli
    indep = prepare(Experiment(neurons=exp.neurons, weights=exp.weights, stimuli=[exp.stimuli[2]])).drive.i_ext
    assert not np.allclose(indep[:, 0], indep[:, 2])                  # realizzazioni indipendenti
    assert i_ext[:, 0].mean() == pytest.approx(20.0, abs=1.0)          # 10 pA × guadagno 2


def test_example_connectome_loads():
    circuit, warnings = load_edge_csv(EXAMPLE)
    assert "APL" in circuit.neurons and warnings == []
    apl = circuit.neurons.index("APL")
    assert (circuit.weights[apl] <= 0).all() and circuit.weights[apl].min() < 0
    assert circuit.default_stimuli[0]["targets"] == ["PN_DA1", "PN_VA1d"]
