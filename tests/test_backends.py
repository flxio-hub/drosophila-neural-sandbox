"""Il backend Numba deve riprodurre il backend NumPy di riferimento, anche con tutte le funzioni attive."""

import numpy as np
import pytest

from sandbox.config import STPParams, SynapseParams
from sandbox.network import PRESETS
from sandbox.simulation import simulate
from sandbox.simulation.fast import AVAILABLE
from sandbox.stimuli import gaussian_noise, time_vector

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="Numba non installato")


def _inputs(circuit, t, gain, extras):
    n = circuit.n
    i_ext = np.column_stack([gain * gaussian_noise(t, 30.0, 10.0, 5.0, seed=i) for i in range(n)])
    kwargs = {}
    if extras:
        rng = np.random.default_rng(3)
        gap = rng.uniform(0.0, 1.5, (n, n)) * (rng.random((n, n)) < 0.3)
        gap = np.maximum(gap, gap.T)
        np.fill_diagonal(gap, 0.0)
        g_ext = np.zeros((t.size, n))
        g_ext[(t > 50) & (t < 120), 0] = 2.0          # conduttanza "optogenetica" eccitatoria sul neurone 0
        ge_ext = g_ext * 0.0
        g_ext[:, -1] += 0.5                           # "Kir2.1" sull'ultimo neurone
        ge_ext[:, -1] += 0.5 * -80.0
        gain_rel = np.ones((t.size, n))
        gain_rel[t > 150, 1] = 0.0                    # blocco sinaptico del neurone 1 (shibire^ts)
        kwargs = dict(gap=gap, g_ext=g_ext, ge_ext=ge_ext, release_gain=gain_rel,
                      stp=STPParams(enabled=True, U=0.4, tau_rec=150.0, tau_fac=50.0),
                      stp_mask=np.arange(n) % 2 == 0)
    return i_ext, kwargs


@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("model, dt", [("lif", 0.1), ("hh", 0.025)])
@pytest.mark.parametrize("preset", ["Feedback inibitorio (lobo antennale)", "Rete ricorrente E/I casuale"])
def test_numba_matches_numpy(preset, model, dt, extras):
    circuit = PRESETS[preset]()
    t = time_vector(200.0, dt)
    i_ext, kwargs = _inputs(circuit, t, 1.0 if model == "lif" else 3.0, extras)
    common = dict(model=model, syn=SynapseParams(delay=2.0), **kwargs)
    ref = simulate(circuit.weights, i_ext, circuit.neurons, dt, backend="numpy", **common)
    jit = simulate(circuit.weights, i_ext, circuit.neurons, dt, backend="numba", **common)
    assert ref.spike_counts.sum() > 0
    for a, b in zip(ref.spike_times, jit.spike_times):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(ref.v, jit.v, atol=1e-6)
    np.testing.assert_allclose(ref.g_exc, jit.g_exc, atol=1e-5)
    np.testing.assert_allclose(ref.g_inh, jit.g_inh, atol=1e-5)
