"""Simulazione a blocchi (anello chiuso): lo Stepper e il Transducer devono dare esattamente lo stesso
risultato di simulate() e transduce() chiamati una volta sola sull'intero segnale."""

import numpy as np
import pytest

from sandbox.config import STPParams, SynapseParams
from sandbox.network import PRESETS
from sandbox.simulation import Stepper, simulate
from sandbox.simulation.fast import AVAILABLE
from sandbox.stimuli import MODALITIES, Transducer, gaussian_noise, time_vector, transduce
from sandbox.stimuli.sensory import profile_looming

BACKENDS = ["numpy", pytest.param("numba", marks=pytest.mark.skipif(not AVAILABLE, reason="Numba non installato"))]


def _chunks(total):
    """Blocchi di lunghezza irregolare (1, 37, 250, 3 passi, poi il resto)."""
    edges = [0, 1, 38, 288, 291, total]
    return list(zip(edges[:-1], edges[1:]))


def _setup(model, dt):
    circuit = PRESETS["Sensorio-motorio: fuga della Giant Fiber"]()  # ha anche le gap junction
    t = time_vector(200.0, dt)
    n = circuit.n
    gain = 1.0 if model == "lif" else 3.0
    i_ext = np.column_stack([gain * gaussian_noise(t, 30.0, 10.0, 5.0, seed=i) for i in range(n)])
    g_ext = np.zeros((t.size, n))
    g_ext[(t > 50) & (t < 120), 0] = 2.0      # conduttanza eccitatoria (E = 0 mV) su LC4
    release_gain = np.ones((t.size, n))
    release_gain[t > 150, 2] = 0.0            # rilascio della GF bloccato a metà
    inputs = dict(g_ext=g_ext, ge_ext=np.zeros_like(g_ext), release_gain=release_gain)
    params = dict(model=model, gap=circuit.gap, syn=SynapseParams(delay=2.0),
                  stp=STPParams(enabled=True, U=0.4, tau_rec=150.0, tau_fac=50.0), stp_mask=np.arange(n) % 2 == 0)
    return circuit, i_ext, inputs, params


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model, dt", [("lif", 0.1), ("hh", 0.025)])
def test_chunks_match_single_run(backend, model, dt):
    circuit, i_ext, inputs, params = _setup(model, dt)
    ref = simulate(circuit.weights, i_ext, circuit.neurons, dt, backend=backend, **params, **inputs)
    stepper = Stepper(circuit.weights, circuit.neurons, dt, backend=backend, **params)
    parts = [stepper.advance(i_ext[a:b], **{key: x[a:b] for key, x in inputs.items()})
             for a, b in _chunks(len(i_ext))]

    assert ref.spike_counts.sum() > 0
    assert stepper.time == pytest.approx(ref.duration)
    np.testing.assert_array_equal(np.concatenate([p.t for p in parts]), ref.t)
    np.testing.assert_array_equal(np.concatenate([p.v for p in parts]), ref.v)
    np.testing.assert_array_equal(np.concatenate([p.g_exc for p in parts]), ref.g_exc)
    np.testing.assert_array_equal(np.concatenate([p.g_inh for p in parts]), ref.g_inh)
    for i in range(circuit.n):
        np.testing.assert_array_equal(np.concatenate([p.spike_times[i] for p in parts]), ref.spike_times[i])


def test_stepper_rejects_wrong_input_size():
    circuit = PRESETS["Sensorio-motorio: fuga della Giant Fiber"]()
    stepper = Stepper(circuit.weights, circuit.neurons, 0.1)
    with pytest.raises(ValueError):
        stepper.advance(np.zeros((10, circuit.n + 1)))
    with pytest.raises(ValueError):
        stepper.advance(np.zeros((10, circuit.n)), g_ext=np.zeros((10, circuit.n)))


@pytest.mark.parametrize("modality", list(MODALITIES))
def test_transducer_chunks_match_single_call(modality):
    dt = 0.1
    t = time_vector(500.0, dt)
    signal = profile_looming(t, 350.0, 40.0) + 0.3 * ((t > 100) & (t < 200))
    p = MODALITIES[modality]["params"]
    transducer = Transducer(dt, p)
    parts = np.concatenate([transducer(signal[a:b]) for a, b in _chunks(t.size)])
    np.testing.assert_allclose(parts, transduce(signal, dt, p), rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_set_weights_between_blocks(backend):
    # pesi invariati: identico a non chiamarla; pesi nuovi: valgono dal blocco dopo, uguali nei due backend
    circuit, i_ext, inputs, params = _setup("lif", 0.1)
    ref = Stepper(circuit.weights, circuit.neurons, 0.1, backend=backend, **params)
    same = Stepper(circuit.weights, circuit.neurons, 0.1, backend=backend, **params)
    half = len(i_ext) // 2
    first = {key: x[:half] for key, x in inputs.items()}
    second = {key: x[half:] for key, x in inputs.items()}
    ref.advance(i_ext[:half], **first)
    same.advance(i_ext[:half], **first)
    same.set_weights(circuit.weights.copy())
    tail = ref.advance(i_ext[half:], **second)
    np.testing.assert_array_equal(same.advance(i_ext[half:], **second).v, tail.v)
    ref_g = tail.g_exc[300:, 2].max()

    silent = circuit.weights.copy()
    silent[:, 2] = 0.0  # niente sinapsi sul neurone 2 (la GF)
    results = []
    for name in ("numpy", backend):
        stepper = Stepper(circuit.weights, circuit.neurons, 0.1, backend=name, **params)
        stepper.advance(i_ext[:half], **first)
        stepper.set_weights(silent)
        results.append(stepper.advance(i_ext[half:], **second))
    np.testing.assert_allclose(results[0].v, results[1].v, atol=1e-9)
    assert results[0].g_exc[300:, 2].max() < 1e-2 < ref_g  # dopo il ritardo e qualche τ la GF non riceve quasi più nulla
    with pytest.raises(ValueError):
        ref.set_weights(np.zeros((2, 2)))
