"""Validazione fisiologica dei modelli contro risultati analitici e valori di letteratura."""

import numpy as np
import pytest

from sandbox.config import LIFParams, SynapseParams
from sandbox.simulation import firing_rates, simulate
from sandbox.stimuli import square_pulse, time_vector


def _single(current_pA, model="lif", dt=0.1, duration=500.0, **kw):
    t = time_vector(duration, dt)
    i_ext = np.full((t.size, 1), float(current_pA))
    return simulate(np.zeros((1, 1)), i_ext, ["A"], dt, model=model, **kw)


def lif_analytic_rate(i_pA, p: LIFParams):
    """f = 1 / (t_ref + τ ln((V∞ − V_reset)/(V∞ − V_th))) con V∞ = V_rest + R·I."""
    v_inf = p.v_rest + p.r_m * i_pA
    if v_inf <= p.v_thresh:
        return 0.0
    return 1000.0 / (p.t_ref + p.tau_m * np.log((v_inf - p.v_reset) / (v_inf - p.v_thresh)))


@pytest.mark.parametrize("current", [20.0, 30.0, 50.0])
def test_lif_matches_analytic_fi_curve(current):
    p = LIFParams(theta_jump=0.0, t_ref=0.0)
    rate = firing_rates(_single(current, lif=p, duration=2000.0))[0]
    assert rate == pytest.approx(lif_analytic_rate(current, p), rel=0.05)


def test_lif_is_silent_below_rheobase():
    p = LIFParams()
    rheobase = (p.v_thresh - p.v_rest) / p.r_m
    assert _single(rheobase * 0.9, lif=p).spike_counts[0] == 0


def test_threshold_adaptation_lowers_rate():
    plain = firing_rates(_single(40.0, lif=LIFParams(theta_jump=0.0)))[0]
    adaptive = firing_rates(_single(40.0, lif=LIFParams(theta_jump=3.0, tau_theta=200.0)))[0]
    assert adaptive < 0.8 * plain


def test_hodgkin_huxley_repetitive_firing():
    """HH classico: nessuno scarico ripetitivo sotto ~6 µA/cm², ~70 Hz a 10 µA/cm² (area 1000 µm² → ×10 pA)."""
    below = _single(40.0, model="hh", dt=0.025, duration=300.0)
    above = _single(100.0, model="hh", dt=0.025, duration=300.0)
    assert below.spike_counts[0] <= 1
    assert 55 <= firing_rates(above)[0] <= 85
    assert above.v.max() > 20.0  # overshoot del potenziale d'azione


def _pair(weight, delay=1.0):
    dt = 0.1
    t = time_vector(200.0, dt)
    i_ext = np.zeros((t.size, 2))
    i_ext[:, 0] = square_pulse(t, 60.0, 20.0, 180.0)
    w = np.array([[0.0, weight], [0.0, 0.0]])
    return simulate(w, i_ext, ["pre", "post"], dt, syn=SynapseParams(delay=delay))


def test_excitatory_synapse_depolarizes_and_inhibitory_hyperpolarizes():
    rest = LIFParams().v_rest
    exc = _pair(+0.5)
    inh = _pair(-0.5)
    assert exc.v[:, 1].max() > rest + 1.0
    assert inh.v[:, 1].min() < rest - 0.5


def test_synaptic_delay_shifts_postsynaptic_response():
    def onset(delay):
        r = _pair(+0.5, delay=delay)
        first_pre = r.spike_times[0][0]
        rest = LIFParams().v_rest
        idx = np.argmax(r.v[:, 1] > rest + 0.05)
        return r.t[idx] - first_pre

    assert onset(5.0) - onset(1.0) == pytest.approx(4.0, abs=0.25)
