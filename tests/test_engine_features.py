"""Verifiche analitiche delle funzioni del motore: STP, gap junction, conduttanze esterne, blocco del rilascio."""

import numpy as np
import pytest

from sandbox.config import LIFParams, STPParams, SynapseParams
from sandbox.models.synapses import ShortTermPlasticity
from sandbox.simulation import simulate
from sandbox.stimuli import square_pulse, time_vector

PASSIVE = LIFParams(v_thresh=1e6, theta_jump=0.0)  # nessuno spike: membrana puramente passiva


def test_stp_depression_matches_tsodyks_markram():
    U, tau_rec, dt, interval = 0.5, 200.0, 0.1, 50.0
    stp = ShortTermPlasticity(STPParams(enabled=True, U=U, tau_rec=tau_rec), np.array([True]), dt)
    spike, silent = np.array([True]), np.array([False])
    first = stp.efficacy(spike)[0]
    for _ in range(int(round(interval / dt)) - 1):
        stp.efficacy(silent)
    second = stp.efficacy(spike)[0]
    assert first == pytest.approx(1.0)
    assert second == pytest.approx(1.0 - U * np.exp(-interval / tau_rec), rel=1e-3)


def test_stp_facilitation_increases_efficacy():
    stp = ShortTermPlasticity(STPParams(enabled=True, U=0.1, tau_rec=5.0, tau_fac=300.0), np.array([True]), 0.1)
    spike, silent = np.array([True]), np.array([False])
    first = stp.efficacy(spike)[0]
    for _ in range(199):
        stp.efficacy(silent)
    assert stp.efficacy(spike)[0] > 1.5 * first


def _pair_with_pulses(stp, weight=1.0, isi=40.0):
    """pre scarica con due brevi impulsi separati da ``isi``; restituisce i salti di g_exc su post."""
    dt = 0.1
    t = time_vector(200.0, dt)
    i_ext = np.zeros((t.size, 2))
    for onset in (20.0, 20.0 + isi):
        i_ext[:, 0] += square_pulse(t, 400.0, onset, onset + 1.0)
    syn = SynapseParams(delay=1.0)
    r = simulate(np.array([[0.0, weight], [0.0, 0.0]]), i_ext, ["pre", "post"], dt,
                 lif=LIFParams(theta_jump=0.0, t_ref=5.0), syn=syn, stp=stp)
    g = r.g_exc[:, 1].astype(float)
    jumps = g[1:] - g[:-1] * np.exp(-dt / syn.tau_exc)
    assert r.spike_counts[0] == 2
    return jumps[jumps > 1e-3]


def test_paired_pulse_depression_in_network():
    jumps = _pair_with_pulses(STPParams(enabled=True, U=0.5, tau_rec=200.0))
    assert jumps[0] == pytest.approx(1.0, rel=1e-3)
    assert jumps[1] / jumps[0] == pytest.approx(1 - 0.5 * np.exp(-40.0 / 200.0), rel=0.02)


def test_gap_junction_coupling_coefficient():
    """Stato stazionario di due cellule passive accoppiate: ΔV_B/ΔV_A = G / (G + g_L)."""
    dt, g_gap = 0.1, 2.0
    t = time_vector(400.0, dt)
    i_ext = np.zeros((t.size, 2))
    i_ext[:, 0] = 10.0
    gap = np.array([[0.0, g_gap], [g_gap, 0.0]])
    r = simulate(np.zeros((2, 2)), i_ext, ["A", "B"], dt, lif=PASSIVE, gap=gap)
    dv_a, dv_b = r.v[-1] - PASSIVE.v_rest
    g_l = 1.0 / PASSIVE.r_m
    assert dv_b / dv_a == pytest.approx(g_gap / (g_gap + g_l), rel=1e-3)


def test_lif_spikelets_cross_gap_junctions():
    dt = 0.1
    t = time_vector(200.0, dt)
    i_ext = np.zeros((t.size, 2))
    i_ext[:, 0] = square_pulse(t, 40.0, 20.0, 180.0)
    gap = np.array([[0.0, 1.0], [1.0, 0.0]])
    r = simulate(np.zeros((2, 2)), i_ext, ["A", "B"], dt, gap=gap)
    assert r.spike_counts[0] > 0
    # lo spikelet è un salto netto di V_B nel passo successivo allo spike di A
    k = int(round(r.spike_times[0][0] / dt))
    assert r.v[k + 1, 1] - r.v[k, 1] > 2.0


def test_external_conductance_sets_equilibrium_potential():
    dt, g, e_rev = 0.1, 3.0, -80.0
    t = time_vector(300.0, dt)
    r = simulate(np.zeros((1, 1)), np.zeros((t.size, 1)), ["A"], dt, lif=PASSIVE,
                 g_ext=np.full((t.size, 1), g), ge_ext=np.full((t.size, 1), g * e_rev))
    g_l = 1.0 / PASSIVE.r_m
    expected = (g_l * PASSIVE.v_rest + g * e_rev) / (g_l + g)
    assert r.v[-1, 0] == pytest.approx(expected, abs=1e-6)


def test_zero_release_gain_blocks_transmission():
    dt = 0.1
    t = time_vector(200.0, dt)
    i_ext = np.zeros((t.size, 2))
    i_ext[:, 0] = 50.0
    gain = np.ones((t.size, 2))
    gain[:, 0] = 0.0
    r = simulate(np.array([[0.0, 5.0], [0.0, 0.0]]), i_ext, ["pre", "post"], dt, release_gain=gain)
    assert r.spike_counts[0] > 0 and r.g_exc[:, 1].max() == 0.0 and r.spike_counts[1] == 0
