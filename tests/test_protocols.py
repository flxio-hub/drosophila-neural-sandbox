"""Protocolli sperimentali e analisi avanzate, verificati contro risultati noti."""

import numpy as np
import pytest

from sandbox.config import HHParams, LIFParams, SimParams, STPParams, SynapseParams
from sandbox.experiment import Experiment, run_experiment
from sandbox.network import PRESETS
from sandbox.protocols import (current_value, fi_curve, hh_voltage_clamp, impedance_profile, paired_pulse,
                               parameter_sweep, repeated_trials, stp_efficacies, sweepable_parameters)
from sandbox.simulation import (cross_correlogram, instantaneous_rate, isi_histogram, power_spectrum,
                                spike_reliability, spike_triggered_average, synaptic_currents)
from sandbox.stimuli import time_vector


def _cascade(**kwargs):
    c = PRESETS["Cascata feed-forward"]()
    return Experiment(**{"neurons": c.neurons, "weights": c.weights, "stimuli": c.default_stimuli, **kwargs})


# ----------------------------------------------------------------------------- analisi

def test_isi_histogram_of_regular_train():
    centers, counts = isi_histogram(np.arange(0.0, 500.0, 20.0), bin_ms=2.0)
    assert counts.sum() == 24 and centers[np.argmax(counts)] == pytest.approx(21.0)


def test_instantaneous_rate_integrates_to_spike_count():
    t = time_vector(1000.0, 0.1)
    spikes = np.array([200.0, 400.0, 600.0])
    rate = instantaneous_rate(spikes, t, sigma_ms=10.0)
    assert rate.sum() * 0.1 / 1000.0 == pytest.approx(3.0, rel=1e-3)


def test_cross_correlogram_peaks_at_the_lag():
    ref = np.arange(10.0, 1000.0, 37.0)
    lags, rate = cross_correlogram(ref, ref + 4.0, window_ms=20.0, bin_ms=1.0)
    assert lags[np.argmax(rate)] == pytest.approx(4.5)


def test_sta_recovers_the_triggering_stimulus():
    dt = 0.1
    stim = np.zeros(10000)
    stim[np.arange(500, 9500, 500)] = 1.0          # impulso sempre 3 ms prima dello "spike"
    spikes = (np.arange(500, 9500, 500) + 30) * dt
    lags, sta, used = spike_triggered_average(stim, dt, spikes, window_ms=10.0)
    assert used == 18 and lags[np.argmax(sta)] == pytest.approx(-3.0)


def test_power_spectrum_finds_oscillation():
    t = time_vector(4000.0, 1.0)
    freqs, psd = power_spectrum(np.sin(2 * np.pi * 12.0 * t / 1000.0), dt_ms=1.0)
    assert freqs[np.argmax(psd)] == pytest.approx(12.0, abs=1.0)


def test_synaptic_currents_sign():
    result, _ = run_experiment(_cascade())
    i_exc, i_inh = synaptic_currents(result)
    assert i_exc[:, 1].max() > 0 and np.all(i_inh == 0)


def test_reliability_extremes():
    t = time_vector(500.0, 0.1)
    same = [np.array([100.0, 200.0, 300.0])] * 5
    rng = np.random.default_rng(0)
    random = [np.sort(rng.uniform(0, 500, 3)) for _ in range(5)]
    assert spike_reliability(same, t) == pytest.approx(1.0)
    assert spike_reliability(random, t) < 0.3


# ----------------------------------------------------------------------------- protocolli

def test_fi_curve_matches_lif_theory():
    p = LIFParams(theta_jump=0.0, t_ref=0.0)
    curve = fi_curve("lif", p, HHParams(), np.arange(0.0, 61.0, 5.0), step_ms=1000.0)
    assert curve.rheobase == pytest.approx(20.0)  # griglia a passi di 5 pA, reobase teorica 15 pA (esclusa: V∞ = soglia)
    row = curve.table.set_index("Corrente (pA)").loc[40.0]
    v_inf = p.v_rest + p.r_m * 40.0
    theory = 1000.0 / (p.tau_m * np.log((v_inf - p.v_reset) / (v_inf - p.v_thresh)))
    assert row["Frequenza a regime (Hz)"] == pytest.approx(theory, rel=0.05)


def test_fi_curve_shows_adaptation():
    curve = fi_curve("lif", LIFParams(theta_jump=3.0, tau_theta=200.0), HHParams(), [60.0], step_ms=500.0)
    assert curve.table["Indice di adattamento"].iloc[0] > 0.3


def test_parameter_sweep_weight_threshold():
    exp = _cascade()
    table = parameter_sweep(exp, "Moltiplicatore pesi chimici (×)", [0.0, 1.0], "Numero di spike")
    n5 = table[table["Neurone"] == "N5"].set_index("Moltiplicatore pesi chimici (×)")["Numero di spike"]
    assert n5[0.0] == 0 and n5[1.0] > 0


def test_parameter_sweep_2d_and_stimulus_parameters():
    exp = _cascade()
    params = sweepable_parameters(exp)
    assert "S1 (Onda quadra): amplitude" in params and "W: N1 → N2 (nS)" in params
    assert current_value(exp, "S1 (Onda quadra): amplitude") == 30.0
    assert current_value(exp, "W: N1 → N2 (nS)") == 3.0
    assert current_value(exp, "LIF: τ_m (ms)") == 20.0
    table = parameter_sweep(exp, "S1 (Onda quadra): amplitude", [10.0, 40.0], "Frequenza media (Hz)",
                            param2="W: N1 → N2 (nS)", values2=[0.0, 3.0])
    assert len(table) == 4 * exp.weights.shape[0]
    n1 = table[(table["Neurone"] == "N1")]
    assert n1.groupby("S1 (Onda quadra): amplitude")["Frequenza media (Hz)"].mean().is_monotonic_increasing


def test_repeated_trials_variability():
    exp = _cascade(sim=SimParams(noise_sigma=6.0))
    noisy = repeated_trials(exp, 6).statistics()
    frozen_det = repeated_trials(_cascade(), 3).statistics()
    assert (frozen_det["Fattore di Fano"].fillna(0) == 0).all()
    assert np.allclose(frozen_det["Affidabilità"], 1.0)
    assert (noisy["Affidabilità"] < 0.9).all() and (noisy["Fattore di Fano"] > 0).all()


def test_stp_efficacies_closed_form_matches_theory():
    stp = STPParams(enabled=True, U=0.5, tau_rec=100.0)
    eff = stp_efficacies([100, 600], stp, dt=0.1)   # spike a 10 ms e 60 ms
    assert eff[0] == pytest.approx(1.0) and eff[1] == pytest.approx(1 - 0.5 * np.exp(-50.0 / 100.0), rel=1e-3)
    assert np.allclose(stp_efficacies([10, 20, 30], STPParams(enabled=False), 0.1), 1.0)


def test_paired_pulse_depression_recovers_with_interval():
    test = paired_pulse(1.0, STPParams(enabled=True, U=0.6, tau_rec=200.0), SynapseParams(), LIFParams(),
                        isis=[20.0, 100.0, 800.0], train_hz=20.0, n_train=5)
    ppr = test.pairs["PPR (efficacia)"].to_numpy()
    assert ppr[0] < ppr[1] < ppr[2] < 1.0 and ppr[2] > 0.95
    assert test.train["Efficacia (norm.)"].is_monotonic_decreasing
    assert (test.pairs["PSP 1 (mV)"] > 0).all()


def test_lif_impedance_is_low_pass():
    lif = LIFParams(v_thresh=1e6)
    prof = impedance_profile("lif", lif, HHParams(), amplitude=5.0, f0=1.0, f1=60.0, duration=4000.0)
    for f in (2.0, 8.0, 30.0):
        i = np.argmin(np.abs(prof.freqs - f))
        theory = lif.r_m / np.sqrt(1 + (2 * np.pi * f * lif.tau_m / 1000.0) ** 2)
        assert prof.magnitude[i] == pytest.approx(theory, rel=0.1)
    assert prof.spikes == 0 and prof.resonance_hz < 3.0


def test_hh_voltage_clamp_currents():
    vc = hh_voltage_clamp(HHParams(), commands=[-40.0, 0.0, 40.0, 60.0])
    iv = vc.iv.set_index("V comando (mV)")
    assert iv.loc[0.0, "I_Na di picco (pA)"] < 0                    # corrente di Na⁺ entrante
    assert iv.loc[60.0, "I_Na di picco (pA)"] > 0                   # oltre E_Na = 50 mV si inverte
    assert iv.loc[40.0, "I_K a regime (pA)"] > iv.loc[0.0, "I_K a regime (pA)"] > 0
