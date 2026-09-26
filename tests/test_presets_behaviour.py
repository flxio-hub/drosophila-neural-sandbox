"""I circuiti preimpostati riproducono il fenomeno che dichiarano (e le manipolazioni lo alterano)."""

import numpy as np

from sandbox.experiment import Experiment, run_experiment
from sandbox.network import PRESETS


def _run(name, **kwargs):
    c = PRESETS[name]()
    exp = Experiment(neurons=c.neurons, weights=c.weights, gap=c.gap, stimuli=c.default_stimuli, **kwargs)
    return run_experiment(exp)[0]


def _count(result, name, t0, t1):
    s = result.spike_times[result.neurons.index(name)]
    return int(((s >= t0) & (s < t1)).sum())


def test_t4_is_direction_selective():
    r = _run("Selettività di direzione (T4)")          # barra → a 50 ms, barra ← a 300 ms
    assert _count(r, "T4_dx", 50, 250) >= 2 and _count(r, "T4_dx", 300, 500) == 0
    assert _count(r, "T4_sx", 300, 500) >= 2 and _count(r, "T4_sx", 50, 250) == 0


def test_t4_selectivity_needs_null_side_inhibition():
    r = _run("Selettività di direzione (T4)", manipulations={"ablated": ["CT1_3"]})
    assert _count(r, "T4_dx", 300, 500) > 0            # senza inibizione risponde anche alla direzione nulla


def test_mushroom_body_sparse_coding_depends_on_apl():
    def active_kcs(result, t0, t1):
        return {n for n in result.neurons if n.startswith("KC") and _count(result, n, t0, t1) > 0}

    ctrl = _run("Corpo fungo: codifica sparsa")
    no_apl = _run("Corpo fungo: codifica sparsa", manipulations={"ablated": ["APL"]})
    n_kc = sum(n.startswith("KC") for n in ctrl.neurons)
    a, b = active_kcs(ctrl, 50, 250), active_kcs(ctrl, 280, 480)
    assert 0 < len(a) <= 0.3 * n_kc and 0 < len(b) <= 0.3 * n_kc      # codifica sparsa
    assert len(active_kcs(no_apl, 50, 250)) > len(a)                   # l'APL rende la codifica più sparsa


def test_giant_fiber_gap_junctions_give_short_latency():
    def latency(result):
        gf, ttmn = (result.spike_times[result.neurons.index(n)] for n in ("GF", "TTMn"))
        return ttmn[0] - gf[0] if ttmn.size else np.inf   # nessuna risposta = latenza infinita

    electrical = latency(_run("Sensorio-motorio: fuga della Giant Fiber"))
    blocked = latency(_run("Sensorio-motorio: fuga della Giant Fiber", manipulations={"gap_block": 1.0}))
    assert electrical < 0.5 and blocked > 3.0          # come nei mutanti shakB²: risposta rapida persa


def test_every_preset_produces_activity():
    for name in PRESETS:
        assert _run(name).spike_counts.sum() > 0, name
