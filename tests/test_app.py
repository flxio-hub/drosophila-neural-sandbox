"""Smoke test dell'interfaccia: esegue app.py in modalità headless e interagisce con i widget principali."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sandbox.network import PRESETS
from sandbox.ui.controls import CONNECTOME, signature

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CASCADE_SIG = signature(tuple(f"N{i}" for i in range(1, 6)))  # firma del circuito di default (5 stadi)


def key(name: str, generation: int = 0) -> str:
    return f"g{generation}·{name}"


def _app(timeout=120) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert not at.exception, at.exception
    return at


def _spikes(at) -> int:
    return int(at.metric[2].value)


@pytest.mark.parametrize("preset", [*PRESETS, CONNECTOME])
def test_app_runs_for_every_preset(preset):
    at = _app()
    at.sidebar.selectbox(key=key("preset")).select(preset).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 5 and _spikes(at) > 0


def test_hodgkin_huxley_mode_runs():
    at = _app()
    at.sidebar.radio(key=key("model")).set_value("Hodgkin-Huxley").run()
    assert not at.exception, at.exception
    assert _spikes(at) > 0


def test_manipulation_changes_result():
    at = _app()
    before = _spikes(at)
    at.multiselect(key=key(f"{CASCADE_SIG}_man_tnt")).set_value(["N2"]).run()
    assert not at.exception, at.exception
    assert _spikes(at) < before


def test_every_stimulus_kind_renders():
    from sandbox.stimuli import KINDS
    at = _app()
    kind_key = key(f"{CASCADE_SIG}_ch0_kind")
    for kind in KINDS:
        at.selectbox(key=kind_key).select(kind).run()
        assert not at.exception, (kind, at.exception)


@pytest.mark.parametrize("protocol, button", [
    ("Curva f-I", "proto_fi_run"), ("Esplorazione parametrica", "proto_sw_run"),
    ("Trial ripetuti", "proto_tr_run"), ("Impedenza (ZAP)", "proto_z_run"),
    ("Paired-pulse (STP)", None), ("Voltage clamp (HH)", None),
])
def test_protocols_run(protocol, button):
    at = _app(timeout=240)
    at.radio(key=key("proto_choice")).set_value(protocol).run()
    assert not at.exception, at.exception
    if button:
        at.button(key=button).click().run()
        assert not at.exception, at.exception
    assert len(at.get("plotly_chart")) > 4  # i grafici del protocollo si aggiungono a quelli della simulazione


def test_control_comparison():
    at = _app()
    next(b for b in at.button if b.label.startswith("📌")).click().run()
    assert not at.exception and at.session_state["control"] is not None
    at.sidebar.slider(key=key("cascade_w")).set_value(1.0).run()
    assert not at.exception, at.exception


def test_import_restores_widgets_and_circuit():
    at = _app()
    at.session_state["_imported_circuit"] = {"name": "Coppia", "description": "", "neurons": ["A", "B"],
                                             "weights": [[0.0, 3.0], [0.0, 0.0]], "gap": [[0.0, 0.0], [0.0, 0.0]]}
    at.session_state["_imported_widgets"] = {"preset": "📥 Circuito importato (JSON)", "model": "Hodgkin-Huxley"}
    at.session_state["_widget_generation"] = 1
    at.run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox(key=key("preset", 1)).value == "📥 Circuito importato (JSON)"
    assert at.sidebar.radio(key=key("model", 1)).value == "Hodgkin-Huxley"
    assert int(at.metric[0].value) == 2
