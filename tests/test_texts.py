"""Testi centralizzati: ogni controllo ha un tooltip, ogni riferimento a texts.py esiste."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from sandbox.data import store
from sandbox.network import PRESETS
from sandbox.stimuli import KINDS, MODALITIES, OPSINS, PROFILES
from sandbox.ui import state, texts
from sandbox.ui.controls import CONNECTOME, signature
from sandbox.ui.pages import DATA
from sandbox.ui.stimulus_panel import MODALITY_CODES

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
SIG = signature(tuple(f"N{i}" for i in range(1, 6)))
UI_SOURCES = [*sorted((ROOT / "sandbox" / "ui").rglob("*.py")), ROOT / "app.py"]


def key(name: str) -> str:
    return f"g0·{name}"


def test_references_to_texts_exist():
    source = "\n".join(p.read_text(encoding="utf-8") for p in UI_SOURCES)
    for name in re.findall(r'TIPS\["(\w+)"\]', source):
        assert name in texts.TIPS, name
    for name in re.findall(r'(?:explain|section)\([^)]*?"(\w+)"\)', source):
        assert name in texts.EXPLAIN, name
    assert set(texts.KIND_HELP) == set(KINDS)
    assert set(texts.PROTOCOL_EXPLAIN) == set(texts.PROTOCOLS)
    assert all(v in texts.EXPLAIN for v in texts.PROTOCOL_EXPLAIN.values())


def test_explanations_have_body_and_are_not_empty():
    for name, (body, example) in texts.EXPLAIN.items():
        assert len(body) > 40, name
    for name, tip in texts.TIPS.items():
        assert tip.strip() and len(tip) < 320, name  # tooltip brevi


@pytest.mark.parametrize("name, keys", [
    ("3fa2c01b9e_ch1_se_orn_amp", ["se_orn_amp", "se_amp", "amp"]),
    ("3fa2c01b9e_ch0_op_csc_int", ["op_csc_int", "op_int", "int"]),
    ("3fa2c01b9e_ch2_sq_ton", ["sq_ton", "ton"]),
    ("3fa2c01b9e_sw_3_lo", ["sw_lo"]),
    ("duration_hh", ["duration_hh", "duration"]),
    ("lif_tau_m", ["lif_tau_m"]),
])
def test_tip_keys(name, keys):
    assert state.tip_keys(name) == keys


def test_every_widget_has_a_tooltip(tmp_path, monkeypatch):
    """Percorre l'interfaccia (tutti i circuiti, stimoli, modalità, manipolazioni, protocolli e la pagina Dati)
    e controlla che ogni widget creato con gli helper di ``state`` abbia un testo in ``texts.TIPS``."""
    monkeypatch.setenv(store.ENV_VAR, str(tmp_path))
    store.save_stimulus("s.csv", b"t,v\n0,0\n10,1\n")
    (tmp_path / "e.csv").write_text("pre_root_id,post_root_id,neuropil,syn_count,nt_type\n1,2,AL_L,9,ACH\n")
    (tmp_path / "n.csv").write_text("root_id,super_class,class,sub_class,cell_type,side,flow,nt_type\n"
                                    "1,a,b,c,d,left,x,ACH\n2,a,b,c,e,left,x,GABA\n")
    for name in ("e.csv", "n.csv"):
        store.import_table(tmp_path / name, name)

    names: set[str] = set()
    at = AppTest.from_file(APP, default_timeout=240)

    def run(element=None):
        (element.run() if element is not None else at.run())
        assert not at.exception, at.exception
        names.update(at.session_state[state.REGISTRY])

    run()
    kind = key(f"{SIG}_ch0_kind")
    for k in KINDS:
        run(at.selectbox(key=kind).select(k))
    run(at.selectbox(key=kind).select("Sensoriale"))
    for modality in MODALITIES:
        run(at.selectbox(key=key(f"{SIG}_ch0_se_mod")).select(modality))
        for profile in PROFILES:
            run(at.selectbox(key=key(f"{SIG}_ch0_se_{MODALITY_CODES[modality]}_prof")).select(profile))
    run(at.selectbox(key=kind).select("Optogenetica"))
    for opsin in OPSINS:
        run(at.selectbox(key=key(f"{SIG}_ch0_op_opsin")).select(opsin))
    run(at.multiselect(key=key(f"{SIG}_man_kir")).set_value(["N1"]))
    run(at.multiselect(key=key(f"{SIG}_man_trpa1")).set_value(["N1"]))
    run(at.sidebar.toggle(key=key("stp_enabled")).set_value(True))
    for protocol in texts.PROTOCOLS:
        run(at.radio(key=key("proto_choice")).set_value(protocol))
        button = {"Curva f-I": "proto_fi_run", "Trial ripetuti": "proto_tr_run"}.get(protocol)
        if button:
            run(at.button(key=button).click())
        if protocol == "Esplorazione parametrica":
            run(at.number_input(key=key(f"{SIG}_sw_0_n")).set_value(2))
            run(at.button(key="proto_sw_run").click())
            run(at.checkbox(key=key(f"{SIG}_sw_2d")).check())
            run(at.button(key="proto_sw_run").click())
    run(at.sidebar.radio(key=key("model")).set_value("Hodgkin-Huxley"))
    for preset in [*PRESETS, CONNECTOME]:
        run(at.sidebar.selectbox(key=key("preset")).select(preset))
    for source in ("File CSV", "FlyWire live (CAVEclient)", "Hemibrain live (neuPrint)"):
        run(at.sidebar.radio(key=key("cx_source")).set_value(source))
    run(at.switch_page(DATA))

    missing = sorted(n for n in names if not n.endswith("_samples") and state.tip(n) is None)
    assert not missing, f"Widget senza tooltip in texts.TIPS: {missing}"
    assert {"xp_super_class", "xp_edges", "stim_preview", "se_orn_prof"} <= {k for n in names
                                                                             for k in state.tip_keys(n)}
