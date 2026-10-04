"""Pagine dell'app: Dati (caricamento, esploratore, libreria), passaggio del sottocircuito alla Sandbox e
conservazione dei valori dei widget fra le pagine."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from sandbox.data import store
from sandbox.ui.controls import SUBCIRCUIT, signature
from sandbox.ui.pages import DATA, SIMULATION

APP = str(Path(__file__).resolve().parents[1] / "app.py")
CASCADE_SIG = signature(tuple(f"N{i}" for i in range(1, 6)))

CLASSIFICATION = pd.DataFrame({"root_id": [11, 12, 13, 14], "super_class": ["sensory", "sensory", "central", "central"],
                               "class": ["olfactory", "olfactory", "ALPN", "ALLN"],
                               "cell_type": ["ORN_DA1", "ORN_DA1", "DA1_lPN", "LN_1"],
                               "nt_type": ["ACH", "ACH", "ACH", "GABA"]})
CONNECTIONS = pd.DataFrame({"pre_root_id": [11, 12, 11, 14], "post_root_id": [13, 13, 14, 13],
                            "neuropil": ["AL_L"] * 4, "syn_count": [40, 30, 20, 25], "nt_type": ["ACH"] * 3 + ["GABA"]})


def key(name: str, generation: int = 0) -> str:
    return f"g{generation}·{name}"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(store.ENV_VAR, str(tmp_path / "dati"))
    return tmp_path / "dati"


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _go(at: AppTest, page: str) -> AppTest:
    at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def _write_codex(tmp_path) -> None:
    for df, name in ((CLASSIFICATION, "classification.csv"), (CONNECTIONS, "connections.csv")):
        source = tmp_path / name
        df.to_csv(source, index=False)
        store.import_table(source, name)


def test_data_page_renders_when_empty():
    at = _go(_app(), DATA)
    assert any("Nessuna tabella" in i.value for i in at.info)


def test_import_from_path_list_and_delete(tmp_path):
    source = tmp_path / "classification.csv"
    CLASSIFICATION.to_csv(source, index=False)
    at = _go(_app(), DATA)
    at.text_input(key=key("data_path")).input(f'"{source}"').run()  # le virgolette di "Copia come percorso" vanno bene
    at.button(key="data_import_path").click().run()
    assert not at.exception, at.exception
    assert [t.name for t in store.list_tables()] == ["classification.parquet"]
    assert any("classification.parquet" in s.value for s in at.success)
    at.checkbox(key=key("data_confirm")).check().run()
    at.button(key="data_delete").click().run()
    assert not at.exception and store.list_tables() == []


def test_explorer_filters_and_creates_subcircuit(tmp_path):
    _write_codex(tmp_path)
    at = _go(_app(), DATA)
    at.multiselect(key=key("xp_super_class")).set_value(["sensory"]).run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "2"
    at.multiselect(key=key("xp_super_class")).set_value([]).run()
    at.number_input(key=key("xp_min_syn")).set_value(10).run()
    at.button(key="xp_build").click().run()
    assert not at.exception, at.exception
    # la Sandbox si apre con il nuovo circuito
    assert at.sidebar.selectbox(key=key("preset")).value == SUBCIRCUIT
    assert int(at.metric[0].value) == 4
    assert at.metric[1].value == "3 / 1 / 0"  # 3 sinapsi colinergiche, 1 GABAergica (LN → PN)
    assert int(at.metric[2].value) > 0       # gli ORN (sorgenti) sono stimolati di default


def test_widget_values_survive_page_switch():
    at = _app()
    at.sidebar.slider(key=key("cascade_w")).set_value(1.5).run()
    at.toggle(key=key("live")).set_value(True).run()
    _go(at, DATA)
    _go(at, SIMULATION)
    assert at.sidebar.slider(key=key("cascade_w")).value == 1.5


def test_library_stimulus_in_csv_channel():
    store.save_stimulus("gradino.csv", b"t,v\n0,0\n99,0\n100,40\n300,40\n301,0\n500,0\n")
    at = _app()
    at.selectbox(key=key(f"{CASCADE_SIG}_ch0_kind")).select("Da file CSV").run()
    at.selectbox(key=key(f"{CASCADE_SIG}_ch0_csv_lib")).select("gradino.csv").run()
    assert not at.exception, at.exception
    samples = at.session_state["_widget_registry"][f"{CASCADE_SIG}_ch0_csv_samples"]
    assert samples["source"] == "gradino.csv" and max(samples["v"]) == 40
    assert int(at.metric[2].value) > 0
    # la libreria compare anche nella pagina Dati, con anteprima
    _go(at, DATA)
    assert at.selectbox(key=key("stim_preview")).value == "gradino.csv"


def test_welcome_panel_can_be_hidden_and_shown():
    at = _app()
    at.button(key="welcome_hide").click().run()
    assert not at.exception and at.session_state["_welcome_hidden"]
    assert not any(b.key == "welcome_hide" for b in at.button)
    at.button(key="welcome_show").click().run()
    assert any(b.key == "welcome_hide" for b in at.button)
