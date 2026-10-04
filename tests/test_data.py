"""Cartella dati esterna: conversione in Parquet, esploratore del connettoma, libreria di stimoli."""

import gzip
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from sandbox.data import codex, store

BASE = 720575940600000000  # ID in stile FlyWire (18 cifre: servono int64)
IDS = [BASE + k for k in range(1, 7)]

CLASSIFICATION = pd.DataFrame({
    "root_id": IDS,
    "flow": ["afferent", "afferent", "intrinsic", "intrinsic", "intrinsic", "intrinsic"],
    "super_class": ["sensory", "sensory", "central", "central", "central", "central"],
    "class": ["olfactory", "olfactory", "ALPN", "ALLN", "Kenyon_Cell", "MBON"],
    "cell_type": ["ORN_DA1", "ORN_DA1", "DA1_lPN", "LN_1", "KCg-m", "MBON01"],
    "side": ["left", "right", "left", "left", "left", "left"],
})
NEURONS_NT = pd.DataFrame({"root_id": IDS, "nt_type": ["ACH", "ACH", "ACH", "GABA", "ACH", "GLUT"],
                           "nt_type_score": [0.9] * 6})
# (pre, post, neuropil, sinapsi): 1→3 compare in due neuropili; 6→5 è sotto la soglia di 5
EDGES = [(1, 3, "AL_L", 10), (1, 3, "AL_R", 3), (2, 3, "AL_L", 8), (1, 4, "AL_L", 6), (4, 3, "AL_L", 7),
         (3, 5, "MB_CA_L", 5), (3, 5, "MB_CA_R", 1), (5, 6, "MB_ML_L", 12), (6, 5, "MB_ML_L", 2)]
CONNECTIONS = pd.DataFrame({
    "pre_root_id": [BASE + a for a, *_ in EDGES], "post_root_id": [BASE + b for _, b, *_ in EDGES],
    "neuropil": [n for *_, n, _ in EDGES], "syn_count": [s for *_, s in EDGES],
    "nt_type": ["GABA" if a == 4 else "GLUT" if a == 6 else "ACH" for a, *_ in EDGES],
})


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(store.ENV_VAR, str(tmp_path / "dati"))
    return tmp_path / "dati"


def _csv_bytes(df: pd.DataFrame, **kwargs) -> bytes:
    return df.to_csv(index=False, **kwargs).encode()


def _import(df: pd.DataFrame, filename: str):
    data = _csv_bytes(df)
    if filename.endswith(".gz"):
        data = gzip.compress(data)
    return store.import_table(io.BytesIO(data), filename)


def _codex():
    _import(CLASSIFICATION, "classification.csv.gz")
    _import(NEURONS_NT, "neurons.csv")
    _import(CONNECTIONS, "connections.csv.gz")
    folder = store.folder(store.CONNECTOME)
    return codex.load_connectome([folder / "classification.parquet", folder / "neurons.parquet"],
                                 folder / "connections.parquet")


# ============================================================================ cartella e conversione

def test_data_root_follows_environment(data_dir):
    assert store.data_root() == data_dir
    assert store.folder(store.STIMULI).is_dir() and store.folder(store.CONNECTOME).is_dir()


@pytest.mark.parametrize("filename, stem", [("connections.csv.gz", "connections"), ("a b (1).csv", "a_b_1"),
                                            ("body-annotations-male-cns-v1.0.feather",
                                             "body-annotations-male-cns-v1.0"), ("../x.parquet", "x")])
def test_safe_stem(filename, stem):
    assert store.safe_stem(filename) == stem


def test_csv_and_gzip_convert_to_the_same_parquet():
    plain = _import(CONNECTIONS, "plain.csv")
    packed = _import(CONNECTIONS, "packed.csv.gz")
    a, b = store.preview(plain, 100), store.preview(packed, 100)
    pd.testing.assert_frame_equal(a, b)
    assert a["pre_root_id"].dtype == np.int64 and a["pre_root_id"].iloc[0] == BASE + 1  # nessuna perdita di cifre
    names = {t.name: t for t in store.list_tables()}
    assert names["plain.parquet"].rows == len(EDGES) and "neuropil" in names["plain.parquet"].columns
    assert not list(store.folder(store.CONNECTOME).glob("*.part"))


def test_import_from_disk_path_and_tab_separated(tmp_path):
    source = tmp_path / "export.tsv"
    source.write_bytes(_csv_bytes(CLASSIFICATION, sep="\t"))
    path = store.import_table(source, source.name)
    assert list(store.preview(path).columns) == list(CLASSIFICATION.columns)
    assert source.exists()  # il file originale dell'utente non si tocca


def test_csv_with_types_changing_between_blocks(monkeypatch):
    monkeypatch.setattr(store, "BLOCK_SIZE", 256)
    n = 200
    df = pd.DataFrame({"root_id": range(n),
                       "late_text": [""] * (n - 5) + ["x"] * 5,                    # vuota nel primo blocco
                       "mixed": [str(i) for i in range(n - 1)] + ["non-numero"]})  # intera, poi testo
    table = store.preview(_import(df, "tricky.csv"), n)
    assert len(table) == n and table["mixed"].iloc[-1] == "non-numero" and table["late_text"].iloc[-1] == "x"


def test_feather_import_from_path_and_upload(tmp_path):
    source = tmp_path / "body-annotations.feather"
    feather.write_feather(pa.Table.from_pandas(CLASSIFICATION), source, chunksize=2)
    from_path = store.preview(store.import_table(source, source.name), 10)
    upload = io.BytesIO(source.read_bytes())
    from_upload = store.preview(store.import_table(upload, "copia.feather"), 10)
    pd.testing.assert_frame_equal(from_path, from_upload)
    assert len(from_path) == len(CLASSIFICATION)


def test_parquet_upload_and_delete():
    buffer = io.BytesIO()
    CLASSIFICATION.to_parquet(buffer)
    store.import_table(buffer, "cls.parquet")
    assert [t.name for t in store.list_tables()] == ["cls.parquet"]
    store.delete(store.CONNECTOME, "cls.parquet")
    assert store.list_tables() == []
    with pytest.raises(ValueError):
        store.delete(store.CONNECTOME, "../fuori.parquet")


def test_invalid_file_leaves_nothing_behind():
    with pytest.raises(Exception):
        store.import_table(io.BytesIO(b"non e' un feather"), "rotto.feather")
    assert not list(store.folder(store.CONNECTOME).iterdir())


# ============================================================================ riconoscimento e esploratore

@pytest.mark.parametrize("columns, role", [
    (["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"], "archi"),       # Codex
    (["body_pre", "body_post", "weight"], "archi"),                                       # Male CNS
    (["bodyId_pre", "bodyId_post", "roi", "weight"], "archi"),                            # neuPrint
    (["root_id", "flow", "super_class", "class", "sub_class", "cell_type"], "neuroni"),   # Codex
    (["bodyId", "type", "instance", "superclass", "somaSide"], "neuroni"),                # Male CNS
    (["root_id", "primary_type", "additional_type(s)"], "neuroni"),
    (["bodyId", "pre", "post", "downstream", "upstream"], "altro"),                       # statistiche
    (["root_id", "position", "supervoxel_id"], "altro"),
    (["pre_root_id", "post_root_id", "x", "y", "z"], "sinapsi"),                          # synapse_coordinates
    (["pre_x", "pre_y", "size", "pre_root_id_720575940", "post_root_id_720575940", "neuropil"], "altro"),
    (["root_id", "hemisphere", "type", "column_id", "x", "y", "p", "q"], "neuroni"),      # column_assignment
    (["neuropil", "count_total", "count_proof", "proof_ratio", "side"], "altro"),
])
def test_detect_role(columns, role):
    assert codex.detect_role(columns) == role


def test_connectome_merges_tables_and_counts_synapses():
    cx = _codex()
    assert len(cx.neurons) == 6
    assert set(cx.fields) >= {"super_class", "class", "cell_type", "nt", "side", "flow"}
    pn = cx.neurons.loc[BASE + 3]
    assert pn["cell_type"] == "DA1_lPN" and pn["nt"] == "ACH"
    assert pn["syn_in"] == 10 + 3 + 8 + 7 and pn["syn_out"] == 5 + 1
    assert cx.neuropils[0] == "AL_L"  # ordinati per numero di sinapsi


def test_filters_by_field_text_and_neuropil():
    cx = _codex()
    assert len(codex.filter_neurons(cx, {"super_class": ["sensory"]})) == 2
    assert len(codex.filter_neurons(cx, {"super_class": ["central"], "nt": ["GABA", "GLUT"]})) == 2
    assert list(codex.filter_neurons(cx, text="kcg, mbon").index) == [BASE + 5, BASE + 6]
    assert list(codex.filter_neurons(cx, text=str(BASE + 4)).index) == [BASE + 4]
    in_calyx = codex.filter_neurons(cx, neuropils=["MB_CA_L", "MB_CA_R"], min_neuropil_syn=6)
    assert list(in_calyx.index) == [BASE + 3, BASE + 5]
    assert codex.filter_neurons(cx, neuropils=["MB_CA_R"], min_neuropil_syn=2).empty


def test_category_counts_groups_the_tail():
    counts = codex.category_counts(pd.Series(list("abcdefghij") + ["a"] * 5), top=3)
    assert list(counts.index) == ["a", "b", "c", "altro"] and counts["altro"] == 7


def test_build_subcircuit_signs_weights_and_labels():
    cx = _codex()
    circuit, warnings, kept = codex.build_subcircuit(cx, IDS[:4], syn_to_nS=0.1, min_synapses=5)
    assert circuit.neurons == ["ORN_DA1", "ORN_DA1#2", "DA1_lPN", "LN_1"] and kept == IDS[:4]
    w = pd.DataFrame(circuit.weights, index=circuit.neurons, columns=circuit.neurons)
    assert w.loc["ORN_DA1", "DA1_lPN"] == pytest.approx(1.3)       # 10 + 3 sinapsi sommate sui neuropili
    assert w.loc["LN_1", "DA1_lPN"] == pytest.approx(-0.7)         # GABA → inibitoria
    assert (w.loc[["ORN_DA1", "ORN_DA1#2"]] >= 0).all().all()
    assert circuit.default_stimuli[0]["targets"] == ["ORN_DA1", "ORN_DA1#2"]  # sorgenti senza input


def test_subcircuit_threshold_drops_weak_edges_and_isolated_neurons():
    cx = _codex()
    circuit, warnings, kept = codex.build_subcircuit(cx, [BASE + 5, BASE + 6, BASE + 1], min_synapses=5)
    assert circuit.neurons == ["KCg-m", "MBON01"] and kept == [BASE + 5, BASE + 6]
    assert circuit.weights[1, 0] == 0  # 6 → 5 ha solo 2 sinapsi
    assert any("scartati" in w for w in warnings)


def test_subcircuit_limits():
    cx = _codex()
    with pytest.raises(ValueError):
        codex.build_subcircuit(cx, IDS[:1])
    with pytest.raises(ValueError):
        codex.build_subcircuit(cx, range(codex.MAX_SUBCIRCUIT + 1))


def test_neuprint_style_tables_without_neuropils(tmp_path):
    annotations = pd.DataFrame({"bodyId": [11, 12, 13], "type": ["A", "B", "C"], "superclass": ["x", "x", "y"],
                                "predictedNt": ["acetylcholine", "gaba", "histamine"]})
    weights = pd.DataFrame({"body_pre": [11, 12, 13, 11], "body_post": [12, 13, 11, 13], "weight": [9, 9, 9, 1]})
    for df, name in ((annotations, "body-annotations.feather"), (weights, "connectome-weights.feather")):
        feather.write_feather(pa.Table.from_pandas(df), tmp_path / name)
        store.import_table(tmp_path / name, name)
    folder = store.folder(store.CONNECTOME)
    cx = codex.load_connectome([folder / "body-annotations.parquet"], folder / "connectome-weights.parquet")
    assert cx.neuropils == [] and cx.neurons.loc[11, "syn_out"] == 10
    circuit, _, _ = codex.build_subcircuit(cx, [11, 12, 13], syn_to_nS=1.0, min_synapses=2)
    w = pd.DataFrame(circuit.weights, index=circuit.neurons, columns=circuit.neurons)
    assert w.loc["A", "B"] == 9 and w.loc["B", "C"] == -9 and w.loc["C", "A"] == -9  # GABA e istamina
    assert w.loc["A", "C"] == 0


def test_edges_only_derives_transmitter_from_connections():
    _import(CONNECTIONS, "connections.csv")
    cx = codex.load_connectome([], store.folder(store.CONNECTOME) / "connections.parquet")
    assert len(cx.neurons) == 6 and cx.neurons.loc[BASE + 4, "nt"] == "GABA"
    assert codex.filter_neurons(cx, {"nt": ["GLUT"]}).index.tolist() == [BASE + 6]


# ============================================================================ libreria di stimoli

def test_stimulus_library_roundtrip():
    name = store.save_stimulus("registrazione cellula 3.csv", b"tempo;corrente\n0;0\n10;25.5\n20;0\n")
    assert name == "registrazione_cellula_3.csv" and store.list_stimuli() == [name]
    loaded = store.load_stimulus(name)
    assert loaded["t"] == [0.0, 10.0, 20.0] and loaded["v"] == [0.0, 25.5, 0.0] and loaded["source"] == name
    store.delete(store.STIMULI, name)
    assert store.list_stimuli() == []


@pytest.mark.parametrize("data", [b"solo\n1\n2\n", b"t,v\n0,1\n", b"t,v\na,b\nc,d\n"])
def test_invalid_stimulus_is_rejected(data):
    with pytest.raises(ValueError):
        store.save_stimulus("x.csv", data)
    assert store.list_stimuli() == []
