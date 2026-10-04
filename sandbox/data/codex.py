"""Esploratore del connettoma: tabelle di FlyWire Codex (FAFB, Male CNS…) o di neuPrint già convertite in Parquet.

Le tabelle si riconoscono dalle colonne, qualunque sia il nome del file:

* **archi** (``connections``): ID presinaptico e postsinaptico, numero di sinapsi, opzionali neuropilo e
  neurotrasmettitore — es. ``pre_root_id, post_root_id, neuropil, syn_count, nt_type`` (Codex) oppure
  ``body_pre, body_post, weight`` / ``bodyId_pre, bodyId_post, roi, weight`` (Male CNS, neuPrint);
* **neuroni** (``classification``, ``neurons``, ``names``, ``consolidated_cell_types``, annotazioni
  neuPrint…): un ID per riga più almeno un'annotazione (superclasse, classe, tipo, trasmettitore…).
  Più tabelle di neuroni vengono unite per ID; se due forniscono lo stesso campo vince la prima.

Le statistiche per neurone (sinapsi in ingresso/uscita, sinapsi per neuropilo) si calcolano con un'unica
lettura a blocchi della tabella degli archi, quindi la memoria resta contenuta anche con milioni di righe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..network.connectome import edges_to_circuit
from ..network.topologies import Circuit

MAX_SUBCIRCUIT = 150  # oltre, la simulazione diventa troppo lenta per l'uso interattivo
MISSING = "n.d."

ID_COLUMNS = ("root_id", "bodyid", "body_id", "body", "segment_id", "pt_root_id", "id")
PRE_COLUMNS = ("pre_root_id", "pre_pt_root_id", "bodyid_pre", "body_pre", "pre_id", "pre", "source")
POST_COLUMNS = ("post_root_id", "post_pt_root_id", "bodyid_post", "body_post", "post_id", "post", "target")
WEIGHT_COLUMNS = ("syn_count", "weight", "synapses", "n_synapses", "count", "syn")
NEUROPIL_COLUMNS = ("neuropil", "roi", "neuropil_name", "region")
EDGE_NT_COLUMNS = ("nt_type", "nt", "predicted_nt", "neurotransmitter")
SYNAPSE_COLUMNS = ("x", "y", "z", "pre_x", "post_x", "x_pre", "x_post", "ctr_x")  # coordinate di singole sinapsi

# campo canonico → colonne riconosciute, in ordine di preferenza
FIELDS = {
    "super_class": ("super_class", "superclass"),
    "class": ("class", "cell_class"),
    "sub_class": ("sub_class", "subclass"),
    "cell_type": ("cell_type", "primary_type", "type", "celltype"),
    "nt": ("nt_type", "consensusnt", "consensus_nt", "predictednt", "predicted_nt", "celltypepredictednt",
           "top_nt", "neurotransmitter", "nt"),
    "side": ("side", "somaside", "soma_side"),
    "flow": ("flow",),
    "name": ("name", "instance"),
}
FIELD_LABELS = {"super_class": "Superclasse", "class": "Classe", "sub_class": "Sottoclasse",
                "cell_type": "Tipo cellulare", "nt": "Neurotrasmettitore", "side": "Lato", "flow": "Flusso",
                "name": "Nome"}
FILTER_FIELDS = ("super_class", "class", "sub_class", "cell_type", "nt", "side", "flow")


def find_column(columns, candidates) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def detect_role(columns) -> str:
    """``"archi"``, ``"neuroni"``, ``"sinapsi"`` (una riga per sinapsi, con coordinate: non serve) oppure
    ``"altro"`` (tabella non usata dall'esploratore)."""
    has_id = find_column(columns, ID_COLUMNS) is not None
    if not has_id and find_column(columns, PRE_COLUMNS) and find_column(columns, POST_COLUMNS):
        # es. synapse_coordinates di Codex: pre, post, x, y, z (ID vuoti nelle righe ripetute)
        return "sinapsi" if find_column(columns, SYNAPSE_COLUMNS) else "archi"
    if has_id and any(find_column(columns, names) for names in FIELDS.values()):
        return "neuroni"
    return "altro"


def edge_columns(columns) -> dict:
    return {"pre": find_column(columns, PRE_COLUMNS), "post": find_column(columns, POST_COLUMNS),
            "weight": find_column(columns, WEIGHT_COLUMNS), "neuropil": find_column(columns, NEUROPIL_COLUMNS),
            "nt": find_column(columns, EDGE_NT_COLUMNS)}


@dataclass
class Connectome:
    neurons: pd.DataFrame            # indice = ID; campi canonici presenti + syn_in, syn_out
    fields: list[str]                # campi canonici disponibili
    edges_path: Path | None = None
    edge_cols: dict = field(default_factory=dict)
    neuropil_syn: pd.DataFrame | None = None   # colonne: id, neuropil, syn (ingresso + uscita)

    @property
    def neuropils(self) -> list[str]:
        if self.neuropil_syn is None:
            return []
        totals = self.neuropil_syn.groupby("neuropil")["syn"].sum().sort_values(ascending=False)
        return [str(n) for n in totals.index]


# ============================================================================ caricamento

def _normalize_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("int64")
    return values.astype(str)


def _read_neuron_table(path: Path) -> pd.DataFrame:
    columns = pq.read_schema(path).names
    id_col = find_column(columns, ID_COLUMNS)
    found = {name: find_column(columns, cands) for name, cands in FIELDS.items()}
    found = {name: col for name, col in found.items() if col is not None}
    df = pq.read_table(path, columns=[id_col, *dict.fromkeys(found.values())]).to_pandas()
    out = pd.DataFrame({name: df[col] for name, col in found.items()})
    out.index = _normalize_ids(df[id_col])
    out.index.name = "id"
    return out[~out.index.duplicated()]


BATCH_ROWS = 2_000_000


def _codes(column: pa.ChunkedArray | pa.Array, vocabulary: dict) -> np.ndarray:
    """Codici interi stabili fra i blocchi per una colonna testuale (neuropilo, trasmettitore); −1 = mancante."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if not pa.types.is_dictionary(array.type):
        array = pc.dictionary_encode(array.cast(pa.string()))
    lookup = np.array([vocabulary.setdefault(str(v), len(vocabulary)) for v in array.dictionary.to_pylist()]
                      + [-1], dtype=np.int64)
    indices = array.indices.fill_null(len(lookup) - 1).to_numpy(zero_copy_only=False)
    return lookup[indices]


def _accumulate(acc: np.ndarray, rows: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Somma ``weights`` nella matrice densa neurone × categoria (allargata se compaiono nuove categorie)."""
    ok = codes >= 0
    width = max(acc.shape[1], int(codes.max(initial=-1)) + 1)
    if width > acc.shape[1]:
        acc = np.hstack([acc, np.zeros((acc.shape[0], width - acc.shape[1]))])
    acc += np.bincount(rows[ok] * width + codes[ok], weights=weights[ok],
                       minlength=acc.size).reshape(acc.shape)
    return acc


def _edge_statistics(path: Path, cols: dict):
    """Sinapsi in uscita/ingresso per neurone, per neuropilo e trasmettitore prevalente.

    Due letture a blocchi della tabella degli archi (la prima raccoglie gli ID): gli ID diventano indici
    interi e i testi codici di dizionario, così le somme sono conteggi su interi (``bincount``) in matrici
    neurone × neuropilo; la memoria dipende dal numero di neuroni, non dalla lunghezza del file.
    """
    pre, post, weight, neuropil, nt = (cols[k] for k in ("pre", "post", "weight", "neuropil", "nt"))
    parquet = pq.ParquetFile(path, read_dictionary=[c for c in (neuropil, nt) if c])
    uniques = [pd.unique(np.concatenate([b.column(0).to_numpy(zero_copy_only=False),
                                         b.column(1).to_numpy(zero_copy_only=False)]))
               for b in parquet.iter_batches(batch_size=BATCH_ROWS, columns=[pre, post])]
    ids = np.unique(np.concatenate(uniques)) if uniques else np.array([], dtype=np.int64)
    n = len(ids)
    syn_out, syn_in = np.zeros(n), np.zeros(n)
    by_neuropil, by_nt = np.zeros((n, 0)), np.zeros((n, 0))
    np_vocab, nt_vocab = {}, {}
    needed = [c for c in dict.fromkeys((pre, post, weight, neuropil, nt)) if c]
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=needed):
        table = pa.Table.from_batches([batch])
        i_pre = np.searchsorted(ids, table[pre].to_numpy(zero_copy_only=False))
        i_post = np.searchsorted(ids, table[post].to_numpy(zero_copy_only=False))
        syn = (table[weight].to_numpy(zero_copy_only=False).astype(float) if weight
               else np.ones(table.num_rows))
        syn_out += np.bincount(i_pre, weights=syn, minlength=n)
        syn_in += np.bincount(i_post, weights=syn, minlength=n)
        if neuropil:
            code = _codes(table[neuropil], np_vocab)
            by_neuropil = _accumulate(by_neuropil, i_pre, code, syn)
            by_neuropil = _accumulate(by_neuropil, i_post, code, syn)
        if nt:
            by_nt = _accumulate(by_nt, i_pre, _codes(table[nt], nt_vocab), syn)

    index = _normalize_ids(pd.Series(ids)).to_numpy()
    totals = pd.DataFrame({"syn_out": syn_out, "syn_in": syn_in}, index=pd.Index(index, name="id"))
    neuropil_syn = None
    if neuropil:
        rows, codes = np.nonzero(by_neuropil)
        neuropil_syn = pd.DataFrame({"id": index[rows], "neuropil": np.array(list(np_vocab), dtype=object)[codes],
                                     "syn": by_neuropil[rows, codes]})
    nt_mode = None
    if nt and by_nt.shape[1]:
        has_nt = by_nt.sum(axis=1) > 0
        best = np.array(list(nt_vocab), dtype=object)[by_nt.argmax(axis=1)]
        nt_mode = pd.Series(best[has_nt], index=index[has_nt])
    return totals, neuropil_syn, nt_mode


@lru_cache(maxsize=4)
def _load_cached(neuron_files: tuple, edge_file: tuple | None) -> Connectome:
    tables = [_read_neuron_table(Path(path)) for path, _ in neuron_files]
    neurons = None
    for table in tables:
        neurons = table if neurons is None else neurons.combine_first(table)[
            list(dict.fromkeys([*neurons.columns, *table.columns]))]

    edges_path, cols, neuropil_syn = None, {}, None
    if edge_file is not None:
        edges_path = Path(edge_file[0])
        cols = edge_columns(pq.read_schema(edges_path).names)
        totals, neuropil_syn, nt_mode = _edge_statistics(edges_path, cols)
        if neurons is None:
            neurons = pd.DataFrame(index=totals.index)
        neurons = neurons.join(totals, how="left") if len(neurons.index) else totals
        if nt_mode is not None and "nt" not in neurons.columns:
            neurons["nt"] = nt_mode.reindex(neurons.index)
    if neurons is None:
        raise ValueError("Nessuna tabella di neuroni o di archi.")
    for name in ("syn_in", "syn_out"):
        neurons[name] = neurons[name].fillna(0.0) if name in neurons.columns else 0.0
    fields = [name for name in FIELDS if name in neurons.columns]
    for name in fields:
        neurons[name] = neurons[name].astype("string").fillna(MISSING).replace("", MISSING)
    neurons.index.name = "id"
    return Connectome(neurons=neurons, fields=fields, edges_path=edges_path, edge_cols=cols,
                      neuropil_syn=neuropil_syn)


def load_connectome(neuron_paths, edges_path=None) -> Connectome:
    """Unisce le tabelle dei neuroni e calcola le statistiche dagli archi (risultato in cache finché i file
    non cambiano)."""
    key = tuple((str(p), Path(p).stat().st_mtime) for p in neuron_paths)
    edge_key = (str(edges_path), Path(edges_path).stat().st_mtime) if edges_path else None
    return _load_cached(key, edge_key)


# ============================================================================ filtri e statistiche

def filter_neurons(cx: Connectome, selections: dict | None = None, text: str = "", neuropils=(),
                   min_neuropil_syn: float = 1.0) -> pd.DataFrame:
    """Neuroni che soddisfano tutti i filtri (liste vuote = nessun vincolo).

    ``text``: parole separate da virgole cercate (senza distinguere maiuscole) in ID, nome e tipo; basta una.
    ``neuropils``: il neurone deve avere almeno ``min_neuropil_syn`` sinapsi (ingresso + uscita) nei neuropili scelti.
    """
    df = cx.neurons
    mask = np.ones(len(df), dtype=bool)
    for name, values in (selections or {}).items():
        if values and name in df.columns:
            mask &= df[name].isin([str(v) for v in values]).to_numpy()
    words = [w.strip().lower() for w in text.split(",") if w.strip()]
    if words:
        haystack = pd.Series(df.index.astype(str), index=df.index)
        for name in ("name", "cell_type"):
            if name in df.columns:
                haystack = haystack + " " + df[name].astype(str)
        haystack = haystack.str.lower()
        hit = np.zeros(len(df), dtype=bool)
        for word in words:
            hit |= haystack.str.contains(word, regex=False).to_numpy()
        mask &= hit
    if neuropils and cx.neuropil_syn is not None:
        rows = cx.neuropil_syn[cx.neuropil_syn["neuropil"].isin(list(neuropils))]
        per_neuron = rows.groupby("id")["syn"].sum()
        mask &= df.index.isin(per_neuron[per_neuron >= min_neuropil_syn].index)
    return df[mask]


def category_counts(values: pd.Series, top: int = 7) -> pd.Series:
    """Conteggi dei primi ``top`` valori; il resto finisce in "altro" (max 8 barre, come la palette)."""
    counts = values.value_counts()
    if len(counts) > top + 1:
        counts = pd.concat([counts.iloc[:top], pd.Series({"altro": counts.iloc[top:].sum()})])
    return counts


def neuropil_counts(cx: Connectome, ids, top: int = 7) -> pd.Series:
    if cx.neuropil_syn is None:
        return pd.Series(dtype=float)
    rows = cx.neuropil_syn[cx.neuropil_syn["id"].isin(ids)]
    totals = rows.groupby("neuropil")["syn"].sum().sort_values(ascending=False)
    if len(totals) > top + 1:
        totals = pd.concat([totals.iloc[:top], pd.Series({"altro": totals.iloc[top:].sum()})])
    return totals


# ============================================================================ sottocircuito

def label_for(row: pd.Series, neuron_id) -> str:
    for name in ("cell_type", "name"):
        value = row.get(name, MISSING)
        if isinstance(value, str) and value and value != MISSING:
            return value
    return str(neuron_id)


def subcircuit_edges(cx: Connectome, ids) -> pd.DataFrame:
    """Archi tra i neuroni ``ids`` letti dal Parquet (solo le righe e le colonne necessarie)."""
    if cx.edges_path is None:
        raise ValueError("Serve una tabella di archi (es. connections.csv) per costruire un sottocircuito.")
    cols = cx.edge_cols
    schema = pq.read_schema(cx.edges_path)
    wanted = pa.array(list(ids)).cast(schema.field(cols["pre"]).type, safe=False)
    columns = [c for c in dict.fromkeys((cols["pre"], cols["post"], cols["weight"], cols["nt"])) if c]
    table = pq.read_table(cx.edges_path, columns=columns,
                          filters=pc.field(cols["pre"]).isin(wanted) & pc.field(cols["post"]).isin(wanted))
    edges = pd.DataFrame({
        "pre": _normalize_ids(table[cols["pre"]].to_pandas()),
        "post": _normalize_ids(table[cols["post"]].to_pandas()),
        "syn_count": table[cols["weight"]].to_pandas().astype(float) if cols["weight"] else 1.0,
    })
    # segno dal trasmettitore del neurone presinaptico (tabella dei neuroni, altrimenti colonna degli archi)
    if "nt" in cx.neurons.columns:
        nt = cx.neurons["nt"].replace(MISSING, np.nan)
        edges["nt_type"] = edges["pre"].map(nt)
    elif cols["nt"]:
        edges["nt_type"] = table[cols["nt"]].to_pandas()
    return edges


def build_subcircuit(cx: Connectome, ids, *, syn_to_nS: float = 0.15, min_synapses: int = 5,
                     name: str = "Sottocircuito") -> tuple[Circuit, list[str], list]:
    """Circuito della Sandbox dai neuroni scelti. Restituisce (circuito, avvisi, ID nell'ordine dei neuroni)."""
    ids = list(dict.fromkeys(ids))
    if len(ids) < 2:
        raise ValueError("Servono almeno 2 neuroni.")
    if len(ids) > MAX_SUBCIRCUIT:
        raise ValueError(f"Al massimo {MAX_SUBCIRCUIT} neuroni (ne hai scelti {len(ids)}).")
    edges = subcircuit_edges(cx, ids)
    if edges.empty:
        raise ValueError("Nessuna sinapsi fra i neuroni scelti.")
    edges["pre"], edges["post"] = edges["pre"].astype(str), edges["post"].astype(str)
    labels = {str(i): label_for(cx.neurons.loc[i], i) if i in cx.neurons.index else str(i) for i in ids}
    circuit, warnings = edges_to_circuit(edges, syn_to_nS=syn_to_nS, min_synapses=min_synapses,
                                         max_neurons=len(ids), neuron_ids=[str(i) for i in ids], labels=labels,
                                         name=name)
    # come edges_to_circuit: restano i neuroni con almeno una connessione sopra soglia
    pairs = edges.groupby(["pre", "post"])["syn_count"].sum()
    strong = pairs[pairs >= min_synapses].index
    present = set(strong.get_level_values(0)) | set(strong.get_level_values(1))
    kept = [i for i in ids if str(i) in present]
    dropped = len(ids) - len(kept)
    if dropped:
        warnings.insert(0, f"{dropped} neuroni scartati: nessuna connessione ≥ {min_synapses} sinapsi con gli altri.")
    return circuit, warnings, kept
