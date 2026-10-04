"""Caricamento di connettività reale (FlyWire / Hemibrain) come :class:`Circuit`.

Tre vie d'ingresso, dalla più semplice alla più "live":

1. ``load_edge_csv``          — lista di archi CSV. Funziona offline e accetta
   direttamente gli export di FlyWire Codex (``connections.csv``: colonne
   ``pre_root_id, post_root_id, neuropil, syn_count, nt_type``) o di neuPrint.
2. ``fetch_flywire_edges``    — query live con ``caveclient`` (datastack
   pubblico ``flywire_fafb_public``, richiede un token CAVE).
3. ``fetch_hemibrain_edges``  — query live con ``neuprint-python`` (lo stesso
   client esposto da ``navis.interfaces.neuprint``; richiede un token neuPrint).

In tutti i casi il numero di sinapsi viene convertito in conduttanza di picco
(``syn_to_nS`` nS per sinapsi) e il segno dipende dal neurotrasmettitore
predetto del neurone presinaptico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .topologies import Circuit

# Segno del trasmettitore nel SNC di Drosophila. I neuromodulatori (DA, 5-HT, OA)
# agiscono per lo più via recettori metabotropici: qui li trattiamo come eccitatori
# deboli di default; correggi a mano la matrice se ti serve un modello diverso.
NT_SIGN = {
    "ach": 1.0, "acetylcholine": 1.0,
    "gaba": -1.0,
    "glut": -1.0, "glutamate": -1.0,
    "da": 1.0, "dopamine": 1.0,
    "ser": 1.0, "serotonin": 1.0,
    "oct": 1.0, "octopamine": 1.0,
    "tyr": 1.0, "tyramine": 1.0,
    # l'istamina apre canali al cloro (HisCl, es. fotorecettori → neuroni laminari): inibitoria
    "his": -1.0, "histamine": -1.0,
}

_PRE_COLUMNS = ("pre", "pre_root_id", "pre_pt_root_id", "bodyId_pre", "source", "pre_id")
_POST_COLUMNS = ("post", "post_root_id", "post_pt_root_id", "bodyId_post", "target", "post_id")
_WEIGHT_COLUMNS = ("weight", "syn_count", "synapses", "count", "n_synapses")
_NT_COLUMNS = ("nt", "nt_type", "neurotransmitter", "predicted_nt", "top_nt")


def _find_column(df: pd.DataFrame, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValueError(f"Colonna non trovata: attesa una tra {', '.join(candidates)}")
    return None


def edges_to_circuit(edges: pd.DataFrame, *, syn_to_nS: float = 0.15, min_synapses: int = 1,
                     max_neurons: int = 60, neuron_ids=None, labels: dict | None = None,
                     name: str = "Connettoma") -> tuple[Circuit, list[str]]:
    """Aggrega una lista di archi in una matrice W[pre, post] (nS).

    Restituisce il circuito e una lista di avvisi da mostrare all'utente.
    """
    warnings: list[str] = []
    pre_col = _find_column(edges, _PRE_COLUMNS)
    post_col = _find_column(edges, _POST_COLUMNS)
    w_col = _find_column(edges, _WEIGHT_COLUMNS, required=False)
    nt_col = _find_column(edges, _NT_COLUMNS, required=False)

    df = edges.copy()
    df[pre_col] = df[pre_col].astype(str)
    df[post_col] = df[post_col].astype(str)
    df["_w"] = df[w_col].astype(float) if w_col else 1.0

    if neuron_ids:
        ids = {str(i) for i in neuron_ids}
        df = df[df[pre_col].isin(ids) & df[post_col].isin(ids)]

    # somma sui neuropili (Codex/neuPrint riportano una riga per regione)
    agg = df.groupby([pre_col, post_col], as_index=False)["_w"].sum()
    agg = agg[agg["_w"] >= min_synapses]
    if agg.empty:
        raise ValueError("Nessuna connessione rimasta dopo i filtri: abbassa la soglia di sinapsi o controlla gli ID.")

    present = set(agg[pre_col]) | set(agg[post_col])
    if neuron_ids:
        order = [str(i) for i in neuron_ids if str(i) in present]
    else:
        # ordine di prima comparsa nel file (tipicamente sensoriale → motorio)
        order = list(dict.fromkeys(v for pair in zip(df[pre_col], df[post_col]) for v in pair if v in present))
        if len(order) > max_neurons:
            # rete troppo grande: tengo i neuroni con più sinapsi (input + output)
            degree = agg.groupby(pre_col)["_w"].sum().add(agg.groupby(post_col)["_w"].sum(), fill_value=0)
            keep = set(degree.nlargest(max_neurons).index)
            warnings.append(f"{len(order)} neuroni trovati: tengo i {max_neurons} più connessi.")
            order = [nid for nid in order if nid in keep]

    index = {nid: i for i, nid in enumerate(order)}
    agg = agg[agg[pre_col].isin(index) & agg[post_col].isin(index)]

    sign = np.ones(len(order))
    if nt_col:
        nt_per_neuron = (df.dropna(subset=[nt_col])
                         .groupby(pre_col)[nt_col]
                         .agg(lambda s: s.astype(str).str.lower().mode().iat[0]))
        unknown = set()
        for nid, nt in nt_per_neuron.items():
            if nid in index:
                if nt not in NT_SIGN:
                    unknown.add(nt)
                sign[index[nid]] = NT_SIGN.get(nt, 1.0)
        if unknown:
            warnings.append(f"Trasmettitori non riconosciuti (trattati come eccitatori): {', '.join(sorted(unknown))}")
    else:
        warnings.append("Nessuna colonna di neurotrasmettitore: tutte le sinapsi sono eccitatorie. "
                        "Inverti il segno delle righe dei neuroni inibitori nell'editor.")

    weights = np.zeros((len(order), len(order)))
    for pre, post, w in agg[[pre_col, post_col, "_w"]].itertuples(index=False):
        weights[index[pre], index[post]] = sign[index[pre]] * w * syn_to_nS

    names = [str(labels.get(nid, nid)) if labels else nid for nid in order]
    names = _deduplicate(names)
    # di default si stimolano i neuroni "sorgente" (senza input nel sottocircuito)
    sources = [names[i] for i in range(len(names)) if not weights[:, i].any()] or names[:1]
    circuit = Circuit(
        name=name,
        neurons=names,
        weights=weights,
        description=f"{len(order)} neuroni, {int((weights != 0).sum())} connessioni, "
                    f"{syn_to_nS:g} nS per sinapsi.",
        default_stimuli=[{"kind": "Onda quadra", "targets": sources,
                          "amplitude": 30.0, "t_on": 50.0, "t_off": 400.0}],
    )
    return circuit, warnings


def _deduplicate(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}#{seen[n]}")
    return out


def load_edge_csv(source, **kwargs) -> tuple[Circuit, list[str]]:
    """Legge un CSV (percorso o file caricato) e lo converte in circuito.

    Se il file contiene le colonne ``pre_name``/``post_name`` vengono usate come etichette.
    """
    edges = pd.read_csv(source)
    labels = {}
    for id_candidates, name_col in ((_PRE_COLUMNS, "pre_name"), (_POST_COLUMNS, "post_name")):
        if name_col in edges.columns:
            id_col = _find_column(edges, id_candidates)
            labels.update(dict(zip(edges[id_col].astype(str), edges[name_col].astype(str))))
    return edges_to_circuit(edges, labels=labels or None, **kwargs)


def fetch_flywire_edges(root_ids, *, datastack: str = "flywire_fafb_public",
                        materialization_version: int | None = None) -> pd.DataFrame:
    """Scarica le sinapsi tra ``root_ids`` dal connettoma FlyWire tramite CAVEclient.

    Richiede ``pip install caveclient`` e un token salvato con
    ``CAVEclient().auth.save_token(...)`` (vedi https://fafbseg-py.readthedocs.io).
    """
    try:
        from caveclient import CAVEclient
    except ImportError as exc:
        raise ImportError("Installa le dipendenze opzionali: pip install -r requirements-connectome.txt") from exc

    client = CAVEclient(datastack)
    ids = [int(i) for i in root_ids]
    synapses = client.materialize.synapse_query(pre_ids=ids, post_ids=ids,
                                                materialization_version=materialization_version)
    pre_col, post_col = "pre_pt_root_id", "post_pt_root_id"
    edges = synapses.groupby([pre_col, post_col]).size().rename("syn_count").reset_index()

    # Predizioni di neurotrasmettitore per sinapsi (Eckstein et al. 2024), se presenti nella tabella
    nt_cols = [c for c in ("gaba", "ach", "glut", "oct", "ser", "da") if c in synapses.columns]
    if nt_cols:
        nt = synapses.groupby(pre_col)[nt_cols].mean().idxmax(axis=1).rename("nt_type")
        edges = edges.merge(nt, left_on=pre_col, right_index=True, how="left")
    return edges


def fetch_hemibrain_edges(body_ids, *, token: str, dataset: str = "hemibrain:v1.2.1",
                          server: str = "neuprint.janelia.org") -> tuple[pd.DataFrame, dict]:
    """Scarica le adiacenze tra ``body_ids`` dal connettoma Hemibrain via neuPrint.

    Restituisce (archi, etichette {bodyId: tipo_istanza}). Hemibrain v1.2.1 non
    include predizioni di neurotrasmettitore: i segni vanno impostati a mano.
    """
    try:
        from neuprint import Client, NeuronCriteria, fetch_adjacencies
    except ImportError:
        try:  # navis re-esporta neuprint-python
            from navis.interfaces.neuprint import Client, NeuronCriteria, fetch_adjacencies
        except ImportError as exc:
            raise ImportError("Installa le dipendenze opzionali: pip install -r requirements-connectome.txt") from exc

    client = Client(server, dataset=dataset, token=token)
    criteria = NeuronCriteria(bodyId=[int(b) for b in body_ids])
    neurons, conn = fetch_adjacencies(criteria, criteria, client=client)
    edges = conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    labels = {}
    if "type" in neurons.columns:
        labels = {str(b): (t if isinstance(t, str) and t else str(b))
                  for b, t in zip(neurons["bodyId"], neurons["type"])}
    return edges, labels
