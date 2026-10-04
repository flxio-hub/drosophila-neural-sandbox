"""Pagina "Dati": cartella dati esterna, tabelle del connettoma, esploratore e libreria di stimoli."""

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from sandbox.data import codex, store
from sandbox.ui import look, state
from sandbox.ui.controls import SUBCIRCUIT, SUBCIRCUIT_KEY, signature
from sandbox.ui.look import chart, explain
from sandbox.ui.pages import SIMULATION
from sandbox.ui.texts import TIPS
from sandbox.viz.data_plots import category_figure, waveform_figure

state.keep_alive()  # i widget della Sandbox ritrovano i loro valori al ritorno
theme = look.current_theme()
UPLOAD_GEN = "_data_upload_generation"
MESSAGES = "_data_messages"

st.title("📁 Dati")
st.caption(f"Cartella dati: `{store.data_root()}` · connettoma in `connectome/`, stimoli in `stimuli/`. "
           f"Per spostarla imposta la variabile d'ambiente `{store.ENV_VAR}`.")
explain("data_folder")
for kind, text in st.session_state.pop(MESSAGES, []):
    getattr(st, kind)(text)


def _notify(kind: str, text: str) -> None:
    st.session_state.setdefault(MESSAGES, []).append((kind, text))


def _import(source, filename: str) -> None:
    status = st.empty()
    start = time.perf_counter()
    try:
        path = store.import_table(source, filename,
                                  progress=lambda rows: status.caption(f"{filename}: {rows:,} righe convertite…"))
        rows = next(t.rows for t in store.list_tables() if t.path == path)
        _notify("success", f"«{filename}» → `{path.name}`: {rows:,} righe in {time.perf_counter() - start:.1f} s.")
    except Exception as exc:
        _notify("error", f"«{filename}» non importato: {exc}")
    status.empty()


tab_files, tab_explorer, tab_stimuli = st.tabs(["🗂️ Tabelle del connettoma", "🧭 Esploratore del connettoma",
                                                 "⚡ Libreria di stimoli"])

# ============================================================================ tabelle del connettoma
with tab_files:
    st.subheader("Carica le tabelle")
    explain("codex_files")
    col_up, col_path = st.columns(2, gap="large")
    with col_up:
        st.markdown("**Dal browser**")
        uploads = st.file_uploader("CSV, CSV.gz, Feather o Parquet", type=store.TABLE_TYPES,
                                   accept_multiple_files=True, help=TIPS["upload_connectome"],
                                   key=f"data_upload_{st.session_state.get(UPLOAD_GEN, 0)}")
        if st.button("⬆️ Importa i file caricati", disabled=not uploads, help=TIPS["btn_import_upload"],
                     key="data_import_upload"):
            with st.spinner("Converto in Parquet…"):
                for upload in uploads:
                    _import(upload, upload.name)
            st.session_state[UPLOAD_GEN] = st.session_state.get(UPLOAD_GEN, 0) + 1  # svuota il caricatore
            st.rerun()
    with col_path:
        st.markdown("**Da un percorso sul disco** (file molto grandi)")
        path_text = state.text("Percorso del file", "data_path", "",
                               placeholder=r"C:\Users\…\Downloads\connections.csv.gz").strip().strip('"')
        if st.button("📥 Importa dal percorso", disabled=not path_text, help=TIPS["btn_import_path"],
                     key="data_import_path"):
            source = Path(path_text).expanduser()
            if source.is_file():
                with st.spinner("Converto in Parquet…"):
                    _import(source, source.name)
            else:
                _notify("error", f"File non trovato: {source}")
            st.rerun()

    st.subheader("Tabelle disponibili")
    tables = store.list_tables()
    if not tables:
        st.info("Nessuna tabella: carica almeno gli archi (es. `connections`) e una tabella di neuroni "
                "(es. `classification`).")
    else:
        roles = {t.name: codex.detect_role(t.columns) for t in tables}
        st.dataframe(pd.DataFrame({
            "File": [t.name for t in tables],
            "Tipo": [roles[t.name] for t in tables],
            "Righe": [t.rows for t in tables],
            "Colonne": [", ".join(t.columns[:8]) + (" …" if len(t.columns) > 8 else "") for t in tables],
            "MB": [round(t.size_mb, 1) for t in tables],
        }), hide_index=True, width="stretch")
        st.caption("Tipo **archi** = connessioni fra neuroni; **neuroni** = annotazioni per neurone (vengono unite "
                   "per ID); **sinapsi** = una riga per singola sinapsi, **altro** = tabelle non usate "
                   "dall'esploratore (puoi eliminarle per risparmiare spazio).")
        chosen = state.select("Anteprima", "data_preview", [t.name for t in tables])
        info = next(t for t in tables if t.name == chosen)
        st.dataframe(store.preview(info.path, 20), hide_index=True, width="stretch")
        c1, c2 = st.columns([0.3, 0.7], vertical_alignment="center")
        confirm = state.check(f"Confermo: elimina «{chosen}»", "data_confirm", False, where=c1)
        if c2.button("🗑️ Elimina", disabled=not confirm, help=TIPS["btn_delete"], key="data_delete"):
            store.delete(store.CONNECTOME, chosen)
            _notify("success", f"«{chosen}» eliminato.")
            st.rerun()


# ============================================================================ esploratore
def explorer() -> None:
    st.subheader("Esploratore del connettoma")
    explain("explorer")
    tables = store.list_tables()
    # prima la tabella di archi più piccola: di solito quella con soglia (es. connections_princeton)
    edge_tables = sorted((t for t in tables if codex.detect_role(t.columns) == "archi"), key=lambda t: t.rows)
    neuron_tables = [t for t in tables if codex.detect_role(t.columns) == "neuroni"]
    if not edge_tables and not neuron_tables:
        st.info("Carica prima le tabelle del connettoma nella scheda **🗂️ Tabelle del connettoma**.")
        return
    edges = None
    if edge_tables:
        edges_name = state.select("Tabella degli archi", "xp_edges", [t.name for t in edge_tables])
        edges = next(t for t in edge_tables if t.name == edges_name)
    st.caption("Neuroni da: " + (", ".join(f"`{t.name}`" for t in neuron_tables) or "solo tabella degli archi")
               + (f" · archi da `{edges.name}`" if edges else " · nessuna tabella di archi: niente sottocircuiti"))
    try:
        with st.spinner("Calcolo le statistiche del connettoma (solo la prima volta, può richiedere un minuto)…"):
            cx = codex.load_connectome([t.path for t in neuron_tables], edges.path if edges else None)
    except Exception as exc:
        st.error(f"Impossibile leggere il connettoma: {exc}")
        return

    # ------------------------------------------------------------------ filtri (a cascata)
    subset = cx.neurons
    selections = {}
    fields = [f for f in codex.FILTER_FIELDS if f in cx.fields]
    columns = st.columns(3)
    for k, name in enumerate(fields):
        options = subset[name].value_counts().index.tolist()  # i valori più frequenti per primi
        selections[name] = state.multi(codex.FIELD_LABELS[name], f"xp_{name}", options, where=columns[k % 3],
                                       placeholder="Tutti")
        if selections[name]:
            subset = subset[subset[name].isin(selections[name])]
    c1, c2, c3 = st.columns([0.4, 0.4, 0.2])
    text = state.text("Cerca (nome, tipo o ID)", "xp_text", "", where=c1, placeholder="es. DA1, KCg")
    neuropils = state.multi("Neuropili", "xp_neuropils", cx.neuropils, where=c2, placeholder="Tutti",
                            disabled=not cx.neuropils)
    min_np = state.number("Sinapsi min.", "xp_np_min", 1, 10_000, 5, where=c3, disabled=not neuropils)
    selected = codex.filter_neurons(cx, selections, text, neuropils, min_np)

    total_in, total_out = selected["syn_in"].sum(), selected["syn_out"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Neuroni selezionati", f"{len(selected):,}", help=f"su {len(cx.neurons):,} nel connettoma")
    m2.metric("Sinapsi in ingresso", f"{total_in:,.0f}")
    m3.metric("Sinapsi in uscita", f"{total_out:,.0f}")
    m4.metric("Tipi cellulari", f"{selected['cell_type'].nunique():,}" if "cell_type" in selected else "—")

    if len(selected):
        charts = []
        group = next((f for f in ("super_class", "class", "cell_type") if f in selected
                      and selected[f].nunique() > 1), None)
        if group:
            charts.append((f"Neuroni per {codex.FIELD_LABELS[group].lower()}",
                           category_figure(codex.category_counts(selected[group]), theme, "Neuroni")))
        if "nt" in selected:
            charts.append(("Neuroni per trasmettitore",
                           category_figure(codex.category_counts(selected["nt"]), theme, "Neuroni",
                                           transmitters=True)))
        if cx.neuropil_syn is not None:
            charts.append(("Sinapsi per neuropilo (ingresso + uscita)",
                           category_figure(codex.neuropil_counts(cx, selected.index), theme, "Sinapsi")))
        for col, (title, fig) in zip(st.columns(max(len(charts), 1)), charts):
            with col:
                st.markdown(f"**{title}**")
                chart(fig)
        if "nt" in selected:
            st.caption("Colori dei trasmettitori come nella matrice W: rosso = eccitatorio (ACh), blu = inibitorio "
                       "(GABA, glutammato, istamina), grigio = neuromodulatori o non determinato.")

    # ------------------------------------------------------------------ tabella e sottocircuito
    ranked = selected.assign(sinapsi=selected["syn_in"] + selected["syn_out"]).sort_values("sinapsi",
                                                                                            ascending=False)
    shown = ranked.head(1000)
    st.markdown(f"**Neuroni** — i {len(shown):,} più connessi fra i selezionati; seleziona le righe per sceglierli "
                "a mano.")
    table = st.dataframe(shown.reset_index().drop(columns="sinapsi"), hide_index=True, width="stretch",
                         height=300, on_select="rerun", selection_mode="multi-row", key="xp_table")
    picked_rows = list(table.selection.rows) if table is not None and hasattr(table, "selection") else []
    picked_rows = [r for r in picked_rows if r < len(shown)]  # la tabella può essere cambiata con i filtri

    st.subheader("🧬 Crea sottocircuito")
    explain("subcircuit")
    c1, c2 = st.columns([0.6, 0.4])
    mode = state.radio("Neuroni da usare", "xp_pick", ["I più connessi fra i selezionati", "Righe scelte nella tabella",
                                                       "Tutti i selezionati"], where=c1)
    top_n = state.number("Quanti", "xp_top_n", 2, codex.MAX_SUBCIRCUIT, 30, where=c2,
                         disabled=mode != "I più connessi fra i selezionati")
    if mode.startswith("I più connessi"):
        ids = ranked.index[: int(top_n)].tolist()
    elif mode.startswith("Righe"):
        ids = shown.index[picked_rows].tolist()
    else:
        ids = ranked.index.tolist()
    c1, c2, c3 = st.columns(3)
    syn_to_nS = state.slider("Conduttanza per sinapsi (nS)", "xp_syn_to_nS", 0.01, 1.0, 0.15, 0.01, where=c1)
    min_syn = state.number("Soglia sinapsi", "xp_min_syn", 1, 100, 5, where=c2)
    auto_name = "Connettoma: " + (", ".join(v for values in selections.values() for v in values[:2])[:60]
                                  or text or "selezione")
    name = state.text("Nome del circuito", "xp_name", "", where=c3, placeholder=auto_name) or auto_name
    too_many = len(ids) > codex.MAX_SUBCIRCUIT
    if too_many:
        st.warning(f"{len(ids):,} neuroni: al massimo {codex.MAX_SUBCIRCUIT}. Restringi i filtri o usa «I più "
                   "connessi».")
    elif len(ids) < 2:
        st.caption("Scegli almeno 2 neuroni.")
    else:
        st.caption(f"{len(ids)} neuroni pronti per la Sandbox.")
    if st.button("🧬 Crea sottocircuito e apri la Sandbox", type="primary", key="xp_build",
                 disabled=too_many or len(ids) < 2 or edges is None, help=TIPS["btn_subcircuit"]):
        try:
            with st.spinner("Leggo le sinapsi fra i neuroni scelti…"):
                circuit, warnings, kept = codex.build_subcircuit(cx, ids, syn_to_nS=syn_to_nS,
                                                                 min_synapses=int(min_syn), name=name)
        except Exception as exc:
            st.error(f"Impossibile creare il sottocircuito: {exc}")
        else:
            st.session_state[SUBCIRCUIT_KEY] = {
                "name": circuit.name, "neurons": circuit.neurons, "weights": circuit.weights.tolist(),
                "description": circuit.description + f" Da {edges.name}.",
                "default_stimuli": circuit.default_stimuli, "warnings": warnings,
                "ids": [str(i) for i in kept],
                "signature": signature(tuple(map(str, kept)), syn_to_nS, int(min_syn), time.time()),
            }
            st.session_state[state.key("preset")] = SUBCIRCUIT
            st.switch_page(SIMULATION)


with tab_explorer:
    explorer()

# ============================================================================ libreria di stimoli
with tab_stimuli:
    st.subheader("Libreria di stimoli")
    explain("stimulus_library")
    uploads = st.file_uploader("Forme d'onda (.csv: tempo in ms, valore in pA)", type=["csv", "txt"],
                               accept_multiple_files=True, help=TIPS["upload_stimulus"],
                               key=f"stim_upload_{st.session_state.get(UPLOAD_GEN, 0)}")
    if st.button("💾 Salva nella libreria", disabled=not uploads, help=TIPS["btn_save_stimuli"], key="stim_save"):
        for upload in uploads:
            try:
                _notify("success", f"«{upload.name}» salvato come `{store.save_stimulus(upload.name, upload.getvalue())}`.")
            except Exception as exc:
                _notify("error", f"«{upload.name}» non salvato: {exc}")
        st.session_state[UPLOAD_GEN] = st.session_state.get(UPLOAD_GEN, 0) + 1
        st.rerun()
    names = store.list_stimuli()
    if not names:
        st.info("La libreria è vuota. Gli stimoli salvati compaiono nel menu «Forma d'onda» dei canali di tipo "
                "*Da file CSV*.")
    else:
        chosen = state.select("Stimolo", "stim_preview", names)
        try:
            waveform = store.load_stimulus(chosen)
            chart(waveform_figure(waveform["t"], waveform["v"], theme, chosen))
            st.caption(f"{len(waveform['t'])} campioni tra {min(waveform['t']):g} e {max(waveform['t']):g} ms · "
                       f"valori tra {min(waveform['v']):g} e {max(waveform['v']):g} pA.")
        except Exception as exc:
            st.error(f"«{chosen}» non leggibile: {exc}")
        c1, c2 = st.columns([0.3, 0.7], vertical_alignment="center")
        confirm = state.check(f"Confermo: elimina «{chosen}»", "stim_confirm", False, where=c1)
        if c2.button("🗑️ Elimina", disabled=not confirm, help=TIPS["btn_delete"], key="stim_delete"):
            store.delete(store.STIMULI, chosen)
            _notify("success", f"«{chosen}» eliminato dalla libreria.")
            st.rerun()
