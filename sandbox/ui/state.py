"""Stato dei widget: chiavi stabili, valori importati da file e registro per l'esportazione.

Ogni widget della Sandbox viene creato tramite queste funzioni con un *nome*
logico (es. ``"lif_tau_m"``). Tre meccanismi:

* **registro** — a ogni esecuzione dello script i valori correnti finiscono in un
  dizionario, che diventa la sezione ``widgets`` del file JSON esportato;
* **importazione** — caricare un file imposta i valori di default e incrementa una
  "generazione" che fa parte di tutte le chiavi: i widget vengono ricreati con i
  valori importati, senza scrivere su chiavi di widget già esistenti (che Streamlit
  vieta) e senza avvisi;
* **ripristino** — un'importazione vuota riporta tutto ai valori di default.
"""

import math

import streamlit as st

GENERATION = "_widget_generation"
IMPORTED = "_imported_widgets"
REGISTRY = "_widget_registry"


def begin_run() -> None:
    st.session_state[REGISTRY] = {}


def key(name: str) -> str:
    return f"g{st.session_state.get(GENERATION, 0)}·{name}"


def default(name: str, value):
    return st.session_state.get(IMPORTED, {}).get(name, value)


def record(name: str, value):
    st.session_state.setdefault(REGISTRY, {})[name] = value
    return value


def registry() -> dict:
    return dict(st.session_state.get(REGISTRY, {}))


def load(widgets: dict) -> None:
    """Da chiamare in una callback: al prossimo giro i widget ripartono dai valori dati."""
    st.session_state[IMPORTED] = dict(widgets)
    st.session_state[GENERATION] = st.session_state.get(GENERATION, 0) + 1


def _like(value, reference):
    """Converte un valore importato al tipo atteso dal widget (JSON non distingue 3 da 3.0)."""
    try:
        if isinstance(reference, bool):
            return bool(value)
        if isinstance(reference, int):
            return int(round(float(value)))
        if isinstance(reference, float):
            value = float(value)
            return value if math.isfinite(value) else reference
    except (TypeError, ValueError):
        return reference
    return value


def _clamp(value, lo, hi):
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def slider(label, name, lo, hi, value, step=None, where=st, **kwargs):
    v = _clamp(_like(default(name, value), value), lo, hi)
    return record(name, where.slider(label, lo, hi, v, step, key=key(name), **kwargs))


def number(label, name, lo, hi, value, step=None, where=st, **kwargs):
    v = _clamp(_like(default(name, value), value), lo, hi)
    return record(name, where.number_input(label, lo, hi, v, step, key=key(name), **kwargs))


def select(label, name, options, value=None, where=st, **kwargs):
    options = list(options)
    v = default(name, value if value is not None else options[0])
    return record(name, where.selectbox(label, options, index=options.index(v) if v in options else 0,
                                        key=key(name), **kwargs))


def radio(label, name, options, value=None, where=st, **kwargs):
    options = list(options)
    v = default(name, value if value is not None else options[0])
    return record(name, where.radio(label, options, index=options.index(v) if v in options else 0,
                                    key=key(name), **kwargs))


def multi(label, name, options, value=(), where=st, **kwargs):
    options = list(options)
    chosen = [x for x in default(name, list(value)) if x in options]
    if kwargs.get("max_selections"):
        chosen = chosen[: kwargs["max_selections"]]
    return record(name, where.multiselect(label, options, default=chosen, key=key(name), **kwargs))


def check(label, name, value=False, where=st, **kwargs):
    return record(name, where.checkbox(label, bool(default(name, value)), key=key(name), **kwargs))


def toggle(label, name, value=False, where=st, **kwargs):
    return record(name, where.toggle(label, bool(default(name, value)), key=key(name), **kwargs))


def text(label, name, value="", where=st, area=False, **kwargs):
    widget = where.text_area if area else where.text_input
    return record(name, widget(label, str(default(name, value)), key=key(name), **kwargs))


def select_slider(label, name, options, value, where=st, **kwargs):
    options = list(options)
    v = default(name, value)
    if v not in options:
        v = value
    return record(name, where.select_slider(label, options, value=v, key=key(name), **kwargs))
