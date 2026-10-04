"""Aspetto comune delle pagine: stile, grafici, riquadri "Cosa sto guardando?" e schermata di benvenuto.

I colori e le dimensioni dei titoli sono in ``.streamlit/config.toml`` (colori ricavati dalla palette di
``viz/theme.py``); qui c'è solo un po' di CSS per spaziature e gerarchia, con colori semi-trasparenti che
funzionano sia nel tema chiaro sia in quello scuro. Il logo (``logo.svg``) usa il blu primario del tema.
"""

from pathlib import Path

import streamlit as st

from ..viz import DARK, LIGHT, Theme
from . import texts

LOGO = str(Path(__file__).with_name("logo.svg"))

CSS = """
<style>
/* meno spazio vuoto sopra il titolo (la barra di navigazione in alto è alta 60 px) */
[data-testid="stMainBlockContainer"] { padding-top: 4.75rem; }
/* app locale: il pulsante "Deploy" di Streamlit non serve */
[data-testid="stAppDeployButton"] { display: none; }
/* barra laterale: titoli di sezione compatti, separati da una linea sottile */
section[data-testid="stSidebar"] h2 {
    font-size: 1.05rem; font-weight: 650; letter-spacing: .01em;
    margin-top: .9rem; padding-top: .7rem; border-top: 1px solid rgba(137, 135, 129, .25);
}
section[data-testid="stSidebar"] h2:first-of-type { border-top: none; }
/* titoli di sezione della pagina */
[data-testid="stMain"] h3 { margin-top: .4rem; letter-spacing: -.005em; }
/* schede principali della pagina più marcate di quelle annidate nei pannelli */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stTabs"]
    > div > [role="tablist"] [data-testid="stTab"] p { font-size: 1rem; font-weight: 600; }
/* metriche come piccole card */
[data-testid="stMetric"] {
    border: 1px solid rgba(137, 135, 129, .28); border-radius: .6rem; padding: .6rem .9rem;
    background: rgba(137, 135, 129, .06);
}
[data-testid="stMetricLabel"] p { font-size: .85rem; opacity: .8; }
/* schermata di benvenuto: riquadro tinto col blu primario, card interne neutre */
.st-key-welcome { background: rgba(42, 120, 214, .06); border-color: rgba(42, 120, 214, .35) !important; }
[class*="st-key-welcome_card"] { background: rgba(137, 135, 129, .05); }
/* riquadri "Cosa sto guardando?": interruttori compatti, testo in grigio */
[class*="st-key-explain"] summary { opacity: .8; }
[class*="st-key-explain"] summary:hover { opacity: 1; }
</style>
"""

WELCOME_HIDDEN = "_welcome_hidden"
_EXPLAIN_COUNT = "_explain_count"


def apply_style() -> None:
    st.logo(LOGO, size="large")
    st.html(CSS)
    st.session_state[_EXPLAIN_COUNT] = {}


def current_theme() -> Theme:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def chart(fig) -> None:
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def explain(topic: str, expanded: bool = False) -> None:
    """Riquadro espandibile con la spiegazione di ``texts.EXPLAIN[topic]`` e un esempio di interpretazione."""
    body, example = texts.EXPLAIN[topic]
    # chiave per argomento (con un numero se ricorre), stabile fra un aggiornamento e l'altro: serve al CSS
    counts = st.session_state.setdefault(_EXPLAIN_COUNT, {})
    counts[topic] = counts.get(topic, 0) + 1
    key = f"explain_{topic}" + (f"_{counts[topic]}" if counts[topic] > 1 else "")
    with st.expander("Cosa sto guardando?", expanded=expanded, key=key, icon=":material/help:", type="compact"):
        st.markdown(body)
        if example:
            st.markdown(f"**Esempio di interpretazione.** {example}")


def section(title: str, topic: str | None = None) -> None:
    st.subheader(title)
    if topic:
        explain(topic)


def _set_welcome(hidden: bool) -> None:
    st.session_state[WELCOME_HIDDEN] = hidden


def welcome_panel() -> None:
    """Schermata iniziale: cosa si può fare in 30 secondi. Si chiude e si riapre con un pulsante."""
    if st.session_state.get(WELCOME_HIDDEN):
        return
    w = texts.WELCOME
    with st.container(border=True, key="welcome"):
        st.markdown(f"#### {w['title']}")
        st.markdown(w["lead"])
        for k, (col, (icon, title, body)) in enumerate(zip(st.columns(len(w["cards"])), w["cards"])):
            with col.container(border=True, key=f"welcome_card{k}", height="stretch"):
                st.markdown(f"**{icon} {title}**")
                st.caption(body)
        st.markdown(w["steps"])
        st.button("✓ Ho capito, nascondi", on_click=_set_welcome, args=(True,), key="welcome_hide")


def welcome_toggle(where=st) -> None:
    if st.session_state.get(WELCOME_HIDDEN):
        where.button("❓ Introduzione", on_click=_set_welcome, args=(False,), key="welcome_show",
                     help="Mostra di nuovo la schermata di benvenuto.")
