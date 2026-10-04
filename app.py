"""Drosophila Neural Sandbox — punto d'ingresso Streamlit.

Avvio:  streamlit run app.py

Due pagine (navigazione in alto): 🔬 Sandbox (simulazione, pagina iniziale) e 📁 Dati (connettoma e stimoli
dell'utente, nella cartella ``~/drosophila_data``).
"""

import streamlit as st

from sandbox.ui import look, state
from sandbox.ui.pages import DATA, SIMULATION

st.set_page_config(page_title="Drosophila Neural Sandbox", page_icon=look.LOGO, layout="wide")
state.begin_run()
look.apply_style()

st.navigation([st.Page(SIMULATION, title="Sandbox", icon="🔬", default=True),
               st.Page(DATA, title="Dati", icon="📁")], position="top").run()
