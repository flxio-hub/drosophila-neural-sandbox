"""Scheda "Guida": equazioni del modello, unità ed esperimenti suggeriti."""

import streamlit as st

from .texts import EXPERIMENTS, TOOLS



def guide_panel(model: str) -> None:
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Equazioni")
        if model == "lif":
            st.markdown("**LIF con soglia adattiva** (Eulero esponenziale, esatto a conduttanze costanti)")
            st.latex(r"C_m\frac{dV}{dt} = -\frac{V-V_{rest}}{R_m} - g_e(V-E_e) - g_i(V-E_i) "
                     r"- \sum_k g_k(V-E_k) + I_{ext},\quad C_m=\frac{\tau_m}{R_m}")
            st.latex(r"V \ge V_{th}+\theta \;\Rightarrow\; V\to V_{reset},\;\; \theta\to\theta+\Delta\theta,"
                     r"\qquad \tau_\theta\frac{d\theta}{dt}=-\theta")
        else:
            st.markdown("**Hodgkin-Huxley** (gate con Rush–Larsen tabulato, V con Eulero esponenziale)")
            st.latex(r"C_m\frac{dV}{dt} = -\bar g_{Na}m^3h(V-E_{Na}) - \bar g_K n^4(V-E_K) - g_L(V-E_L)"
                     r" - I_{syn} - \sum_k g_k(V-E_k) + I_{ext}")
            st.latex(r"\frac{dx}{dt} = \alpha_x(V)(1-x) - \beta_x(V)\,x,\qquad x\in\{m,h,n\}")
        st.markdown("**Sinapsi chimiche** con ritardo $d$ ed efficacia $e_k$ (plasticità, blocco del rilascio)")
        st.latex(r"\frac{dg_{e,i}^{(post)}}{dt} = -\frac{g_{e,i}}{\tau_{e,i}}"
                 r" + \sum_{pre}|W_{pre,post}|\sum_k e_k\,\delta(t-t^{pre}_k-d)")
        st.markdown("**Plasticità a breve termine** (Tsodyks-Markram), per ogni neurone presinaptico")
        st.latex(r"e_k = \frac{u_k x_k}{U},\quad x \leftarrow x - u x,\quad u \leftarrow u + U(1-u),"
                 r"\quad \frac{dx}{dt}=\frac{1-x}{\tau_{rec}},\ \frac{du}{dt}=\frac{U-u}{\tau_{fac}}")
        st.markdown("**Gap junction** (simmetriche)")
        st.latex(r"I^{gap}_i = \sum_j G_{ij}\,(V_j - V_i)")
        st.markdown("Unità: ms · mV · pA · nS · GΩ · pF  (GΩ·pA = mV, nS·mV = pA, pF/nS = ms)")
    with col_b:
        st.subheader("Esperimenti suggeriti")
        for name, text in EXPERIMENTS:
            st.markdown(f"**{name}** — {text}")
        st.subheader("Strumenti")
        for name, text in TOOLS:
            st.markdown(f"**{name}** — {text}")
