"""Scheda "Guida": equazioni del modello, unità ed esperimenti suggeriti."""

import streamlit as st

EXPERIMENTS = [
    ("Cascata feed-forward", "Abbassa il peso da 3 a 2 nS: la trasmissione diventa probabilistica e la frequenza "
     "si attenua stadio dopo stadio. Con *Esplorazione parametrica* su «W: N1 → N2» trovi la soglia di trasmissione."),
    ("Feedback inibitorio (lobo antennale)", "Alza l'intensità di S2 (ORN_B): l'LN GABAergico redistribuisce "
     "l'attività. Applica picrotossina al 100%: PN_B, prima soppresso, torna a rispondere. Attiva la STP sugli ORN "
     "(depressione ORN → PN): la risposta dei PN diventa più debole e transitoria."),
    ("Sensorio-motorio: fuga della Giant Fiber", "TTMn segue la GF in ~0,1 ms grazie alle gap junction; bloccale "
     "(*Manipolazioni* → gap junction 100%, come nei mutanti shakB²) e la risposta rapida scompare. Varia l/v del "
     "looming: con l/v = 80 ms la GF scatta ~85 ms prima della collisione, con 20 ms appena ~9 ms prima, con "
     "10 ms troppo tardi."),
    ("Oscillatore half-center (CPG)", "Con i default i due lati alternano burst di ~150 ms. Porta Δθ a 0: un lato "
     "silenzia l'altro. Disattiva «Realizzazione indipendente» nel rumore: con input identico i lati si "
     "sincronizzano. Lo *Spettro di potenza* (scheda Analisi) mostra il ritmo."),
    ("Rete ricorrente E/I casuale", "Varia |w_I| e il moltiplicatore dei pesi; osserva frequenza media, CV degli ISI "
     "e sincronia nel PSTH. Con *Trial ripetuti* e rumore di fondo misuri la variabilità della rete."),
    ("Corpo fungo: codifica sparsa", "Ogni odore attiva solo ~6 cellule di Kenyon su 30, e insiemi quasi disgiunti per "
     "odori simili. Abla l'APL: le KC attive raddoppiano e la sovrapposizione fra odori aumenta (peggiore "
     "separazione dei pattern)."),
    ("Selettività di direzione (T4)", "La barra → attiva solo T4_dx, la barra ← solo T4_sx. Abla CT1_3: T4_dx "
     "risponde anche alla direzione nulla. Porta l'intervallo fra le colonne da 15 a 30–60 ms in entrambi i canali "
     "(barre lente): l'inibizione arriva troppo presto e la selettività si perde."),
]

TOOLS = [
    ("Optogenetica", "CsChrimson (E_rev 0 mV) depolarizza, GtACR1 (canale Cl⁻, E_rev −65 mV) inibisce anche per "
     "shunting. Prova GtACR1 sulla GF durante il looming."),
    ("Termogenetica", "dTrpA1 attiva i neuroni sopra ~27 °C; shibire^ts blocca il loro rilascio sopra ~29,5 °C. "
     "Il protocollo di temperatura compare nel grafico ①."),
    ("Kir2.1 / TNT / ablazione", "Tre modi di togliere un neurone: silenziarlo (Kir2.1: non scarica), renderlo muto "
     "(TNT: scarica ma non trasmette) o rimuoverlo."),
    ("Stimolo di Poisson a conduttanza", "Bombardamento sinaptico di fondo: aumenta la conduttanza totale e "
     "accorcia la costante di tempo efficace (stato ad alta conduttanza)."),
    ("Rumore gaussiano + STA", "Inietta rumore bianco o OU in un neurone e calcola la spike-triggered average "
     "(scheda Analisi): ottieni il filtro che il neurone applica all'input."),
    ("Controllo", "📌 salva il risultato corrente come controllo: frequenze, tracciati e differenze vengono "
     "confrontati automaticamente."),
]


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
