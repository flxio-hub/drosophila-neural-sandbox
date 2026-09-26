# 🔬 Drosophila Neural Sandbox

Applicazione web interattiva per configurare, simulare e perturbare circuiti neuronali di
*Drosophila melanogaster*: stimoli personalizzati, manipolazioni genetiche e farmacologiche,
protocolli elettrofisiologici automatici e analisi, con risposta in tempo reale.

- **Modelli**: LIF con soglia adattiva · Hodgkin-Huxley (Na⁺/K⁺/leak)
- **Sinapsi**: chimiche a conduttanza (eccitatorie/inibitorie, ritardo), **gap junction** e
  **plasticità a breve termine** (depressione/facilitazione di Tsodyks-Markram)
- **12 tipi di stimolo**: onda quadra/treni, rampa, scala di gradini, onde periodiche, chirp (ZAP),
  rumore bianco/OU, barra in movimento, equazione custom sicura, forma d'onda da CSV, stimoli sensoriali
  (fotorecettori, ORN, looming) e stimoli a **conduttanza**: input sinaptico di Poisson e **optogenetica**
  (CsChrimson, GtACR1)
- **Manipolazioni**: Kir2.1, TNT, ablazione, **termogenetica** (dTrpA1, shibire^ts con protocollo di
  temperatura), farmacologia (picrotossina, mecamilamina, blocco delle gap junction), rumore di fondo
- **Protocolli automatici**: curva f-I, esplorazione parametrica 1D/2D, trial ripetuti (Fano, affidabilità),
  paired-pulse, impedenza di membrana, voltage clamp di Hodgkin-Huxley
- **Analisi**: conduttanze e correnti sinaptiche, frequenza istantanea, ISI, cross-correlogrammi,
  spike-triggered average, spettro di potenza, confronto con un controllo
- **Circuiti**: cascata, lobo antennale, Giant Fiber (sinapsi miste), CPG half-center, rete E/I,
  corpo fungo (codifica sparsa), selettività di direzione T4, connettomi reali (FlyWire, Hemibrain)
- **Esperimenti riproducibili**: salvataggio e caricamento in JSON, leggibili anche da Python
- **Motore**: backend NumPy di riferimento + backend **Numba JIT** (20–150× più veloce), verificati l'uno contro l'altro

---

## 1. Struttura del progetto

```
drosophila_sandbox/
├── app.py                        # Entry point Streamlit: layout e orchestrazione
├── requirements.txt              # Dipendenze principali (numba consigliato)
├── requirements-connectome.txt   # Opzionali: caveclient, neuprint-python, navis
├── pyproject.toml                # Metadati + configurazione pytest
├── data/
│   └── example_mushroom_body.csv # Connettoma dimostrativo (sintetico) in formato FlyWire Codex
├── sandbox/
│   ├── config.py                 # Parametri (LIF, HH, sinapsi, STP, simulazione) + unità
│   ├── experiment.py             # Experiment: preparazione, esecuzione, trial, JSON
│   ├── manipulations.py          # Kir2.1, TNT, ablazione, dTrpA1, shibire^ts, farmaci
│   ├── protocols.py              # f-I, sweep, trial, paired-pulse, ZAP, voltage clamp
│   ├── stimuli/
│   │   ├── waveforms.py          # Forme d'onda in corrente
│   │   ├── conductance.py        # Poisson a conduttanza, opsine
│   │   ├── sensory.py            # Luce/odore/looming → corrente di trasduzione
│   │   ├── expression.py         # Parser sicuro (AST whitelist) per equazioni I(t)
│   │   └── builder.py            # Specifica JSON di un canale → segnali
│   ├── models/
│   │   ├── lif.py                # LIF adattivo (Eulero esponenziale, spikelet elettrici)
│   │   ├── hodgkin_huxley.py     # HH con gate Rush–Larsen tabulati
│   │   └── synapses.py           # Sinapsi a conduttanza, ritardo, STP, guadagno di rilascio
│   ├── network/
│   │   ├── topologies.py         # Circuiti preimpostati (Circuit: W chimica + gap)
│   │   └── connectome.py         # CSV / FlyWire (CAVEclient) / Hemibrain (neuPrint) → W
│   ├── simulation/
│   │   ├── engine.py             # simulate(): dispatch NumPy ↔ Numba
│   │   ├── fast.py               # Kernel Numba JIT (stessi schemi numerici)
│   │   ├── drive.py              # Correnti, conduttanze esterne, guadagno di rilascio
│   │   └── analysis.py           # Frequenze, ISI, CCG, STA, spettri, Fano, affidabilità
│   ├── viz/                      # Tema validato per daltonismo (chiaro/scuro) e grafici Plotly
│   └── ui/                       # Pannelli Streamlit (stato dei widget, stimoli, protocolli…)
└── tests/                        # 139 test: fisiologia, analisi, protocolli, backend, interfaccia
```

La libreria `sandbox/` non dipende da Streamlit (tranne `ui/`): si usa anche da notebook o script.

```python
from sandbox.experiment import Experiment, run_experiment
from sandbox.network import PRESETS
from sandbox.protocols import parameter_sweep
from sandbox.simulation import summary_table

c = PRESETS["Corpo fungo: codifica sparsa"]()
exp = Experiment(neurons=c.neurons, weights=c.weights, gap=c.gap, stimuli=c.default_stimuli,
                 manipulations={"ablated": ["APL"]})
result, inputs = run_experiment(exp)
print(summary_table(result))

# come cambia la risposta di ogni neurone scalando tutte le sinapsi chimiche?
table = parameter_sweep(exp, "Moltiplicatore pesi chimici (×)", [0.5, 1.0, 1.5], "Numero di spike")
```

---

## 2. Installazione ed esecuzione

Requisiti: **Python 3.10 – 3.14**.

### Windows (PowerShell)

```powershell
cd drosophila_sandbox
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # se bloccato: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd drosophila_sandbox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L'app si apre su <http://localhost:8501>. Al primo avvio Numba compila il motore (qualche secondo);
la compilazione viene salvata su disco e i riavvii successivi sono immediati.

### Test

```bash
python -m pytest
```

### Connettomi reali (opzionale)

```bash
pip install -r requirements-connectome.txt
```

---

## 3. Come è organizzata l'interfaccia

| Scheda | Contenuto |
|---|---|
| **🔬 Simulazione** | Stimoli, manipolazioni ed editor delle matrici a sinistra; a destra ① protocollo di stimolo (luce/odore, correnti, conduttanze, temperatura in pannelli separati), ② V(t), ③ raster + PSTH, ④ heatmap di W e gap junction con frequenze (e confronto con il controllo). |
| **📈 Analisi** | Conduttanze e correnti sinaptiche di un neurone, frequenza istantanea, ISI, cross-correlogramma, spike-triggered average, spettro di potenza, confronto dei tracciati con il controllo. |
| **🧪 Protocolli** | Esperimenti automatici sull'esperimento corrente, con tabelle esportabili in CSV. |
| **📖 Guida** | Equazioni, unità, esperimenti suggeriti. |

Nella barra laterale: circuito, modello, sinapsi (con STP), simulazione (con rumore di fondo) e
**💾 Salva / carica esperimento**: il file JSON contiene circuito, parametri, stimoli, manipolazioni,
stato di tutti i controlli e il riepilogo dei risultati; ricaricandolo l'app torna esattamente
allo stesso esperimento. **📌 Usa come controllo** salva il risultato corrente per confrontarlo con le
condizioni successive (barre grigie nelle frequenze, tracciati sovrapposti nella scheda Analisi).

---

## 4. Sistema di unità e convenzioni

| grandezza | unità | relazioni usate dal motore |
|---|---|---|
| tempo | ms | GΩ · pA = mV |
| potenziale | mV | nS · mV = pA |
| corrente | pA | ms / GΩ = pF |
| conduttanza | nS | pF / nS = ms |

I default LIF (R_m = 1 GΩ, τ_m = 20 ms → C_m = 20 pF, reobase 15 pA) sono nell'ordine di grandezza dei
neuroni di proiezione del lobo antennale. Hodgkin-Huxley usa le cinetiche classiche del calamaro
riscalate su 1000 µm² (C_m = 10 pF): la reobase è ~65 pA, per questo in modalità HH il **guadagno
delle correnti** parte da 3×.

- **Sinapsi chimiche**: `W[pre, post]` in nS, righe = presinaptico. Positivo = eccitatorio (ACh),
  negativo = inibitorio (GABA, Glu via GluCl). |W| è la conduttanza di picco di un singolo spike "a riposo".
- **Gap junction**: matrice simmetrica in nS, I = G·(V_vicino − V). Con il LIF lo spike attraversa la
  giunzione come *spikelet* (carica G·(V_picco − V)·durata dello spike).
- **STP**: efficacia di ogni spike u·x/U (1 per il primo spike dopo un lungo silenzio).
- **Stimoli a conduttanza**: g(t) con il proprio potenziale di inversione; cambiano anche la resistenza di
  membrana (shunting), a differenza delle correnti.

---

## 5. Esperimenti suggeriti (verificati con il simulatore)

| Circuito | Cosa provare |
|---|---|
| **Cascata feed-forward** | Abbassa il peso da 3 a 2 nS: la trasmissione diventa probabilistica e la frequenza si attenua stadio dopo stadio. TNT, Kir2.1 o ablazione su N2 fermano la propagazione. |
| **Lobo antennale** | Picrotossina al 100%: PN_B, soppresso dall'inibizione laterale, torna a rispondere. STP sugli ORN: la risposta dei PN diventa più debole e transitoria. |
| **Giant Fiber** | TTMn segue la GF in ~0,1 ms grazie alle gap junction; bloccale (come nei mutanti shakB²) e la risposta rapida scompare. Con l/v = 80 ms la GF scatta ~85 ms prima della collisione, con 20 ms ~9 ms prima, con 10 ms troppo tardi. |
| **Half-center (CPG)** | I due lati alternano burst di ~150 ms. Con Δθ = 0 un lato silenzia l'altro; con rumore identico sui due lati si sincronizzano. |
| **Corpo fungo** | Ogni odore attiva ~6 KC su 30, con insiemi quasi disgiunti per odori simili. Abla l'APL: le KC attive raddoppiano e la sovrapposizione fra odori aumenta. |
| **T4** | La barra → attiva solo T4_dx, la barra ← solo T4_sx (DSI = 1). Abla CT1_3 e T4_dx risponde anche alla direzione nulla; con 30–100 ms fra le colonne la DSI scende a 0,3–0,16. |
| **Rete E/I** | Varia |w_I| e il moltiplicatore dei pesi; con *Trial ripetuti* e rumore di fondo misuri la variabilità della rete. |

### Protocolli

| Protocollo | Cosa misura | Verifica nei test |
|---|---|---|
| Curva f-I | Reobase, frequenza iniziale/a regime, adattamento | LIF contro la soluzione analitica (±5%) |
| Esplorazione parametrica | Qualsiasi misura al variare di 1–2 parametri (globali, di ogni stimolo, di ogni sinapsi) | Soglia di trasmissione della cascata |
| Trial ripetuti | PSTH ± SEM, fattore di Fano, affidabilità di Schreiber; "frozen noise" | Affidabilità = 1 senza rumore |
| Paired-pulse | Rapporto paired-pulse e risposta a treni con la STP scelta | Tsodyks-Markram in forma chiusa |
| Impedenza (ZAP) | \|Z(f)\|, fase, frequenza di risonanza | LIF passivo contro il filtro RC teorico (±10%) |
| Voltage clamp (HH) | Correnti Na⁺/K⁺, curve I-V | Inversione di I_Na oltre E_Na |

### Equazioni custom

Variabili: `t` (ms, array), `dt`, `T`. Esempi:

```
15 * np.sin(2 * np.pi * 0.05 * t) * (t > 20)       # 50 Hz dopo 20 ms
40 * alpha(t, 100, 10) + 20 * rect(t, 300, 350)     # funzione alfa + finestra
where(t < 250, 25, 25 + noise(8))                   # rumore solo nella seconda metà
30 * pulses(t, 40, 3) * rect(t, 100, 400)           # treno a 40 Hz, impulsi da 3 ms
```

L'espressione è analizzata con `ast` e confrontata con una whitelist **prima** della valutazione:
niente import, attributi (tranne `np.<funzione ammessa>`), stringhe, lambda, indicizzazione o kwargs;
gli interi diventano float così che `9**9**9` fallisca subito invece di bloccare il server.

### Connettomi reali

- **CSV**: scarica `connections.csv` da [FlyWire Codex](https://codex.flywire.ai) (colonne
  `pre_root_id, post_root_id, neuropil, syn_count, nt_type`) oppure esporta da neuPrint. Le righe dei
  diversi neuropili vengono sommate e il segno segue il neurotrasmettitore predetto.
- **FlyWire live**: richiede un token CAVE salvato con `CAVEclient().auth.save_token(...)`.
- **Hemibrain live**: richiede il token del tuo profilo su neuprint.janelia.org (non viene mai salvato
  nei file esportati). Hemibrain v1.2.1 non include predizioni di neurotrasmettitore.

> Le vie live (FlyWire, Hemibrain) dipendono da servizi esterni e credenziali personali: il codice
> segue le API pubbliche di `caveclient` e `neuprint-python` ma non è coperto dai test automatici.

---

## 6. Limiti

- Parametri cellulari identici per tutti i neuroni del circuito (estendibile con array per neurone).
- HH usa canali generici: per modelli specifici di Drosophila vedi Günay et al. 2015 (motoneuroni).
- Trasduzione sensoriale fenomenologica (Hill + adattamento a due costanti di tempo); cinetiche delle
  opsine indicative.
- dTrpA1 e shibire^ts seguono istantaneamente la temperatura (in vivo servono secondi o minuti).
- La STP è per neurone presinaptico (uguale su tutte le sue sinapsi); niente plasticità a lungo termine (STDP).
