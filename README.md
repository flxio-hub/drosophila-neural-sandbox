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
- **Dati dal browser** (pagina Dati): tabelle di FlyWire Codex anche da centinaia di MB (FAFB, Male CNS;
  CSV, CSV.gz, Feather, Parquet), convertite una volta in Parquet in una cartella esterna al codice;
  esploratore del connettoma (classe, superclasse, tipo cellulare, neuropilo, trasmettitore) con
  **"Crea sottocircuito"** verso la Sandbox; libreria di stimoli personalizzati
- **Spiegazioni integrate**: ogni controllo ha un tooltip (cos'è · unità · valore tipico nella mosca), ogni
  grafico e protocollo un riquadro **Cosa sto guardando?** con un esempio di interpretazione; tutti i
  testi sono in un unico file (`sandbox/ui/texts.py`)
- **Esperimenti riproducibili**: salvataggio e caricamento in JSON, leggibili anche da Python
- **Motore**: backend NumPy di riferimento + backend **Numba JIT** (20–150× più veloce), verificati l'uno contro l'altro

---

## 1. Struttura del progetto

```
drosophila_sandbox/
├── app.py                        # Entry point Streamlit: navigazione fra le pagine Sandbox e Dati
├── .streamlit/config.toml        # Tema chiaro/scuro (palette di viz/theme.py), limite di upload 2 GB
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
│   ├── data/
│   │   ├── store.py              # Cartella dati esterna, conversione a blocchi in Parquet, libreria stimoli
│   │   └── codex.py              # Esploratore del connettoma: tabelle Codex/neuPrint, filtri, sottocircuiti
│   ├── network/
│   │   ├── topologies.py         # Circuiti preimpostati (Circuit: W chimica + gap)
│   │   └── connectome.py         # CSV / FlyWire (CAVEclient) / Hemibrain (neuPrint) → W
│   ├── simulation/
│   │   ├── engine.py             # simulate() e Stepper (a blocchi, per l'anello chiuso; set_weights per la plasticità): NumPy ↔ Numba
│   │   ├── fast.py               # Kernel Numba JIT (stessi schemi numerici)
│   │   ├── drive.py              # Correnti, conduttanze esterne, guadagno di rilascio
│   │   └── analysis.py           # Frequenze, ISI, CCG, STA, spettri, Fano, affidabilità
│   ├── viz/                      # Tema validato per daltonismo (chiaro/scuro) e grafici Plotly
│   └── ui/
│       ├── pages/                # simulation.py (Sandbox) e data.py (Dati)
│       ├── texts.py              # TUTTI i testi: tooltip, "Cosa sto guardando?", benvenuto, guida
│       ├── look.py               # Stile, riquadri esplicativi, schermata di benvenuto
│       └── …                     # Pannelli (stato dei widget, stimoli, manipolazioni, protocolli…)
└── tests/                        # 189 test: fisiologia, analisi, protocolli, backend, dati, interfaccia
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

L'app ha due pagine (navigazione in alto): **Sandbox**, dove si simula, e **Dati**, dove si gestiscono
connettoma e stimoli personali. Al primo avvio un pannello di benvenuto spiega in 30 secondi cosa si può fare
(si chiude con *Ho capito* e si riapre con *Introduzione*). Ogni controllo ha un'icona ⓘ con cosa
rappresenta, unità e valore tipico nella mosca; sotto grafici, pannelli e protocolli c'è il riquadro
**Cosa sto guardando?**. Per correggere un testo basta modificare `sandbox/ui/texts.py`
(un test verifica che nessun controllo resti senza spiegazione).

| Scheda della Sandbox | Contenuto |
|---|---|
| **Simulazione** | Stimoli, manipolazioni ed editor delle matrici a sinistra; a destra ① protocollo di stimolo (luce/odore, correnti, conduttanze, temperatura in pannelli separati), ② V(t), ③ raster + PSTH, ④ heatmap di W e gap junction con frequenze (e confronto con il controllo). |
| **Analisi** | Conduttanze e correnti sinaptiche di un neurone, frequenza istantanea, ISI, cross-correlogramma, spike-triggered average, spettro di potenza, confronto dei tracciati con il controllo. |
| **Protocolli** | Esperimenti automatici sull'esperimento corrente, con tabelle esportabili in CSV. |
| **Guida** | Equazioni, unità, esperimenti suggeriti. |

Nella barra laterale: circuito, modello, sinapsi (con STP), simulazione (con rumore di fondo) e
**Salva / carica esperimento**: il file JSON contiene circuito, parametri, stimoli, manipolazioni,
stato di tutti i controlli e il riepilogo dei risultati; ricaricandolo l'app torna esattamente
allo stesso esperimento. **Usa come controllo** salva il risultato corrente per confrontarlo con le
condizioni successive (barre grigie nelle frequenze, tracciati sovrapposti nella scheda Analisi).

### Pagina Dati

I dati dell'utente stanno fuori dal codice, in `~/drosophila_data/` (su Windows
`C:\Users\<nome utente>\drosophila_data`), oppure nella cartella indicata dalla variabile d'ambiente
`DROSOPHILA_DATA`:

```
drosophila_data/
├── connectome/   # tabelle del connettoma convertite in Parquet (una per file caricato)
└── stimuli/      # forme d'onda CSV della libreria (tempo in ms, valore in pA)
```

| Scheda | Contenuto |
|---|---|
| **Tabelle del connettoma** | Caricamento dal browser (fino a 2 GB, `.streamlit/config.toml`) o **da un percorso sul disco** (consigliato per i file più grandi: non passa dal browser e non occupa RAM). CSV, CSV.gz, TSV, Feather/Arrow e Parquet vengono convertiti **una sola volta** in Parquet leggendoli a blocchi; il CSV caricato non viene conservato. Elenco con tipo riconosciuto, anteprima delle prime righe, eliminazione con conferma. |
| **Esploratore del connettoma** | Filtri a cascata per superclasse, classe, sottoclasse, tipo cellulare, trasmettitore, lato e flusso, ricerca per nome/tipo/ID e filtro per neuropilo (sinapsi minime nei neuropili scelti). Metriche e grafici della selezione, tabella dei neuroni più connessi con selezione delle righe. **Crea sottocircuito** porta fino a 150 neuroni nella Sandbox (preset *Sottocircuito dal connettoma*). |
| **Libreria di stimoli** | Forme d'onda salvate, con anteprima: compaiono nel menu *Forma d'onda* di ogni canale di tipo *Da file CSV*. Anche dal canale stesso si può salvare nella libreria un file appena caricato. |

**Quali file di FlyWire Codex** (codex.flywire.ai → Download, dataset FAFB o Male CNS): servono una tabella
di **archi** (es. `connections`: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`) e almeno una di
**neuroni** (es. `classification`: `root_id, super_class, class, sub_class, cell_type…`); altre tabelle per
neurone (nomi, trasmettitori, tipi consolidati) vengono unite per ID. Il tipo di tabella si riconosce
**dalle colonne**, non dal nome del file, e sono accettati anche gli schemi di neuPrint e dei file Feather
del Male CNS di Janelia (`body_pre, body_post, weight`; `bodyId, type, superclass, predictedNt…`).

Per **FAFB v783** bastano `connections_princeton` (archi con soglia ≥ 5 sinapsi, 5,3 milioni di righe),
`classification`, `consolidated_cell_types` (il tipo cellulare: in v783 non è in `classification`), `neurons`
(trasmettitore) e `names`; `visual_neuron_types` aggiunge i tipi del lobo ottico. Le versioni
`…_no_threshold` (22 milioni di righe) includono anche le connessioni con 1–4 sinapsi. Non servono le tabelle
di singole sinapsi (`synapse_coordinates`, `fafb_v783_princeton_synapse_table`, file da molti GB), che
l'app riconosce come tipo **sinapsi** e non usa, né `coordinates`, `labels`, `cell_stats`,
`neuropil_synapse_table`: i neuropili si ricavano dagli archi.

Nel sottocircuito |W| = numero di sinapsi × *nS per sinapsi* (sommando i neuropili) e il segno segue il
trasmettitore del neurone presinaptico: ACh eccitatoria; GABA, glutammato e istamina inibitori; dopamina,
serotonina, octopamina e tiramina trattati come eccitatori deboli (sono neuromodulatori). I neuroni senza
connessioni sopra soglia con gli altri vengono scartati (con avviso). Le sorgenti (neuroni senza input nel
sottocircuito) ricevono uno stimolo di default.

Prestazioni misurate su questo PC (8 GB di RAM) con i file reali di FAFB v783: conversione di
`connections_princeton.csv.gz` (68 MB, 5,3 milioni di righe) in 9 s e della versione senza soglia (275 MB,
22 milioni di righe) in 37 s; statistiche per 139 255 neuroni e 79 neuropili in 11 s (41 s senza soglia),
solo la prima volta per sessione, con un picco di 0,7 GB di RAM; filtri e creazione di un sottocircuito
< 1 s. Esempio: classe `olfactory` + `ALPN`, ricerca «DA1», neuropilo `AL_R` → 133 neuroni (ORN, 7 lPN
colinergici, 1 vPN GABAergico) che nella Sandbox si simulano in circa 1 s.

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

- **Connettoma completo**: carica le tabelle di Codex nella pagina 📁 Dati e crea sottocircuiti
  dall'esploratore (vedi sopra). Il resto di questa sezione riguarda il preset *📂 Connettoma reale*, che
  legge un singolo file di archi o interroga i server.
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
- Un sottocircuito del connettoma è un modello ridotto: le sinapsi con i neuroni esclusi vengono ignorate e
  i neuromodulatori sono trattati come eccitatori.
- Passando alla pagina Dati i controlli della Sandbox conservano i loro valori, ma le modifiche fatte a mano
  nelle tabelle W e gap junction si perdono (Streamlit non permette di ripristinarle): prima esporta
  l'esperimento in JSON.
