"""Parametri configurabili della Sandbox.

Tutto il progetto usa un unico sistema di unità coerente, scelto in modo che i
valori fisiologici di Drosophila (neuroni piccoli, alta resistenza d'ingresso)
siano numeri "comodi":

    tempo [ms] · potenziale [mV] · corrente [pA] · conduttanza [nS]
    resistenza [GΩ] · capacità [pF]

Relazioni che ne derivano (e che il motore sfrutta senza fattori di conversione):
    GΩ · pA = mV      nS · mV = pA      ms / GΩ = pF      pF / nS = ms
"""

from dataclasses import dataclass


@dataclass
class LIFParams:
    """Leaky Integrate-and-Fire con soglia adattiva.

    Valori di default nell'ordine di grandezza dei neuroni di proiezione del
    lobo antennale (R_in ≈ 1 GΩ, τ_m ≈ 10–20 ms).
    """

    v_rest: float = -60.0      # potenziale di riposo (mV)
    v_thresh: float = -45.0    # soglia di scarica di base (mV)
    v_reset: float = -65.0     # potenziale dopo lo spike (mV)
    tau_m: float = 20.0        # costante di tempo di membrana (ms)
    r_m: float = 1.0           # resistenza di membrana (GΩ)
    t_ref: float = 2.0         # periodo refrattario assoluto (ms)
    theta_jump: float = 2.0    # incremento di soglia per spike (mV); 0 = LIF classico
    tau_theta: float = 100.0   # decadimento dell'adattamento di soglia (ms)
    v_peak: float = 20.0       # picco dello spike: disegnato nel tracciato e usato dalle gap junction
    spike_width: float = 1.0   # durata equivalente dello spike (ms): carica trasmessa dalle gap junction


@dataclass
class HHParams:
    """Hodgkin-Huxley classico (cinetiche dell'assone di calamaro, V_rest ≈ -65 mV).

    Le densità (per cm²) vengono convertite in valori assoluti tramite l'area di
    membrana: con 1000 µm² si ottiene C = 10 pF, g_L = 3 nS, dimensioni
    compatibili con un piccolo neurone d'insetto.
    """

    area_um2: float = 1000.0   # area di membrana (µm²)
    c_m: float = 1.0           # capacità specifica (µF/cm²)
    g_na: float = 120.0        # conduttanza massima Na⁺ (mS/cm²)
    g_k: float = 36.0          # conduttanza massima K⁺ (mS/cm²)
    g_l: float = 0.3           # conduttanza di leak (mS/cm²)
    e_na: float = 50.0         # potenziale di inversione Na⁺ (mV)
    e_k: float = -77.0         # potenziale di inversione K⁺ (mV)
    e_l: float = -54.387       # potenziale di inversione leak (mV)
    v_init: float = -65.0      # potenziale iniziale (mV)
    spike_detect: float = 0.0  # soglia di rilevamento dello spike (attraversamento verso l'alto, mV)


@dataclass
class SynapseParams:
    """Sinapsi chimiche a conduttanza con decadimento esponenziale.

    In Drosophila l'acetilcolina (nAChR) è il principale trasmettitore
    eccitatorio; GABA e glutammato (tramite GluCl) sono in prevalenza inibitori.
    """

    tau_exc: float = 5.0       # decadimento conduttanza eccitatoria (ms)
    tau_inh: float = 10.0      # decadimento conduttanza inibitoria (ms)
    e_exc: float = 0.0         # potenziale di inversione eccitatorio (mV)
    e_inh: float = -75.0       # potenziale di inversione inibitorio (mV)
    delay: float = 1.0         # ritardo sinaptico + assonale (ms)


@dataclass
class STPParams:
    """Plasticità a breve termine (modello di Tsodyks-Markram), per neurone presinaptico.

    A ogni spike viene rilasciata la frazione u·x delle risorse disponibili x;
    x recupera verso 1 con τ_rec (depressione), u decade verso U con τ_fac
    (facilitazione; 0 = assente). L'efficacia u·x/U vale 1 per il primo spike
    dopo un lungo silenzio, quindi |W| resta la conduttanza di picco "a riposo".
    La sinapsi ORN → PN di Drosophila è fortemente depressiva (Kazama & Wilson 2008).
    """

    enabled: bool = False
    U: float = 0.5             # frazione di rilascio a riposo (0–1)
    tau_rec: float = 400.0     # recupero dalla depressione (ms)
    tau_fac: float = 0.0       # decadimento della facilitazione (ms); 0 = solo depressione


@dataclass
class SimParams:
    duration: float = 500.0    # durata della simulazione (ms)
    dt: float = 0.1            # passo di integrazione (ms)
    seed: int = 0              # seme per le sorgenti stocastiche
    noise_sigma: float = 0.0   # rumore di membrana di fondo su tutti i neuroni (pA, deviazione standard)
    noise_tau: float = 5.0     # tempo di correlazione del rumore di fondo (ms)
