"""Tutti i testi esplicativi dell'interfaccia, in un unico file: per correggerli basta modificare qui.

* ``TIPS``     — tooltip brevi (icona ⓘ accanto ai controlli): cos'è · unità · valore tipico nella mosca.
  La chiave è il nome logico del widget (quello passato agli helper di ``state``) senza firma del circuito
  né numero del canale; ``state.tip_keys`` mostra quali chiavi vengono provate, dalla più specifica alla
  più generica. Esempi: ``"lif_tau_m"``, ``"sq_amp"`` (stimolo Onda quadra), ``"se_amp"`` (Sensoriale),
  ``"ton"`` (inizio di qualunque stimolo), ``"man_kir"`` (manipolazione).
* ``EXPLAIN``  — riquadri "Cosa sto guardando?": (spiegazione per uno studente, esempio di interpretazione).
* ``WELCOME``  — schermata iniziale; ``KIND_HELP``, ``PROTOCOLS``, ``EXPERIMENTS``, ``TOOLS`` — didascalie.

I "valori tipici nella mosca" sono ordini di grandezza indicativi dalla letteratura (es. Wilson & Laurent
2005; Gouwens & Wilson 2009; Kazama & Wilson 2008; Turner et al. 2008; von Reyn et al. 2014): variano
molto fra tipi cellulari. Gli esempi numerici sui circuiti di default sono stati verificati con il simulatore.
"""

# ============================================================================ tooltip dei controlli

TIPS = {
    # ------------------------------------------------------------------ circuito
    "preset": "Circuito da simulare: modelli didattici di circuiti noti della mosca, un connettoma reale (CSV, "
              "FlyWire, Hemibrain) o un sottocircuito creato nella pagina 📁 Dati.",
    "cascade_n": "Neuroni nella catena N1 → N2 → … Ogni stadio aggiunge ritardo sinaptico e tempo di carica.",
    "cascade_w": "Conduttanza di picco di ogni sinapsi della catena. Unità: nS. Nella mosca una connessione "
                 "forte fra due neuroni centrali vale pochi nS (decine di sinapsi).",
    "rand_n": "Numero di neuroni della rete casuale.",
    "rand_p": "Probabilità che esista una sinapsi fra due neuroni qualsiasi (0–1). Le reti locali reali sono "
              "sparse: pochi percento.",
    "rand_frac": "Frazione di neuroni eccitatori (colinergici). Nel cervello della mosca la maggioranza dei neuroni "
                 "è colinergica; GABA e glutammato sono per lo più inibitori.",
    "rand_we": "Peso delle sinapsi eccitatorie. Unità: nS.",
    "rand_wi": "Peso (in valore assoluto) delle sinapsi inibitorie. Unità: nS. L'inibizione è spesso più forte "
               "dell'eccitazione per bilanciare la rete.",
    "rand_seed": "Seme del generatore casuale: stesso seme = stessa rete.",
    "mb_nkc": "Numero di cellule di Kenyon (KC). Nel corpo fungo della mosca sono ~2000 per lato; ogni KC riceve "
              "input da ~5–7 glomeruli (\"claw\").",
    "mb_seed": "Seme della connettività casuale PN → KC: stesso seme = stesse KC collegate agli stessi PN.",
    "cx_source": "Da dove prendere la connettività: l'esempio incluso, un file CSV di archi, oppure una query "
                 "online (servono token e dipendenze opzionali).",
    "cx_syn_to_nS": "Conversione da numero di sinapsi a conduttanza di picco. Unità: nS per sinapsi. Valore "
                    "indicativo: 0,1–0,3 nS per sinapsi colinergica.",
    "cx_min_syn": "Connessioni con meno sinapsi vengono scartate. FlyWire usa spesso ≥ 5 per ridurre i falsi "
                  "positivi della segmentazione automatica.",
    "cx_max_neurons": "Se il file contiene più neuroni si tengono i più connessi. Oltre ~150 neuroni la simulazione "
                      "interattiva rallenta.",
    "cx_ids": "ID dei neuroni (root ID di FlyWire o body ID di neuPrint), separati da spazi o virgole.",

    # ------------------------------------------------------------------ modello neuronale
    "model": "LIF adattivo: veloce, adatto a reti. Hodgkin-Huxley: canali Na⁺/K⁺ espliciti e forma reale del "
             "potenziale d'azione, ma richiede passi più piccoli.",
    "lif_v_rest": "Potenziale di riposo (senza input). Unità: mV. Nella mosca ≈ −50…−65 mV nei neuroni centrali.",
    "lif_v_thresh": "Soglia di scarica di base: quando V la supera parte uno spike. Unità: mV. Nei neuroni di "
                    "proiezione ≈ 10–20 mV sopra il riposo (gli spike nascono nell'assone).",
    "lif_v_reset": "Potenziale a cui torna V dopo lo spike (post-iperpolarizzazione). Unità: mV. Tipicamente "
                   "pochi mV sotto il riposo.",
    "lif_tau_m": "Costante di tempo di membrana τ_m = R_m·C_m: quanto rapidamente V segue l'input. Unità: ms. "
                 "Neuroni di proiezione del lobo antennale ≈ 10–20 ms.",
    "lif_r_m": "Resistenza d'ingresso: 1 pA di corrente sposta V di R_m mV. Unità: GΩ. I neuroni di mosca sono "
               "piccoli e molto resistivi: ≈ 0,5–1 GΩ (PN) fino a diversi GΩ (cellule di Kenyon).",
    "lif_t_ref": "Periodo refrattario assoluto: dopo uno spike il neurone non può scaricare. Unità: ms. "
                 "Tipico 1–3 ms (limita la frequenza massima a qualche centinaio di Hz).",
    "lif_theta_jump": "Aumento della soglia a ogni spike (adattamento di frequenza). Unità: mV. 0 = LIF classico "
                      "senza adattamento; molti neuroni di mosca si adattano durante uno stimolo prolungato.",
    "lif_tau_theta": "Tempo con cui l'aumento di soglia si esaurisce. Unità: ms. Decine–centinaia di ms.",
    "hh_area": "Area della membrana: converte le densità (per cm²) in valori assoluti. Unità: µm². Con 1000 µm² "
               "C_m = 10 pF, ordine di grandezza di un piccolo neurone d'insetto.",
    "hh_g_na": "Conduttanza massima dei canali Na⁺ (densità). Unità: mS/cm². 120 = valore classico del calamaro; "
               "nella mosca i canali Na⁺ (para) sono concentrati nella zona d'innesco dell'assone.",
    "hh_g_k": "Conduttanza massima dei canali K⁺ delayed rectifier (Shab/Shaw nella mosca). Unità: mS/cm². "
              "Classico: 36.",
    "hh_g_l": "Conduttanza di dispersione (leak), fissa. Unità: mS/cm². Classico: 0,3. Più alta = neurone meno "
              "eccitabile e più rapido.",
    "hh_e_na": "Potenziale di inversione del Na⁺ (Nernst). Unità: mV. ≈ +50 mV.",
    "hh_e_k": "Potenziale di inversione del K⁺. Unità: mV. ≈ −75…−90 mV.",
    "hh_e_l": "Potenziale di inversione del leak: determina il riposo. Unità: mV.",

    # ------------------------------------------------------------------ sinapsi
    "syn_tau_exc": "Decadimento della conduttanza eccitatoria (nAChR). Unità: ms. Nella mosca le correnti "
                   "colinergiche rapide durano pochi ms.",
    "syn_tau_inh": "Decadimento della conduttanza inibitoria (GABA_A/Rdl, GluCl). Unità: ms. Tipico 5–20 ms "
                   "(le risposte GABA_B, lente, non sono modellate).",
    "syn_e_exc": "Potenziale di inversione eccitatorio. Unità: mV. ≈ 0 mV (canali cationici non selettivi).",
    "syn_e_inh": "Potenziale di inversione inibitorio (canali al Cl⁻). Unità: mV. ≈ −60…−75 mV.",
    "syn_delay": "Ritardo fra lo spike presinaptico e l'inizio della conduttanza (conduzione + rilascio). "
                 "Unità: ms. Tipico ~1 ms.",
    "stp_enabled": "Attiva la plasticità a breve termine (Tsodyks-Markram): l'efficacia sinaptica dipende dagli "
                   "spike appena emessi.",
    "stp_U": "Frazione delle risorse rilasciata al primo spike (probabilità di rilascio). 0–1. U alto → forte "
             "depressione; U basso con τ_fac > 0 → facilitazione. ORN → PN: rilascio elevato, depressiva.",
    "stp_tau_rec": "Tempo di recupero delle vescicole dopo la depressione. Unità: ms. Centinaia di ms.",
    "stp_tau_fac": "Tempo di decadimento della facilitazione. Unità: ms. 0 = solo depressione.",
    "stp_targets": "Neuroni presinaptici le cui sinapsi in uscita hanno la STP. Vuoto = tutti.",
    "w_scale": "Moltiplica tutte le sinapsi chimiche (utile per esplorare la stabilità della rete). 1 = pesi "
               "della matrice.",
    "gap_scale": "Moltiplica tutte le gap junction. 1 = valori della matrice.",

    # ------------------------------------------------------------------ simulazione
    "dt_lif": "Passo di integrazione. Unità: ms. Con il LIF 0,1 ms è preciso; passi grandi spostano gli spike.",
    "dt_hh": "Passo di integrazione. Unità: ms. Hodgkin-Huxley richiede ≤ 0,025 ms per risolvere il "
             "potenziale d'azione (~1 ms di durata).",
    "duration": "Durata simulata. Unità: ms. Le risposte sensoriali della mosca si studiano su 0,5–3 s.",
    "seed": "Seme delle sorgenti casuali (rumore, pennacchi d'odore, Poisson): stesso seme = stesso risultato.",
    "noise_sigma": "Deviazione standard del rumore di corrente indipendente su ogni neurone. Unità: pA. Qualche "
                   "pA basta a rendere variabili i trial (fluttuazioni di membrana di 1–2 mV con R_m ≈ 1 GΩ).",
    "noise_tau": "Tempo di correlazione del rumore di fondo. Unità: ms. Valori simili a τ sinaptica (≈ 5 ms).",
    "live": "Ricalcola a ogni modifica. Disattiva per reti molto grandi o simulazioni lunghe.",
    "visible": "Neuroni i cui tracciati V(t) sono disegnati (max 8, per mantenere i colori distinguibili).",
    "psth_bin": "Ampiezza dei bin del PSTH. Unità: ms. Bin piccoli = più dettaglio temporale ma più rumore.",

    # ------------------------------------------------------------------ stimoli: comuni
    "nch": "Numero di canali di stimolo indipendenti (max 8). Ogni canale ha un tipo e i suoi bersagli.",
    "gain": "Scala tutti gli stimoli in corrente (non le conduttanze). 1 = ampiezze indicate. Con "
            "Hodgkin-Huxley serve ~3× perché la reobase è più alta.",
    "kind": "Forma dello stimolo. I tipi \"· conduttanza\" aprono canali (effetto dipendente da V) invece di "
            "iniettare corrente.",
    "targets": "Neuroni che ricevono lo stimolo di questo canale.",
    "ton": "Istante di inizio dello stimolo. Unità: ms dall'inizio della simulazione.",
    "toff": "Istante di fine dello stimolo. Unità: ms.",
    # onda quadra
    "sq_amp": "Ampiezza della corrente iniettata. Unità: pA. Positiva depolarizza. Con i default LIF servono "
              "> 15 pA (reobase) per far scaricare il neurone; nei neuroni di mosca bastano decine di pA.",
    "sq_f": "Frequenza del treno di impulsi. Unità: Hz. 0 = gradino continuo.",
    "sq_w": "Durata di ciascun impulso del treno. Unità: ms.",
    # rampa
    "rp_a0": "Corrente all'inizio della rampa. Unità: pA.",
    "rp_a1": "Corrente alla fine della rampa. Unità: pA.",
    "rp_hold": "Dopo la rampa mantiene il valore finale invece di tornare a 0.",
    # scala di gradini
    "st_start": "Ampiezza del primo gradino. Unità: pA.",
    "st_step": "Incremento fra un gradino e il successivo. Unità: pA.",
    "st_n": "Numero di gradini.",
    "st_len": "Durata di ciascun gradino. Unità: ms.",
    "st_gap": "Pausa senza corrente fra i gradini. Unità: ms.",
    # periodica
    "pe_shape": "Forma d'onda periodica.",
    "pe_amp": "Ampiezza dell'oscillazione attorno all'offset. Unità: pA.",
    "pe_dc": "Componente continua sommata all'onda. Unità: pA.",
    "pe_f": "Frequenza dell'onda. Unità: Hz.",
    "pe_ph": "Fase iniziale. Unità: gradi.",
    # chirp
    "ch_amp": "Ampiezza della sinusoide a frequenza crescente. Unità: pA. Tienila piccola (sottosoglia) per "
              "misurare l'impedenza.",
    "ch_dc": "Componente continua. Unità: pA.",
    "ch_f0": "Frequenza iniziale. Unità: Hz.",
    "ch_f1": "Frequenza finale. Unità: Hz.",
    "ch_method": "Come cresce la frequenza nel tempo: lineare o logaritmica (più tempo alle basse frequenze).",
    # rumore
    "no_mu": "Media della corrente di rumore. Unità: pA.",
    "no_sd": "Deviazione standard del rumore. Unità: pA.",
    "no_tau": "Tempo di correlazione. Unità: ms. 0 = rumore bianco; > 0 = processo di Ornstein-Uhlenbeck, la cui "
              "statistica non dipende da dt.",
    "no_indep": "Ogni bersaglio riceve un rumore diverso. Disattiva per un input identico (correlato) su tutti.",
    # barra in movimento
    "mb_dir": "→ percorre i bersagli nell'ordine in cui li hai selezionati, ← al contrario.",
    "mb_amp": "Corrente data a ogni posizione (colonna visiva) attraversata. Unità: pA.",
    "mb_pulse": "Quanto resta attiva ogni posizione. Unità: ms.",
    "mb_int": "Intervallo fra una posizione e la successiva: velocità = 1 colonna ogni intervallo. Unità: ms. "
              "Nell'occhio della mosca le colonne distano ~5°.",
    "mb_rep": "Numero di passaggi della barra.",
    "mb_repms": "Periodo fra l'inizio di due passaggi. Unità: ms.",
    # equazione e file
    "eq_expr": "Espressione in t (ms) che dà la corrente in pA. Funzioni ammesse sotto il riquadro.",
    "csv_lib": "Forma d'onda da usare: caricala qui oppure scegli uno stimolo della libreria (pagina 📁 Dati).",
    "csv_off": "Sposta la forma d'onda nel tempo. Unità: ms (positivo = più tardi).",
    "csv_scale": "Moltiplica i valori del file (es. −1 per invertire, 0,5 per dimezzare).",
    # sensoriale
    "se_mod": "Tipo di neurone sensoriale bersaglio: fotorecettore, neurone olfattivo o rilevatore di looming. "
              "Ognuno ha la propria trasduzione.",
    "se_prof": "Andamento nel tempo dello stimolo fisico (luce, concentrazione d'odore, oggetto in avvicinamento).",
    "se_amp": "Intensità dello stimolo, normalizzata (0–1).",
    "se_bg": "Livello di fondo prima e dopo l'impulso (0–1): con un fondo il recettore è già adattato.",
    "se_s0": "Livello all'inizio del gradiente (0–1).",
    "se_s1": "Livello alla fine del gradiente (0–1).",
    "se_m": "Livello medio della luce sfarfallante (0–1).",
    "se_c": "Contrasto: ampiezza relativa dello sfarfallio (0–1).",
    "se_f": "Frequenza dello sfarfallio. Unità: Hz. I fotorecettori di Drosophila seguono sfarfallii fino a "
            "diverse decine di Hz.",
    "se_r": "Sbuffi d'odore al secondo (arrivi di Poisson), come in un pennacchio turbolento.",
    "se_w": "Durata media di uno sbuffo. Unità: ms.",
    "se_pk": "Concentrazione massima degli sbuffi (0–1).",
    "se_tc": "Istante in cui l'oggetto raggiungerebbe l'occhio. Unità: ms.",
    "se_lv": "Semi-dimensione / velocità dell'oggetto. Unità: ms. Più piccolo = avvicinamento più rapido. La "
             "Giant Fiber è stata studiata con l/v ≈ 10–80 ms (von Reyn et al. 2014).",
    "se_K": "Semi-saturazione della curva di Hill: intensità che dà metà della risposta massima (0–1).",
    "se_n": "Coefficiente di Hill: pendenza della curva dose-risposta.",
    "se_g": "Corrente massima prodotta dalla trasduzione. Unità: pA.",
    "se_tf": "Costante di tempo della trasduzione (filtro passa-basso). Unità: ms.",
    "se_ts": "Costante di tempo dell'adattamento. Unità: ms. Nella mosca da centinaia di ms a secondi.",
    "se_k": "Forza dell'adattamento (0 = nessuno, 1 = la risposta torna al livello di base).",
    # Poisson
    "po_rec": "Tipo di sinapsi simulata: eccitatoria (ACh → E_exc) o inibitoria (GABA/Glu → E_inh).",
    "po_rate": "Somma delle frequenze di tutti gli ingressi non modellati (es. 100 sinapsi × 2 Hz = 200 Hz). "
               "Unità: Hz.",
    "po_g": "Conduttanza aperta da ogni evento sinaptico. Unità: nS. Una singola sinapsi vale ~0,1–1 nS.",
    "po_tau": "Decadimento della conduttanza di ogni evento. Unità: ms.",
    # optogenetica
    "op_opsin": "Canale attivato dalla luce espresso nei bersagli. CsChrimson (luce rossa, cationico) attiva; "
                "GtACR1 (luce blu-verde, Cl⁻) inibisce.",
    "op_int": "Intensità della luce, normalizzata (0–1).",
    "op_g": "Conduttanza massima dell'opsina a luce saturante. Unità: nS.",
    "op_f": "Frequenza degli impulsi di luce. Unità: Hz. 0 = luce continua.",
    "op_pw": "Durata di ciascun impulso di luce. Unità: ms.",

    # ------------------------------------------------------------------ manipolazioni
    "man_ablated": "Il neurone viene rimosso dal circuito (come un'ablazione con reaper/hid o laser).",
    "man_tnt": "Tossina tetanica (UAS-TNT): taglia la sinaptobrevina, il neurone scarica ma non rilascia "
               "neurotrasmettitore. Le gap junction restano.",
    "man_kir": "Canale K⁺ Kir2.1 sempre aperto: iperpolarizza e abbassa la resistenza, il neurone non scarica.",
    "man_kir_g": "Conduttanza aggiunta da Kir2.1. Unità: nS. Qualche nS basta a silenziare un neurone da 1 GΩ.",
    "man_ek": "Potenziale di inversione del K⁺. Unità: mV. ≈ −80…−90 mV.",
    "man_trpa1": "Canale cationico dTrpA1, aperto dal caldo: attiva i neuroni sopra ~25–27 °C.",
    "man_shi": "Dinamina termosensibile shibire^ts: sopra ~29 °C il riciclo delle vescicole si blocca e la "
               "trasmissione chimica si ferma.",
    "man_tb": "Temperatura permissiva, prima e dopo il riscaldamento. Unità: °C. Tipico 18–22 °C.",
    "man_tt": "Temperatura restrittiva raggiunta durante il riscaldamento. Unità: °C. Tipico 30–32 °C.",
    "man_tr": "Durata del passaggio fra le due temperature. Unità: ms.",
    "man_ton": "Istante di inizio del riscaldamento. Unità: ms.",
    "man_toff": "Istante di fine del riscaldamento. Unità: ms.",
    "man_tg": "Conduttanza massima di dTrpA1 a temperatura alta. Unità: nS.",
    "man_th": "Temperatura di semi-attivazione di dTrpA1. Unità: °C.",
    "man_sh": "Temperatura a cui il blocco di shibire è a metà. Unità: °C.",
    "man_ptx": "Picrotossina: blocca i canali al Cl⁻ (GABA_A/Rdl e GluCl), cioè tutte le sinapsi inibitorie "
               "del modello. 100% = blocco completo.",
    "man_mec": "Mecamilamina: antagonista dei recettori nicotinici (nAChR), riduce l'eccitazione colinergica.",
    "man_gap": "Riduce le gap junction, come nei mutanti di shakB (innexina 8) che perdono le sinapsi elettriche "
               "della Giant Fiber.",

    # ------------------------------------------------------------------ analisi
    "an_cond": "Neurone di cui mostrare conduttanze e correnti ricevute.",
    "an_rate_neurons": "Neuroni di cui stimare la frequenza istantanea (max 8).",
    "an_sigma": "Larghezza del kernel gaussiano che trasforma gli spike in frequenza. Unità: ms. Piccolo = "
                "dettaglio, grande = andamento lento.",
    "an_isi": "Neurone di cui mostrare gli intervalli fra spike consecutivi.",
    "an_isi_bin": "Ampiezza dei bin dell'istogramma. Unità: ms.",
    "an_ccg_a": "Neurone di riferimento: gli spike del bersaglio vengono misurati rispetto ai suoi.",
    "an_ccg_b": "Neurone bersaglio.",
    "an_ccg_w": "Ritardi mostrati, da −finestra a +finestra. Unità: ms.",
    "an_sta": "Neurone di cui mediare il segnale prima di ogni spike.",
    "an_sta_src": "Segnale da mediare prima degli spike: corrente iniettata, sinaptica o potenziale.",
    "an_sta_w": "Quanti ms prima di ogni spike considerare. Unità: ms.",
    "an_psd": "Segnale di cui calcolare lo spettro: attività di popolazione o V di un neurone.",
    "an_ctrl": "Neurone da confrontare fra controllo e condizione attuale.",

    # ------------------------------------------------------------------ protocolli
    "proto_choice": "Esperimento automatico da eseguire sulla configurazione attuale.",
    "proto_fi_lo": "Corrente del primo gradino. Unità: pA.",
    "proto_fi_hi": "Corrente dell'ultimo gradino. Unità: pA.",
    "proto_fi_step": "Distanza fra due gradini. Unità: pA. Più piccolo = reobase più precisa.",
    "proto_fi_len": "Durata di ciascun gradino. Unità: ms. Serve abbastanza tempo per vedere l'adattamento.",
    "sw_p": "Parametro da variare. \"W: A → B\" è il peso di una singola sinapsi.",
    "sw_lo": "Primo valore del parametro.",
    "sw_hi": "Ultimo valore del parametro.",
    "sw_n": "Numero di valori equispaziati fra il primo e l'ultimo.",
    "sw_2d": "Varia anche un secondo parametro: il risultato diventa una mappa di calore.",
    "sw_p2": "Secondo parametro da variare.",
    "sw2_lo": "Primo valore del secondo parametro.",
    "sw2_hi": "Ultimo valore del secondo parametro.",
    "sw2_n": "Numero di valori del secondo parametro.",
    "proto_sw_metric": "Grandezza misurata in ogni simulazione.",
    "proto_sw_show": "Neuroni disegnati nel grafico (max 8).",
    "proto_sw_neuron": "Neurone di cui mostrare la mappa.",
    "proto_tr_n": "Numero di ripetizioni dell'esperimento, ognuna con un seme diverso.",
    "proto_tr_frozen": "Gli stimoli casuali restano identici fra i trial; cambia solo il rumore di fondo "
                       "(paradigma di Mainen & Sejnowski, 1995).",
    "proto_tr_neuron": "Neurone di cui mostrare raster e PSTH dei trial.",
    "proto_tr_bin": "Ampiezza dei bin del PSTH. Unità: ms.",
    "proto_pp_U": "Frazione di rilascio a riposo del banco di prova.",
    "proto_pp_rec": "Recupero dalla depressione. Unità: ms.",
    "proto_pp_fac": "Decadimento della facilitazione. Unità: ms.",
    "proto_pp_w": "Conduttanza di picco della sinapsi di prova. Unità: nS.",
    "proto_pp_sign": "Segno della sinapsi di prova.",
    "proto_pp_hz": "Frequenza del treno di spike presinaptici. Unità: Hz.",
    "proto_pp_n": "Numero di spike nel treno.",
    "proto_pp_isis": "Intervalli fra i due spike delle coppie, in ms, separati da virgole.",
    "proto_z_amp": "Ampiezza del chirp. Unità: pA. Deve restare sottosoglia (nessuno spike).",
    "proto_z_f0": "Frequenza iniziale del chirp. Unità: Hz.",
    "proto_z_f1": "Frequenza finale del chirp. Unità: Hz.",
    "proto_z_T": "Durata del chirp. Unità: ms. Più lungo = spettro più pulito.",
    "proto_z_bias": "Corrente continua che sposta il potenziale di partenza. Unità: pA. La risonanza dipende da V.",
    "proto_vc_hold": "Potenziale imposto prima e dopo i gradini. Unità: mV.",
    "proto_vc_lo": "Primo potenziale di comando. Unità: mV.",
    "proto_vc_hi": "Ultimo potenziale di comando. Unità: mV.",
    "proto_vc_step": "Distanza fra due comandi. Unità: mV.",
    "proto_vc_len": "Durata di ciascun gradino di potenziale. Unità: ms.",
    "proto_vc_show": "Tracciati di corrente disegnati (max 5).",

    # ------------------------------------------------------------------ pagina Dati
    "data_path": "Percorso completo di un file già sul disco (es. nella cartella Download). Per i file più grandi "
                 "è la via consigliata: non passa dal browser e non occupa RAM.",
    "xp_edges": "Tabella di archi (pre → post, numero di sinapsi) da cui calcolare statistiche e sottocircuiti. "
                "Le versioni \"no_threshold\" includono anche connessioni con 1–4 sinapsi (più rumorose).",
    "xp_super_class": "Grande categoria funzionale (es. sensory, central, motor, visual_projection…).",
    "xp_class": "Classe di neuroni (es. ALPN = neuroni di proiezione del lobo antennale, Kenyon_Cell, MBON…).",
    "xp_sub_class": "Suddivisione della classe.",
    "xp_cell_type": "Tipo cellulare: neuroni con morfologia e connettività equivalenti (es. DA1_lPN, KCg-m).",
    "xp_nt": "Neurotrasmettitore predetto dalle immagini EM (Eckstein et al. 2024): decide il segno delle "
             "sinapsi (ACh eccitatoria; GABA, glutammato e istamina inibitori).",
    "xp_side": "Emisfero del corpo cellulare.",
    "xp_flow": "Direzione del flusso d'informazione: afferente (sensoriale), intrinseco, efferente (motorio).",
    "xp_text": "Cerca nel nome, nel tipo cellulare o nell'ID. Più parole separate da virgole: basta che ne "
               "compaia una.",
    "xp_neuropils": "Tiene i neuroni con abbastanza sinapsi nei neuropili scelti (es. AL_L = lobo antennale "
                    "sinistro, MB_CA_L = calice del corpo fungo).",
    "xp_np_min": "Sinapsi minime (ingresso + uscita) nei neuropili scelti.",
    "xp_pick": "Quali neuroni filtrati portare nella Sandbox.",
    "xp_top_n": "Numero di neuroni più connessi (sinapsi in ingresso + uscita) da tenere.",
    "xp_syn_to_nS": "Conversione da numero di sinapsi a conduttanza di picco. Unità: nS per sinapsi. "
                    "Indicativo: 0,1–0,3.",
    "xp_min_syn": "Connessioni più deboli vengono scartate. Codex usa ≥ 5 sinapsi di default.",
    "xp_name": "Nome del circuito nella Sandbox.",
    "stim_preview": "Stimolo della libreria da mostrare.",

    # ------------------------------------------------------------------ pulsanti e caricamenti (non registrati)
    "btn_run": "Esegue la simulazione con i parametri attuali (serve solo con il tempo reale disattivato).",
    "btn_control": "Salva questo risultato per confrontarlo con le condizioni successive.",
    "btn_subcircuit": "Costruisce la matrice W fra i neuroni scelti e apre la Sandbox con il nuovo circuito.",
    "upload_connectome": "CSV o CSV.gz (export di FlyWire Codex), Feather (Male CNS di Janelia) o Parquet. I file "
                         "vengono convertiti una volta in Parquet nella cartella dati.",
    "upload_stimulus": "CSV con due colonne numeriche: tempo (ms) e valore (pA). Separatore automatico.",
    "upload_waveform": "CSV con due colonne numeriche: tempo (ms) e valore (pA). Separatore automatico.",
    "save_to_library": "Copia questa forma d'onda nella libreria: sarà disponibile in tutti i canali.",
    "btn_export": "Circuito, parametri, stimoli, manipolazioni e riepilogo dei risultati in un file JSON.",
    "neuprint_token": "Dal tuo profilo su neuprint.janelia.org (non viene salvato).",
    "metric_exc_share": "|Q_exc| / (|Q_exc| + |Q_inh|): bilancio fra eccitazione e inibizione ricevute.",
    "metric_fi_slope": "Guadagno a regime sopra la reobase (regressione lineare). Unità: Hz/pA.",
    "metric_q": "|Z| al picco / |Z| alla frequenza più bassa. Q ≈ 1: nessuna risonanza (filtro passa-basso).",
    "btn_import_upload": "Converte in Parquet i file caricati qui sopra. Una sola volta: poi si leggono dal disco.",
    "btn_import_path": "Converte il file indicato senza passare dal browser (consigliato oltre ~1 GB).",
    "btn_delete": "Elimina definitivamente il file dalla cartella dati.",
    "btn_save_stimuli": "Salva i CSV caricati nella libreria (vengono controllati: servono due colonne numeriche).",
    "data_preview": "Tabella di cui mostrare le prime righe.",
    "data_confirm": "Conferma per abilitare l'eliminazione.",
    "stim_confirm": "Conferma per abilitare l'eliminazione.",
}

# ============================================================================ "Cosa sto guardando?"
# chiave → (spiegazione, esempio di interpretazione)

EXPLAIN = {
    "stimulus_plot": (
        "Ogni riga è un canale di stimolo (S1, S2…) nel tempo, con i neuroni bersaglio nell'etichetta. Le correnti "
        "sono in **pA** (positive = depolarizzanti); gli stimoli a conduttanza (Poisson, optogenetica) sono in "
        "**nS**, perché il loro effetto dipende da V; gli stimoli sensoriali mostrano anche lo stimolo fisico "
        "normalizzato. Se usi dTrpA1 o shibire compare la temperatura.",
        "Nella cascata di default c'è un gradino di 30 pA su N1 fra 50 e 400 ms. Con R_m = 1 GΩ porterebbe V "
        "a −60 + 30 = −30 mV, sopra la soglia di −45 mV: per questo N1 scarica per tutta la durata del gradino.",
    ),
    "voltage_plot": (
        "Il potenziale di membrana **V(t)** in mV dei neuroni scelti. Nel LIF lo spike è disegnato come un picco "
        "verticale e la linea tratteggiata è la soglia di base; con Hodgkin-Huxley vedi la vera forma del "
        "potenziale d'azione. Fra uno spike e l'altro si legge l'integrazione degli input: salite = "
        "eccitazione, discese = inibizione o ritorno al riposo.",
        "Nella cascata di default N1 raggiunge la soglia ~14 ms dopo l'inizio del gradino (primo spike a 63,8 "
        "ms) e ogni stadio successivo segue con ~3,6 ms di ritardo: 1 ms di ritardo sinaptico più il tempo per "
        "caricare la membrana fino alla soglia.",
    ),
    "raster_plot": (
        "In alto il **raster**: un trattino per ogni spike, una riga per neurone. In basso il **PSTH**: spike di "
        "tutta la popolazione per bin, convertiti in frequenza media (Hz per neurone). Il raster mostra *chi* "
        "scarica e *quando*; il PSTH l'andamento complessivo.",
        "Nella cascata di default i trattini di N1…N5 formano diagonali: la stessa sequenza si propaga con ~3,6 ms "
        "di ritardo per stadio. Righe vuote = neuroni silenti; colonne allineate fra neuroni = sincronia.",
    ),
    "connectivity": (
        "La matrice **W[pre, post]** effettivamente simulata (moltiplicatori e manipolazioni inclusi): riga = "
        "neurone presinaptico, colonna = postsinaptico. Rosso = eccitatoria, blu = inibitoria, intensità = "
        "conduttanza di picco in nS. La scheda *Gap junction* mostra le sinapsi elettriche (simmetriche).",
        "Nella cascata la diagonale sopra la principale è rossa (N1 → N2, N2 → N3…). Nel lobo antennale la riga "
        "dell'LN è blu: l'interneurone GABAergico inibisce i PN. Dopo un'ablazione riga e colonna del neurone "
        "diventano grigie.",
    ),
    "rates": (
        "Frequenza media di scarica di ogni neurone in Hz, calcolata su tutta la simulazione. Se hai salvato un "
        "controllo (📌) le barre grigie mostrano i valori di riferimento accanto alla condizione attuale.",
        "Con TNT su N2 nella cascata, N1 e N2 restano a 26 Hz mentre N3–N5 scendono a 0: N2 scarica ma non "
        "trasmette. Con l'ablazione di N2 va a 0 anche N2.",
    ),
    "stimuli": (
        "Uno stimolo **in corrente** (pA) inietta carica indipendentemente da V, come un elettrodo. Uno stimolo "
        "**a conduttanza** (nS) apre canali con un proprio potenziale di inversione: spinge V verso quel valore "
        "e abbassa la resistenza della membrana (shunting). Gli stimoli **sensoriali** passano per un modello di "
        "trasduzione (curva di Hill, filtro, adattamento).",
        "Una corrente di −20 pA iperpolarizza sempre; GtACR1 (E_rev −65 mV) invece iperpolarizza un neurone "
        "depolarizzato ma lascia quasi invariato uno già vicino a −65 mV, pur riducendone la risposta agli input.",
    ),
    "manipulations": (
        "Gli strumenti usati in laboratorio sulla mosca, applicati ai neuroni scelti come con un driver GAL4. "
        "**Genetiche**: tolgono un neurone in tre modi diversi (ablazione, TNT, Kir2.1). **Termogenetiche**: "
        "agiscono solo quando la temperatura sale. **Farmacologiche**: agiscono su tutto il circuito.",
        "Per capire se un neurone è *necessario* per una risposta: salva il controllo (📌), silenzialo con Kir2.1 "
        "e confronta le frequenze. Se la risposta a valle scompare anche con TNT, serve la sua trasmissione "
        "chimica; se scompare solo con Kir2.1, conta anche il suo accoppiamento elettrico.",
    ),
    "weights": (
        "Editor della matrice delle sinapsi chimiche. Ogni cella è la conduttanza di picco evocata da un singolo "
        "spike del neurone di riga nel neurone di colonna: positiva = eccitatoria (ACh), negativa = inibitoria "
        "(GABA, glutammato via GluCl). Le modifiche fatte a mano qui si perdono passando alla pagina 📁 Dati: "
        "prima esporta l'esperimento (💾).",
        "Scrivi −5 nella cella N3 → N1 della cascata: N3 inibisce N1 e crei un anello di feedback negativo; "
        "la scarica di N1 diventa più rada.",
    ),
    "gap": (
        "Le gap junction (innexine, es. ShakB) collegano direttamente il citoplasma di due neuroni: la corrente "
        "fluisce in entrambe le direzioni ed è proporzionale alla differenza di potenziale. Sono veloci (nessun "
        "ritardo sinaptico) e sincronizzano.",
        "Nella via di fuga della Giant Fiber il motoneurone TTMn segue la GF in ~0,1 ms grazie alla gap junction: "
        "bloccala (Manipolazioni → gap junction 100%) e la risposta rapida scompare.",
    ),
    "metrics": (
        "Riepilogo della simulazione: neuroni, sinapsi eccitatorie / inibitorie / elettriche, spike totali, "
        "frequenza media della rete e tempo di calcolo.",
        "",
    ),
    # ------------------------------------------------------------------ analisi
    "an_conductance": (
        "Le conduttanze sinaptiche ricevute dal neurone (nS, sopra) e le correnti che ne risultano (pA, sotto). "
        "Eccitazione in rosso, inibizione in blu, stimoli a conduttanza in acqua. La corrente dipende sia da g "
        "sia dalla distanza di V dal potenziale di inversione.",
        "Se g inibitoria (blu) cresce proprio quando il neurone smette di scaricare, la pausa è dovuta "
        "all'inibizione e non all'adattamento della soglia.",
    ),
    "an_rate": (
        "Frequenza istantanea stimata convolvendo gli spike con un kernel gaussiano di larghezza σ: trasforma "
        "una sequenza di spike in una curva continua in Hz.",
        "Un picco iniziale seguito da un plateau più basso indica adattamento (o depressione sinaptica); con σ "
        "grande il picco si appiattisce.",
    ),
    "an_isi": (
        "Istogramma degli intervalli fra spike consecutivi (ISI). Il coefficiente di variazione CV = deviazione "
        "standard / media misura la regolarità: ~0 = metronomo, ~1 = casuale (Poisson), > 1 = a burst.",
        "N1 della cascata di default ha CV ≈ 0,09: scarica regolare, come atteso per un LIF con input costante. "
        "Aggiungi rumore di fondo e il CV sale.",
    ),
    "an_ccg": (
        "Cross-correlogramma: per ogni spike del riferimento, quando scarica il bersaglio (ritardo positivo = "
        "dopo). È il modo classico di inferire connessioni dalle registrazioni di più neuroni.",
        "Riferimento N1 e bersaglio N2 nella cascata: picco a +4–5 ms, la latenza sinaptica. Poiché la scarica è "
        "molto regolare (ISI ≈ 27 ms) compaiono anche picchi a −23 e +33 ms, dovuti agli spike dei cicli vicini: "
        "aggiungi rumore di fondo per isolare il picco sinaptico. Un avvallamento indica inibizione; un picco "
        "centrato a 0 indica input comune o gap junction.",
    ),
    "an_sta": (
        "Spike-triggered average: la media del segnale scelto nei ms che precedono ogni spike. Mostra quale "
        "andamento dell'input tende a far scaricare il neurone.",
        "Con uno stimolo di rumore gaussiano la STA della corrente iniettata sale negli ultimi ms prima dello "
        "spike: è una stima del filtro lineare del neurone (la sua \"finestra di integrazione\", ~τ_m).",
    ),
    "an_psd": (
        "Spettro di potenza: quanta varianza del segnale cade a ciascuna frequenza. Picchi = oscillazioni "
        "periodiche; uno spettro che scende dolcemente = attività irregolare.",
        "Nell'oscillatore half-center compare un picco alla frequenza del ritmo di alternanza (l'inverso del "
        "periodo fra due burst dello stesso lato).",
    ),
    "an_control": (
        "Confronto fra il risultato salvato come controllo (📌, grigio) e la condizione attuale: tracciato V(t) "
        "sovrapposto e tabella delle frequenze con la differenza.",
        "È il disegno sperimentale di base: una sola variabile cambiata rispetto al controllo, tutto il resto "
        "identico (stesso seme, stessi stimoli).",
    ),
    # ------------------------------------------------------------------ protocolli
    "pr_fi": (
        "Curva f-I: frequenza di scarica in funzione della corrente iniettata in un neurone isolato. La "
        "**reobase** è la corrente minima che fa scaricare; la **pendenza** è il guadagno (Hz/pA).",
        "Con i default LIF la reobase teorica è (V_th − V_rest) / R_m = (−45 + 60) / 1 = 15 pA. Aumentando Δθ la "
        "curva si abbassa e si appiattisce: l'adattamento riduce il guadagno.",
    ),
    "pr_sweep": (
        "Ripete l'esperimento della scheda Simulazione variando uno o due parametri e misura la risposta di ogni "
        "neurone: è lo strumento per trovare soglie e regimi.",
        "Varia «W: N1 → N2» nella cascata: sotto una certa conduttanza N2 non segue più N1 e la sua frequenza "
        "crolla. Quella è la soglia di trasmissione della sinapsi.",
    ),
    "pr_trials": (
        "Lo stesso esperimento con semi casuali diversi. Raster dei trial e PSTH mostrano la variabilità; il "
        "**fattore di Fano** (varianza/media dei conteggi) vale 1 per un processo di Poisson; l'**affidabilità** "
        "misura quanto gli spike cadono agli stessi istanti.",
        "Con rumore di fondo e stimolo costante gli spike si sparpagliano (affidabilità bassa); con \"frozen noise\" "
        "gli spike si allineano fra i trial: il neurone è preciso quando l'input è ricco di fluttuazioni.",
    ),
    "pr_pp": (
        "Due spike presinaptici a intervalli crescenti: il rapporto fra la seconda e la prima risposta (**PPR**) "
        "misura la plasticità a breve termine. PPR < 1 = depressione, > 1 = facilitazione. A destra un treno "
        "regolare di spike.",
        "Con U alto (es. 0,5) e τ_rec di centinaia di ms la seconda risposta è più piccola e recupera lentamente: "
        "come la sinapsi ORN → PN (Kazama & Wilson 2008).",
    ),
    "pr_zap": (
        "Una corrente chirp sottosoglia misura l'**impedenza** |Z(f)| della membrana: quanto V oscilla per pA di "
        "input a ogni frequenza. Un massimo a f > 0 indica **risonanza**.",
        "Il LIF è un filtro passa-basso: |Z| è massima a bassa frequenza e dimezza la potenza a f = 1/(2π τ_m) ≈ "
        "8 Hz con τ_m = 20 ms. Con Hodgkin-Huxley (default) compare un picco di risonanza a ~60 Hz (Q ≈ 2), dovuto "
        "alle cinetiche dei canali: attivazione del K⁺ e inattivazione del Na⁺.",
    ),
    "pr_vclamp": (
        "L'esperimento di Hodgkin e Huxley: il potenziale viene imposto a gradini e si misurano le correnti "
        "ioniche. A sinistra le correnti nel tempo, a destra le curve corrente-potenziale (I-V) di picco.",
        "I_Na è transitoria e verso l'interno (negativa) e si inverte vicino a E_Na = +50 mV; I_K è ritardata, "
        "sostenuta e cresce con la depolarizzazione.",
    ),
    # ------------------------------------------------------------------ pagina Dati
    "data_folder": (
        "I dati stanno in una cartella separata dal codice, così aggiornare o reinstallare l'app non li tocca. "
        "**connectome/** contiene le tabelle del connettoma convertite in Parquet (colonnare e compresso: si "
        "leggono solo le colonne e le righe che servono); **stimuli/** le forme d'onda della libreria.",
        "Un connections.csv.gz di FlyWire da ~100 MB diventa un Parquet di dimensioni simili, ma filtrare 150 "
        "neuroni richiede circa un secondo invece di rileggere tutto il CSV.",
    ),
    "codex_files": (
        "Da **FlyWire Codex** (codex.flywire.ai → Download, dataset FAFB o Male CNS) servono almeno la tabella "
        "degli **archi** (es. `connections`: pre, post, neuropilo, sinapsi, trasmettitore) e una tabella dei "
        "**neuroni** (es. `classification`: superclasse, classe, tipo). Tabelle in più (nomi, trasmettitori, tipi "
        "consolidati) vengono unite per ID. Il tipo di tabella si riconosce dalle colonne, non dal nome del file; "
        "sono accettati anche i Feather del Male CNS di Janelia (`body_pre`, `body_post`, `weight`, `bodyId`, "
        "`type`, `superclass`…).",
        "FAFB v783 da Codex: bastano `connections_princeton` (archi, soglia ≥ 5 sinapsi), `classification` "
        "(superclasse, classe), `consolidated_cell_types` (tipo cellulare), `neurons` (trasmettitore) e `names`; "
        "`visual_neuron_types` aggiunge i tipi del lobo ottico. Non servono le tabelle di singole sinapsi "
        "(`synapse_coordinates`, `…_synapse_table`, anche da molti GB), né `coordinates`, `labels`, `cell_stats`, "
        "`neuropil_synapse_table`: i neuropili si ricavano già dagli archi.",
    ),
    "explorer": (
        "Filtra i neuroni del connettoma per annotazioni (superclasse, classe, tipo, trasmettitore, lato) e per "
        "neuropilo. I grafici riassumono chi hai selezionato; le sinapsi in ingresso e in uscita di ogni neurone "
        "sono calcolate dalla tabella degli archi.",
        "Classe = ALPN e neuropilo = AL_L selezionano i neuroni di proiezione del lobo antennale sinistro; "
        "aggiungi la classe Kenyon_Cell e il neuropilo MB_CA_L per seguire l'informazione olfattiva fino al "
        "calice del corpo fungo.",
    ),
    "subcircuit": (
        "Porta nella Sandbox i neuroni scelti (max 150): le sinapsi fra loro diventano la matrice W, con "
        "|W| = numero di sinapsi × nS per sinapsi e segno dal trasmettitore del neurone presinaptico (ACh "
        "eccitatorio; GABA, glutammato e istamina inibitori; i neuromodulatori sono trattati come eccitatori "
        "deboli). Le connessioni con i neuroni esclusi vengono perse: il sottocircuito è un modello *ridotto*.",
        "Con gli ORN, i PN e alcuni LN del glomerulo DA1 ottieni un modello del lobo antennale basato sul connettoma: "
        "stimola gli ORN (sono le sorgenti di default) e confronta la risposta dei PN prima e dopo "
        "la picrotossina.",
    ),
    "stimulus_library": (
        "Forme d'onda salvate (tempo in ms, valore in pA) riutilizzabili in qualunque canale di stimolo con tipo "
        "*Da file CSV*. Utile per riprodurre correnti registrate o protocolli generati altrove.",
        "Registra una corrente sinaptica in un esperimento (⬇️ V(t) o correnti in CSV), salvala qui e iniettala "
        "in un altro neurone per vedere come la filtrano parametri di membrana diversi.",
    ),
}

# ============================================================================ schermata iniziale

WELCOME = {
    "title": "Benvenuto nella Drosophila Neural Sandbox",
    "lead": "Un laboratorio virtuale di elettrofisiologia sui circuiti del moscerino della frutta: costruisci un "
            "circuito, stimolalo come in un esperimento vero, manipola i neuroni e misura cosa cambia.",
    "cards": [
        ("🧬", "Scegli un circuito", "Sette circuiti d'esempio (fuga della Giant Fiber, lobo antennale, corpo "
                                    "fungo, T4…) oppure neuroni reali dal connettoma FlyWire."),
        ("⚡", "Stimola", "Correnti, luce, odori, oggetti in avvicinamento, optogenetica: scheda **Stimoli**."),
        ("🧪", "Manipola", "Ablazione, TNT, Kir2.1, dTrpA1, shibire, farmaci — confronta sempre con il "
                          "controllo 📌."),
        ("📈", "Misura", "Grafici ①–④, schede **Analisi** e **Protocolli** (curva f-I, paired-pulse, "
                        "voltage clamp…)."),
    ],
    "steps": "**In 30 secondi:** scegli una *Topologia* nella barra laterale → i grafici a destra si aggiornano "
             "da soli → premi **📌 Usa come controllo** → cambia uno stimolo o una manipolazione e osserva la "
             "differenza. Ogni controllo ha un ⓘ con unità e valori tipici; sotto i grafici trovi "
             "**:material/help: Cosa sto guardando?**. I tuoi dati (connettoma, stimoli) si gestiscono nella pagina **📁 Dati**.",
}

# ============================================================================ didascalie

KIND_HELP = {
    "Onda quadra": "Gradino di corrente o treno di impulsi a frequenza fissa.",
    "Rampa": "Corrente che cresce linearmente: misura la soglia dinamica e l'accomodazione.",
    "Scala di gradini": "Gradini crescenti in un'unica registrazione: la curva f-I \"a occhio\".",
    "Onda periodica": "Sinusoide, triangolare, dente di sega o quadra.",
    "Chirp (ZAP)": "Frequenza crescente nel tempo: rivela filtri e risonanze della membrana.",
    "Rumore gaussiano": "Bianco o correlato (Ornstein-Uhlenbeck); utile per la spike-triggered average.",
    "Barra in movimento": "Impulsi in sequenza sui bersagli nell'ordine scelto: simula un oggetto che attraversa "
                          "colonne visive adiacenti.",
    "Equazione custom": "Qualsiasi funzione di t, valutata in modo sicuro.",
    "Da file CSV": "Riproduce una forma d'onda registrata (colonne: tempo in ms, valore in pA), caricata qui o "
                   "presa dalla libreria della pagina 📁 Dati.",
    "Sensoriale": "Stimolo fisico (luce, odore, looming) trasdotto in corrente dal recettore.",
    "Input sinaptico di Poisson": "Bombardamento sinaptico di fondo come conduttanza: cambia anche la resistenza "
                                  "di membrana (shunting).",
    "Optogenetica": "Luce su un'opsina espressa nei bersagli: CsChrimson attiva, GtACR1 inibisce.",
}

PROTOCOLS = {
    "Curva f-I": "Gradini di corrente di ampiezza crescente su un neurone isolato con i parametri del modello "
                 "scelto: reobase, guadagno, adattamento.",
    "Esplorazione parametrica": "Ripete l'esperimento corrente variando uno o due parametri e misura la risposta "
                                "di ogni neurone.",
    "Trial ripetuti": "Stesso esperimento con semi casuali diversi: variabilità fra trial, fattore di Fano e "
                      "affidabilità temporale degli spike.",
    "Paired-pulse (STP)": "Due spike presinaptici a intervalli crescenti e un treno regolare: depressione e "
                          "facilitazione a breve termine.",
    "Impedenza (ZAP)": "Corrente chirp sottosoglia: |Z(f)| e risonanza della membrana.",
    "Voltage clamp (HH)": "L'esperimento di Hodgkin e Huxley: gradini di potenziale imposto e correnti di Na⁺ e K⁺.",
}
PROTOCOL_EXPLAIN = {"Curva f-I": "pr_fi", "Esplorazione parametrica": "pr_sweep", "Trial ripetuti": "pr_trials",
                    "Paired-pulse (STP)": "pr_pp", "Impedenza (ZAP)": "pr_zap", "Voltage clamp (HH)": "pr_vclamp"}

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
    ("Connettoma reale", "Nella pagina 📁 Dati carica le tabelle di FlyWire Codex, filtra i neuroni per classe, tipo "
     "e neuropilo e porta fino a 150 neuroni nella Sandbox con «Crea sottocircuito»."),
]
