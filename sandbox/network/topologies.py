"""Circuiti preimpostati ispirati a motivi reali del sistema nervoso di Drosophila.

Ogni preset restituisce un :class:`Circuit` con la matrice chimica ``W[pre, post]``
e quella elettrica ``gap`` (simmetrica) in nS, più uno o più stimoli suggeriti
(letti dall'interfaccia come valori di default).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Circuit:
    name: str
    neurons: list[str]
    weights: np.ndarray                                   # W[pre, post] (nS), sinapsi chimiche
    description: str = ""
    default_stimuli: list[dict] = field(default_factory=list)
    gap: np.ndarray | None = None                         # gap junction (nS), simmetrica

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        n = len(self.neurons)
        if self.weights.shape != (n, n):
            raise ValueError(f"W deve essere {n}×{n}, ricevuto {self.weights.shape}")
        self.gap = np.zeros((n, n)) if self.gap is None else np.asarray(self.gap, dtype=float)
        if self.gap.shape != (n, n):
            raise ValueError(f"La matrice gap deve essere {n}×{n}")

    @property
    def n(self) -> int:
        return len(self.neurons)


def _matrix(neurons: list[str], edges: list[tuple[str, str, float]]) -> np.ndarray:
    index = {name: i for i, name in enumerate(neurons)}
    weights = np.zeros((len(neurons), len(neurons)))
    for pre, post, w in edges:
        weights[index[pre], index[post]] = w
    return weights


def _gap_matrix(neurons: list[str], pairs: list[tuple[str, str, float]]) -> np.ndarray:
    gap = _matrix(neurons, pairs)
    return np.maximum(gap, gap.T)


def feedforward_cascade(n_layers: int = 5, weight: float = 3.0) -> Circuit:
    neurons = [f"N{i + 1}" for i in range(n_layers)]
    edges = [(neurons[i], neurons[i + 1], weight) for i in range(n_layers - 1)]
    return Circuit(
        name="Cascata feed-forward",
        neurons=neurons,
        weights=_matrix(neurons, edges),
        description=(
            "Catena N1 → N2 → … → Nn di sinapsi eccitatorie colinergiche. Osserva la "
            "latenza che si accumula a ogni stadio e come la frequenza si trasforma "
            "(filtraggio passa-basso, soglie di sommazione temporale)."
        ),
        default_stimuli=[{"kind": "Onda quadra", "targets": [neurons[0]],
                          "amplitude": 30.0, "t_on": 50.0, "t_off": 400.0}],
    )


def antennal_lobe_feedback() -> Circuit:
    neurons = ["ORN_A", "ORN_B", "PN_A", "PN_B", "LN"]
    edges = [
        ("ORN_A", "PN_A", 2.0), ("ORN_B", "PN_B", 2.0),     # glomeruli: ORN → PN (ACh)
        ("ORN_A", "LN", 0.6), ("ORN_B", "LN", 0.6),         # input diretto agli interneuroni locali
        ("PN_A", "LN", 0.8), ("PN_B", "LN", 0.8),
        ("LN", "PN_A", -2.5), ("LN", "PN_B", -2.5),         # LN GABAergico: inibizione a feedback
        ("LN", "ORN_A", -0.6), ("LN", "ORN_B", -0.6),       # inibizione presinaptica (approssimata)
    ]
    return Circuit(
        name="Feedback inibitorio (lobo antennale)",
        neurons=neurons,
        weights=_matrix(neurons, edges),
        description=(
            "Due glomeruli (ORN → PN) accoppiati da un interneurone locale GABAergico (LN) "
            "che riceve da entrambi e li inibisce: normalizzazione di guadagno e "
            "inibizione laterale. Stimola ORN_A e aumenta l'input a ORN_B per vedere la "
            "competizione tra i canali."
        ),
        default_stimuli=[
            {"kind": "Sensoriale", "targets": ["ORN_A"], "modality": "Neurone olfattivo (ORN)",
             "profile": "Impulso", "amplitude": 0.6, "t_on": 100.0, "t_off": 400.0},
            {"kind": "Sensoriale", "targets": ["ORN_B"], "modality": "Neurone olfattivo (ORN)",
             "profile": "Impulso", "amplitude": 0.15, "t_on": 100.0, "t_off": 400.0},
        ],
    )


def giant_fiber_escape(w_ttmn: float = 1.0, w_psi: float = 1.0, gap_ttmn: float = 3.0,
                       gap_psi: float = 3.0) -> Circuit:
    neurons = ["LC4", "LPLC2", "GF", "PSI", "TTMn", "DLMn"]
    edges = [
        ("LC4", "GF", 1.2), ("LPLC2", "GF", 1.2),     # neuroni visivi di looming → Giant Fiber
        ("GF", "TTMn", w_ttmn), ("GF", "PSI", w_psi),  # componente chimica (colinergica) delle sinapsi miste
        ("PSI", "DLMn", 3.0),                         # via del volo: interneurone PSI → muscoli DLM
    ]
    # componente elettrica: gap junction formate da ShakB (innexina) tra GF, TTMn e PSI
    gap = _gap_matrix(neurons, [("GF", "TTMn", gap_ttmn), ("GF", "PSI", gap_psi)])
    return Circuit(
        name="Sensorio-motorio: fuga della Giant Fiber",
        neurons=neurons,
        weights=_matrix(neurons, edges),
        gap=gap,
        description=(
            "Circuito di fuga: i neuroni visivi LC4 e LPLC2 rilevano un oggetto in "
            "avvicinamento (looming) e convergono sulla Giant Fiber, che attiva il "
            "motoneurone del salto (TTMn) e, tramite PSI, i muscoli del volo (DLMn). "
            "GF → TTMn e GF → PSI sono sinapsi miste, elettriche (ShakB) e chimiche: "
            "prova a bloccare le gap junction nella scheda Manipolazioni (come nei mutanti shakB²)."
        ),
        default_stimuli=[{"kind": "Sensoriale", "targets": ["LC4", "LPLC2"],
                          "modality": "Rilevatore di looming (LC4 / LPLC2)", "profile": "Looming",
                          "t_collision": 350.0, "l_over_v": 40.0}],
    )


def half_center_cpg() -> Circuit:
    neurons = ["IN_sx", "IN_dx", "MN_sx", "MN_dx"]
    edges = [
        ("IN_sx", "IN_dx", -4.0), ("IN_dx", "IN_sx", -4.0),  # inibizione reciproca
        ("IN_sx", "MN_sx", 2.0), ("IN_dx", "MN_dx", 2.0),
    ]
    return Circuit(
        name="Oscillatore half-center (CPG locomotorio)",
        neurons=neurons,
        weights=_matrix(neurons, edges),
        description=(
            "Due interneuroni che si inibiscono a vicenda ricevono lo stesso drive tonico: "
            "l'adattamento di soglia (Δθ, τ_θ) fa sì che il neurone attivo si 'stanchi' e "
            "l'altro prenda il sopravvento, generando bursts alternati come nei CPG della "
            "locomozione larvale. Prova a variare Δθ e τ_θ."
        ),
        default_stimuli=[{"kind": "Rumore gaussiano", "targets": ["IN_sx", "IN_dx"],
                          "mean": 30.0, "sigma": 8.0, "tau_corr": 5.0}],
    )


def random_ei_network(n: int = 20, p_connect: float = 0.2, frac_exc: float = 0.8,
                      w_exc: float = 1.0, w_inh: float = 3.0, seed: int = 0) -> Circuit:
    """Rete ricorrente sparsa che rispetta la legge di Dale (segno fissato dal neurone presinaptico)."""
    rng = np.random.default_rng(seed)
    n_exc = int(round(frac_exc * n))
    neurons = [f"E{i + 1}" for i in range(n_exc)] + [f"I{i + 1}" for i in range(n - n_exc)]
    mask = rng.random((n, n)) < p_connect
    np.fill_diagonal(mask, False)
    sign = np.where(np.arange(n) < n_exc, w_exc, -w_inh)[:, None]
    weights = mask * sign * rng.uniform(0.5, 1.5, (n, n))
    return Circuit(
        name="Rete ricorrente E/I casuale",
        neurons=neurons,
        weights=weights,
        description=(
            "Rete sparsa di neuroni eccitatori (E, colinergici) e inibitori (I, GABAergici) "
            "che rispetta la legge di Dale. Utile per esplorare bilanciamento E/I, "
            "attività persistente e sincronizzazione."
        ),
        default_stimuli=[{"kind": "Rumore gaussiano", "targets": neurons[: max(1, n_exc // 2)],
                          "mean": 18.0, "sigma": 10.0, "tau_corr": 5.0}],
    )


def mushroom_body(n_pn: int = 12, n_kc: int = 30, claws: int = 5, w_pn_kc: float = 0.5,
                  w_kc_apl: float = 1.0, w_apl_kc: float = 3.0, w_kc_mbon: float = 1.0, odor_size: int = 4,
                  odor_overlap: int = 2, seed: int = 1) -> Circuit:
    """Calice del corpo fungo: ogni cellula di Kenyon (KC) campiona pochi PN a caso ("claw") e l'APL
    GABAergico, eccitato da tutte le KC, le inibisce tutte (Lin et al. 2014)."""
    rng = np.random.default_rng(seed)
    pns = [f"PN{i + 1}" for i in range(n_pn)]
    kcs = [f"KC{i + 1}" for i in range(n_kc)]
    neurons = pns + kcs + ["APL", "MBON"]
    weights = np.zeros((len(neurons), len(neurons)))
    for k in range(n_kc):
        inputs = rng.choice(n_pn, size=min(claws, n_pn), replace=False)
        weights[inputs, n_pn + k] = w_pn_kc * rng.uniform(0.7, 1.3, inputs.size)
    apl, mbon = len(neurons) - 2, len(neurons) - 1
    weights[n_pn:n_pn + n_kc, apl] = w_kc_apl
    weights[apl, n_pn:n_pn + n_kc] = -w_apl_kc
    weights[n_pn:n_pn + n_kc, mbon] = w_kc_mbon
    odor_a = pns[:odor_size]
    odor_b = pns[odor_size - odor_overlap: 2 * odor_size - odor_overlap]  # odori simili: PN in parte condivisi
    return Circuit(
        name="Corpo fungo: codifica sparsa",
        neurons=neurons,
        weights=weights,
        description=(
            "Due odori attivano gruppi sovrapposti di PN; ogni cellula di Kenyon somma ~5 PN a caso "
            "e scarica solo se molti sono attivi insieme, mentre l'APL inibisce tutte le KC in proporzione "
            "alla loro attività. Risultato: pochi KC per odore e insiemi diversi per odori simili "
            "(separazione dei pattern). Prova ad ablare l'APL o ad applicare picrotossina."
        ),
        default_stimuli=[
            {"kind": "Onda quadra", "targets": odor_a, "amplitude": 45.0, "t_on": 50.0, "t_off": 220.0},
            {"kind": "Onda quadra", "targets": odor_b, "amplitude": 45.0, "t_on": 280.0, "t_off": 450.0},
        ],
    )


def t4_direction_selectivity(w_exc: float = 3.0, w_inh: float = 8.0, w_in: float = 4.0) -> Circuit:
    """Selettività di direzione di T4 con soppressione della direzione nulla (Barlow-Levick).

    Tre colonne visive adiacenti (1, 2, 3). T4_dx riceve eccitazione dalla colonna centrale e
    inibizione ritardata (tramite un interneurone GABAergico) dalla colonna 3; T4_sx è speculare.
    Un movimento 3 → 2 porta l'inibizione in anticipo e sopprime T4_dx.
    """
    neurons = ["Mi1_1", "Mi1_2", "Mi1_3", "CT1_1", "CT1_3", "T4_dx", "T4_sx"]
    edges = [
        ("Mi1_2", "T4_dx", w_exc), ("Mi1_2", "T4_sx", w_exc),   # eccitazione dalla colonna centrale
        ("Mi1_3", "CT1_3", w_in), ("CT1_3", "T4_dx", -w_inh),   # inibizione dal lato della direzione nulla
        ("Mi1_1", "CT1_1", w_in), ("CT1_1", "T4_sx", -w_inh),
    ]
    columns = ["Mi1_1", "Mi1_2", "Mi1_3"]
    return Circuit(
        name="Selettività di direzione (T4, Barlow-Levick)",
        neurons=neurons,
        weights=_matrix(neurons, edges),
        description=(
            "Una barra luminosa attraversa tre colonne adiacenti. T4_dx preferisce il movimento verso "
            "destra (1 → 3), T4_sx verso sinistra: nella direzione nulla l'inibizione della colonna "
            "che viene raggiunta per prima arriva insieme all'eccitazione e la annulla. Varia l'intervallo "
            "fra le colonne (velocità) o il segno delle sinapsi CT1 → T4 per vedere la selettività sparire."
        ),
        default_stimuli=[
            {"kind": "Barra in movimento", "targets": columns, "direction": "→", "t_on": 50.0,
             "amplitude": 100.0, "pulse_ms": 20.0, "interval_ms": 15.0},
            {"kind": "Barra in movimento", "targets": columns, "direction": "←", "t_on": 300.0,
             "amplitude": 100.0, "pulse_ms": 20.0, "interval_ms": 15.0},
        ],
    )


PRESETS = {
    "Cascata feed-forward": feedforward_cascade,
    "Feedback inibitorio (lobo antennale)": antennal_lobe_feedback,
    "Sensorio-motorio: fuga della Giant Fiber": giant_fiber_escape,
    "Oscillatore half-center (CPG)": half_center_cpg,
    "Rete ricorrente E/I casuale": random_ei_network,
    "Corpo fungo: codifica sparsa": mushroom_body,
    "Selettività di direzione (T4)": t4_direction_selectivity,
}
