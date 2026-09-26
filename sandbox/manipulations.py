"""Manipolazioni sperimentali: gli strumenti genetici e farmacologici tipici della ricerca su Drosophila.

Genetiche (espresse con GAL4/UAS nei neuroni scelti):
* Kir2.1      canale K⁺ rettificante: conduttanza costante verso E_K → neurone silenziato
* TNT         tossina tetanica (catena leggera): blocca il rilascio sinaptico, sempre
* shibire^ts  dinamina termosensibile: blocca il rilascio sopra ~29 °C
* dTrpA1      canale cationico attivato dal caldo (sopra ~26–27 °C): attiva il neurone
* Ablazione   il neurone viene rimosso dal circuito (nessun input, nessuna sinapsi)

Farmacologiche (applicate al bagno, su tutto il circuito):
* Picrotossina  antagonista GABA_A/GluCl → riduce le sinapsi inibitorie
* Mecamilamina  antagonista nAChR → riduce le sinapsi eccitatorie colinergiche
* Bloccante delle gap junction (in vivo: mutanti shakB)

Nota sui tempi: in vivo shibire^ts e dTrpA1 agiscono in secondi o minuti; qui la
loro attivazione segue istantaneamente il protocollo di temperatura, così che
l'effetto si possa osservare in una simulazione di pochi centinaia di ms.
"""

import numpy as np

from .simulation.drive import NetworkDrive

TRPA1_E_REV = 0.0  # canale cationico non selettivo (mV)

DEFAULT_MANIPULATIONS = {
    "ablated": [], "tnt": [],
    "kir": [], "kir_g": 3.0, "e_k": -80.0,
    "trpa1": [], "trpa1_gmax": 3.0, "trpa1_t_half": 27.0,
    "shibire": [], "shibire_t_half": 29.5,
    "temp_baseline": 22.0, "temp_target": 32.0, "temp_on": 100.0, "temp_off": 400.0, "temp_ramp": 30.0,
    "ptx": 0.0, "mec": 0.0, "gap_block": 0.0,
}


def with_defaults(spec: dict | None) -> dict:
    return {**DEFAULT_MANIPULATIONS, **(spec or {})}


def uses_temperature(spec: dict) -> bool:
    return bool(spec.get("trpa1") or spec.get("shibire"))


def temperature_trace(t: np.ndarray, spec: dict) -> np.ndarray:
    """Protocollo di temperatura: da baseline a target con una rampa, poi ritorno (°C)."""
    s = with_defaults(spec)
    ramp = max(s["temp_ramp"], 1e-9)
    rise = np.clip((t - s["temp_on"]) / ramp, 0.0, 1.0)
    fall = np.clip((t - s["temp_off"]) / ramp, 0.0, 1.0)
    return s["temp_baseline"] + (s["temp_target"] - s["temp_baseline"]) * np.clip(rise - fall, 0.0, 1.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def trpa1_activation(temperature, t_half: float, slope: float = 1.0):
    """Frazione di canali dTrpA1 aperti in funzione della temperatura."""
    return _sigmoid((temperature - t_half) / slope)


def shibire_block(temperature, t_half: float, slope: float = 0.5):
    """Frazione di rilascio bloccata da shibire^ts (0 a temperatura permissiva, 1 a restrittiva)."""
    return _sigmoid((temperature - t_half) / slope)


def apply_manipulations(spec: dict | None, neurons: list[str], weights: np.ndarray, gap: np.ndarray,
                        drive: NetworkDrive, t: np.ndarray):
    """Applica le manipolazioni. Restituisce (W, gap, temperatura o None, note descrittive).

    ``drive`` viene modificato sul posto; W e gap sono copie.
    """
    s = with_defaults(spec)
    index = {name: i for i, name in enumerate(neurons)}
    weights = np.array(weights, dtype=float)
    gap = np.array(gap, dtype=float)
    notes = []

    if s["ptx"] > 0:
        weights[weights < 0] *= 1.0 - s["ptx"]
        notes.append(f"Picrotossina: sinapsi inibitorie ridotte del {100 * s['ptx']:.0f}%")
    if s["mec"] > 0:
        weights[weights > 0] *= 1.0 - s["mec"]
        notes.append(f"Mecamilamina: sinapsi eccitatorie ridotte del {100 * s['mec']:.0f}%")
    if s["gap_block"] > 0:
        gap *= 1.0 - s["gap_block"]
        notes.append(f"Gap junction ridotte del {100 * s['gap_block']:.0f}%")

    for name in s["kir"]:
        if name in index:
            drive.add_conductance(index[name], s["kir_g"], s["e_k"])
    if s["kir"]:
        notes.append(f"Kir2.1 in {', '.join(s['kir'])}: +{s['kir_g']:g} nS verso {s['e_k']:g} mV")

    temperature = temperature_trace(t, s) if uses_temperature(s) else None
    if s["trpa1"]:
        g = s["trpa1_gmax"] * trpa1_activation(temperature, s["trpa1_t_half"])
        for name in s["trpa1"]:
            if name in index:
                drive.add_conductance(index[name], g, TRPA1_E_REV)
        notes.append(f"dTrpA1 in {', '.join(s['trpa1'])}: fino a {s['trpa1_gmax']:g} nS sopra ~{s['trpa1_t_half']:g} °C")
    if s["shibire"]:
        gain = 1.0 - shibire_block(temperature, s["shibire_t_half"])
        for name in s["shibire"]:
            if name in index:
                drive.scale_release(index[name], gain)
        notes.append(f"shibire^ts in {', '.join(s['shibire'])}: rilascio bloccato sopra ~{s['shibire_t_half']:g} °C")
    for name in s["tnt"]:
        if name in index:
            drive.scale_release(index[name], 0.0)
    if s["tnt"]:
        notes.append(f"TNT in {', '.join(s['tnt'])}: rilascio sinaptico abolito")

    for name in s["ablated"]:
        if name in index:
            j = index[name]
            weights[j, :] = weights[:, j] = 0.0
            gap[j, :] = gap[:, j] = 0.0
            drive.silence(j)
    if s["ablated"]:
        notes.append(f"Ablati: {', '.join(s['ablated'])}")
    return weights, gap, temperature, notes
