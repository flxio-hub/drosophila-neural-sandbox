"""Tutti gli input esterni della rete in un unico contenitore.

* ``i_ext``        corrente iniettata (pA)
* ``g_ext``        conduttanza esterna totale (nS)          ┐ stimoli a conduttanza,
* ``ge_ext``       Σ g·E_rev delle conduttanze esterne (pA) ┘ Kir2.1, dTrpA1…
* ``release_gain`` moltiplicatore del rilascio presinaptico (TNT, shibire^ts)

Gli array opzionali vengono allocati solo quando servono.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NetworkDrive:
    i_ext: np.ndarray
    g_ext: np.ndarray | None = None
    ge_ext: np.ndarray | None = None
    release_gain: np.ndarray | None = None

    @classmethod
    def zeros(cls, steps: int, n: int) -> "NetworkDrive":
        return cls(i_ext=np.zeros((steps, n)))

    @property
    def shape(self) -> tuple[int, int]:
        return self.i_ext.shape

    def add_current(self, column: int, values) -> None:
        self.i_ext[:, column] += values

    def add_conductance(self, column: int, g, e_rev: float) -> None:
        if self.g_ext is None:
            self.g_ext, self.ge_ext = np.zeros(self.shape), np.zeros(self.shape)
        self.g_ext[:, column] += g
        self.ge_ext[:, column] += np.asarray(g) * e_rev

    def scale_release(self, column: int, gain) -> None:
        if self.release_gain is None:
            self.release_gain = np.ones(self.shape)
        self.release_gain[:, column] *= gain

    def silence(self, column: int) -> None:
        """Rimuove ogni input esterno dal neurone (usato per l'ablazione)."""
        self.i_ext[:, column] = 0.0
        if self.g_ext is not None:
            self.g_ext[:, column] = 0.0
            self.ge_ext[:, column] = 0.0

    def simulate_kwargs(self) -> dict:
        return {"g_ext": self.g_ext, "ge_ext": self.ge_ext, "release_gain": self.release_gain}
