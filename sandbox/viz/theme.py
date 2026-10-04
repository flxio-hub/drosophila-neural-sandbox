"""Palette e inchiostri dei grafici, in versione chiara e scura.

La palette categoriale (8 slot, ordine fisso) è validata per separazione in
caso di daltonismo; il colore segue l'entità (neurone / canale di stimolo), mai
il rango. Oltre 8 neuroni visibili non si generano nuovi colori: il tracciato
V(t) limita la selezione a 8 e il raster passa all'inchiostro neutro.

Ruoli semantici (coerenti con la heatmap divergente di W):
    eccitatorio = rosso · inibitorio = blu · stimolo esterno = acqua (sempre con etichetta diretta)
    controllo = grigio neutro, per confrontarsi con la condizione attuale senza competere con essa
Le rampe ordinali (5 gradini) e sequenziali sono validate separatamente per i due temi.
"""

from dataclasses import dataclass

# "Source Sans" è il carattere dell'interfaccia di Streamlit: i grafici usano lo stesso, se disponibile
FONT_FAMILY = '"Source Sans", "Source Sans Pro", system-ui, -apple-system, "Segoe UI", sans-serif'
MAX_SERIES = 8


@dataclass(frozen=True)
class Theme:
    mode: str
    series: tuple[str, ...]
    ink: str          # testo primario
    ink_secondary: str
    muted: str        # etichette degli assi
    grid: str         # griglia hairline
    axis: str         # linea di base / assi
    surface: str      # sfondo tooltip
    diverging: tuple  # colorscale inibitorio (blu) ↔ 0 (grigio) ↔ eccitatorio (rosso)
    sequential: tuple  # colorscale di magnitudine (un solo tono, dal "vicino a zero" al massimo)
    ordinal: tuple    # 5 gradini ordinati per serie con un ordine naturale (es. potenziali di comando)
    exc: str
    inh: str
    stim: str
    control: str


LIGHT = Theme(
    mode="light",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb",
    diverging=((0.0, "#184f95"), (0.25, "#6da7ec"), (0.5, "#f0efec"), (0.75, "#ee8f8e"), (1.0, "#b3302f")),
    sequential=((0.0, "#cde2fb"), (0.5, "#3987e5"), (1.0, "#0d366b")),
    ordinal=("#86b6ef", "#5598e7", "#256abf", "#184f95", "#0d366b"),
    exc="#e34948", inh="#2a78d6", stim="#1baf7a", control="#a3a29c",
)

DARK = Theme(
    mode="dark",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ink="#ffffff", ink_secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835", surface="#1a1a19",
    diverging=((0.0, "#9ec5f4"), (0.25, "#2a78d6"), (0.5, "#383835"), (0.75, "#c43e3e"), (1.0, "#f2a2a1")),
    sequential=((0.0, "#104281"), (0.5, "#3987e5"), (1.0, "#cde2fb")),
    ordinal=("#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
    exc="#e66767", inh="#3987e5", stim="#199e70", control="#6b6a65",
)


def neuron_colors(neurons: list[str], visible: list[str], theme: Theme) -> dict[str, str]:
    """Colore stabile per neurone.

    Con ≤ 8 neuroni il colore dipende dalla posizione nel circuito (non cambia
    filtrando); con reti più grandi dipende dall'ordine della selezione (max 8).
    """
    if len(neurons) <= MAX_SERIES:
        return {name: theme.series[i] for i, name in enumerate(neurons)}
    return {name: theme.series[i] for i, name in enumerate(visible[:MAX_SERIES])}
