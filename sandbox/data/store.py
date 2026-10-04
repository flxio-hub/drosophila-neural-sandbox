"""Cartella dati esterna al codice: tabelle del connettoma (Parquet) e libreria di stimoli (CSV).

Posizione: variabile d'ambiente ``DROSOPHILA_DATA`` oppure ``~/drosophila_data``::

    connectome/   una tabella Parquet per ogni file caricato (connections, classification, …)
    stimuli/      forme d'onda CSV (tempo in ms, valore in pA)

I file del connettoma possono pesare centinaia di MB: vengono convertiti **una sola volta** in
Parquet leggendoli a blocchi, così la memoria usata non dipende dalla dimensione del file. Le
letture successive (colonne scelte, filtri sugli ID) sono decine di volte più rapide del CSV.
Formati accettati: CSV/TSV (anche compressi .gz, come gli export di FlyWire Codex), Feather/Arrow
(come i file del Male CNS di Janelia) e Parquet.
"""

from __future__ import annotations

import io
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

ENV_VAR = "DROSOPHILA_DATA"
CONNECTOME, STIMULI = "connectome", "stimuli"
TABLE_TYPES = ["csv", "tsv", "gz", "feather", "arrow", "parquet"]  # estensioni per st.file_uploader
_SUFFIXES = (".csv.gz", ".tsv.gz", ".txt.gz", ".csv", ".tsv", ".txt", ".feather", ".arrow", ".ipc", ".parquet", ".gz")
BLOCK_SIZE = 16 << 20  # byte di CSV letti per blocco
ROW_GROUP = 1_000_000  # righe per row group del Parquet


def data_root() -> Path:
    return Path(os.environ.get(ENV_VAR) or Path.home() / "drosophila_data").expanduser()


def folder(kind: str) -> Path:
    path = data_root() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_stem(filename: str) -> str:
    """Nome di file sicuro senza estensione: ``"connections.csv.gz"`` → ``"connections"``."""
    name = Path(str(filename)).name
    for suffix in _SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^\w.\-]+", "_", name).strip("._") or "tabella"


def _inside(kind: str, name: str) -> Path:
    path = (folder(kind) / name).resolve()
    if path.parent != folder(kind).resolve():
        raise ValueError(f"Nome di file non valido: {name}")
    return path


# ============================================================================ conversione in Parquet

def _opener(source, compressed: bool):
    """Funzione che apre ``source`` (percorso o file in memoria) come flusso Arrow, dall'inizio."""
    if isinstance(source, (str, Path)):
        def open_raw():
            return pa.OSFile(str(source), "rb")
    else:
        buffer = pa.py_buffer(source.getbuffer() if hasattr(source, "getbuffer") else source.read())

        def open_raw():
            return pa.BufferReader(buffer)
    if compressed:
        return lambda: pa.CompressedInputStream(open_raw(), "gzip")
    return open_raw


def _sniff_delimiter(open_stream) -> str:
    with open_stream() as stream:
        head = stream.read(65536).decode("utf-8", "replace")
    line = head.splitlines()[0] if head else ""
    return max((",", "\t", ";"), key=line.count)


def _failing_column(exc: Exception, names: list[str]) -> str | None:
    match = re.search(r"column #(\d+)", str(exc))
    if match and int(match.group(1)) < len(names):
        return names[int(match.group(1))]
    return None


def _write_batches(batches, schema: pa.Schema, target: Path, progress=None) -> int:
    """Scrive i blocchi in Parquet raggruppandoli in row group grandi (letture successive più veloci)."""
    rows, pending = 0, []
    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
        for batch in batches:
            pending.append(batch)
            rows += batch.num_rows
            if sum(b.num_rows for b in pending) >= ROW_GROUP:
                writer.write_table(pa.Table.from_batches(pending, schema))
                pending = []
            if progress:
                progress(rows)
        if pending:
            writer.write_table(pa.Table.from_batches(pending, schema))
    return rows


def _csv_to_parquet(open_stream, target: Path, progress=None) -> int:
    """CSV → Parquet a blocchi. Le colonne con tipi incoerenti fra i blocchi diventano testo."""
    parse = pacsv.ParseOptions(delimiter=_sniff_delimiter(open_stream))
    read = pacsv.ReadOptions(block_size=BLOCK_SIZE)
    column_types: dict[str, pa.DataType] = {}
    names: list[str] = []
    for _ in range(64):
        convert = pacsv.ConvertOptions(column_types=column_types, strings_can_be_null=True)
        try:
            with open_stream() as stream:
                reader = pacsv.open_csv(stream, read_options=read, parse_options=parse, convert_options=convert)
                names = reader.schema.names
                null_columns = [f.name for f in reader.schema if pa.types.is_null(f.type)]
                if null_columns:  # colonna vuota nel primo blocco: il tipo va deciso qui, prima di scrivere
                    column_types.update({name: pa.string() for name in null_columns})
                    continue
                return _write_batches(reader, reader.schema, target, progress)
        except pa.ArrowInvalid as exc:
            column = _failing_column(exc, names)
            if column is None or column in column_types:
                raise ValueError(f"CSV non leggibile: {exc}") from exc
            column_types[column] = pa.string()  # es. ID numerici seguiti da testo: si ricomincia come testo
    raise ValueError("CSV non leggibile: troppe colonne con tipi incoerenti.")


def _ipc_to_parquet(source, target: Path, progress=None) -> int:
    handle = pa.memory_map(str(source)) if isinstance(source, (str, Path)) else _opener(source, False)()
    try:
        reader = ipc.open_file(handle)
        batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        schema = reader.schema
    except pa.ArrowInvalid:
        handle.seek(0)
        try:
            reader = ipc.open_stream(handle)
            batches, schema = iter(reader), reader.schema
        except pa.ArrowInvalid:  # Feather v1 (vecchio formato): si legge tutto in una volta
            import pyarrow.feather as feather
            handle.seek(0)
            table = feather.read_table(handle)
            batches, schema = iter(table.to_batches()), table.schema
    return _write_batches(batches, schema, target, progress)


def _copy_parquet(source, target: Path) -> int:
    if isinstance(source, (str, Path)):
        shutil.copyfile(source, target)
    else:
        source.seek(0)
        with open(target, "wb") as out:
            shutil.copyfileobj(source, out)
    return pq.read_metadata(target).num_rows


def import_table(source, filename: str, progress=None) -> Path:
    """Converte un file (percorso sul disco o file caricato dal browser) in ``connectome/<nome>.parquet``.

    Scrive prima un file temporaneo: se la conversione fallisce non resta nulla di parziale.
    Un file con lo stesso nome viene sostituito.
    """
    lower = str(filename).lower()
    target = folder(CONNECTOME) / f"{safe_stem(filename)}.parquet"
    partial = target.with_name(target.name + ".part")
    try:
        if lower.endswith(".parquet"):
            _copy_parquet(source, partial)
        elif lower.endswith((".feather", ".arrow", ".ipc")):
            _ipc_to_parquet(source, partial, progress)
        else:
            _csv_to_parquet(_opener(source, lower.endswith(".gz")), partial, progress)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target


# ============================================================================ elenco, anteprima, eliminazione

@dataclass(frozen=True)
class TableInfo:
    name: str          # nome del file Parquet
    path: Path
    rows: int
    columns: tuple[str, ...]
    size_mb: float
    modified: float    # timestamp (s): fa parte delle chiavi di cache


def list_tables() -> list[TableInfo]:
    out = []
    for path in sorted(folder(CONNECTOME).glob("*.parquet")):
        try:
            meta = pq.read_metadata(path)
        except Exception:  # file danneggiato o non Parquet: lo mostriamo comunque per poterlo eliminare
            out.append(TableInfo(path.name, path, 0, (), path.stat().st_size / 1e6, path.stat().st_mtime))
            continue
        out.append(TableInfo(path.name, path, meta.num_rows, tuple(meta.schema.to_arrow_schema().names),
                             path.stat().st_size / 1e6, path.stat().st_mtime))
    return out


def preview(path: Path, n: int = 20) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=n):
        return batch.to_pandas()
    return parquet.schema_arrow.empty_table().to_pandas()


def delete(kind: str, name: str) -> None:
    _inside(kind, name).unlink(missing_ok=True)


# ============================================================================ libreria di stimoli

def parse_waveform(data: bytes) -> tuple[list[float], list[float]]:
    """Legge una forma d'onda CSV: prima colonna numerica = tempo (ms), seconda = valore (pA)."""
    table = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    numeric = table.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise ValueError("servono due colonne numeriche (tempo in ms, valore in pA)")
    samples = numeric.iloc[:, :2].dropna()
    if len(samples) < 2:
        raise ValueError("servono almeno due campioni")
    return samples.iloc[:, 0].astype(float).tolist(), samples.iloc[:, 1].astype(float).tolist()


def save_stimulus(filename: str, data: bytes) -> str:
    """Valida e salva una forma d'onda in ``stimuli/`` (formato normalizzato ``t_ms,valore_pA``)."""
    t, v = parse_waveform(data)
    name = f"{safe_stem(filename)}.csv"
    pd.DataFrame({"t_ms": t, "valore_pA": v}).to_csv(_inside(STIMULI, name), index=False)
    return name


def list_stimuli() -> list[str]:
    return sorted(p.name for p in folder(STIMULI).glob("*.csv"))


def load_stimulus(name: str) -> dict:
    t, v = parse_waveform(_inside(STIMULI, name).read_bytes())
    return {"t": t, "v": v, "source": name}
