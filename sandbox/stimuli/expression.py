"""Valutazione sicura di equazioni di stimolo scritte dall'utente.

L'espressione (es. ``15 * np.sin(2 * np.pi * 0.05 * t) * (t > 20)``) viene
analizzata con ``ast`` e confrontata con una whitelist PRIMA di essere
eseguita. Non si usa mai ``eval`` su testo non validato:

* sono ammessi solo numeri, operatori aritmetici/confronto/bitwise e chiamate
  a funzioni in whitelist (NumPy o helper dedicati);
* nessun accesso ad attributi diversi da ``np.<funzione ammessa>``, nessuna
  stringa, lambda, comprehension, indicizzazione o nome che inizi con ``_``;
* le costanti intere vengono convertite in float, così ``9**9**9`` genera un
  OverflowError immediato invece di bloccare il processo;
* lunghezza massima dell'espressione limitata.

Variabili disponibili: ``t`` (ms, array), ``dt`` (ms), ``T`` (durata, ms).
"""

import ast
import math
from types import SimpleNamespace

import numpy as np

MAX_EXPRESSION_LENGTH = 400

_NUMPY_FUNCTIONS = (
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "log2", "sqrt", "abs",
    "sign", "floor", "ceil", "minimum", "maximum", "clip", "where",
    "heaviside", "mod", "power", "hypot", "deg2rad", "rad2deg",
)
_NUMPY_CONSTANTS = ("pi", "e")

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name,
    ast.Load, ast.Constant, ast.Attribute,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Invert, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)

HELP_TEXT = """
**Variabili**: `t` (ms), `dt`, `T` (durata) · **Costanti**: `pi`, `e`
**Funzioni** (anche con prefisso `np.`): `sin cos tan exp log sqrt abs tanh clip where heaviside minimum maximum …`
**Helper**:
`step(x)` gradino di Heaviside · `rect(t, a, b)` finestra [a, b) ·
`gauss(t, mu, sigma)` · `alpha(t, t0, tau)` funzione alfa ·
`pulses(t, freq_hz, width_ms)` treno di impulsi · `noise(sigma)` rumore gaussiano
**Logica**: usa `&`, `|`, `~` e i confronti (`t > 20`), non `and/or`.
"""


class ExpressionError(ValueError):
    """Espressione non valida o non sicura."""


class _FloatConstants(ast.NodeTransformer):
    """Converte gli interi in float: blocca potenze intere gigantesche (DoS)."""

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError("Sono ammesse solo costanti numeriche.")
        return ast.copy_location(ast.Constant(float(node.value)), node)


def _helpers(t: np.ndarray, seed):
    rng = np.random.default_rng(seed)

    def step(x):
        return np.heaviside(x, 1.0)

    def rect(x, a, b):
        return ((x >= a) & (x < b)).astype(float)

    def gauss(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def alpha(x, t0, tau):
        s = np.clip((x - t0) / tau, 0.0, None)
        return s * np.exp(1.0 - s)

    def pulses(x, freq_hz, width_ms):
        period = 1000.0 / freq_hz
        return ((x % period) < width_ms).astype(float)

    def noise(sigma):
        return sigma * rng.standard_normal(t.size)

    return {"step": step, "rect": rect, "gauss": gauss, "alpha": alpha, "pulses": pulses, "noise": noise}


def _validate(tree: ast.AST, allowed_names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(f"Costrutto non permesso: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ExpressionError(f"Nome sconosciuto: '{node.id}'")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
                raise ExpressionError("Accesso ad attributi consentito solo come np.<funzione>.")
            if node.attr not in _NUMPY_FUNCTIONS and node.attr not in _NUMPY_CONSTANTS:
                raise ExpressionError(f"np.{node.attr} non è in whitelist.")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in ("t", "dt", "T", "np", "pi", "e"):
                    raise ExpressionError(f"'{func.id}' non è una funzione.")
            elif not (isinstance(func, ast.Attribute) and func.attr in _NUMPY_FUNCTIONS):
                raise ExpressionError("Sono ammesse solo chiamate a funzioni in whitelist.")
            if node.keywords:  # es. out=t modificherebbe array in-place
                raise ExpressionError("Gli argomenti con nome (kwargs) non sono ammessi.")


def evaluate_expression(expression: str, t: np.ndarray, seed=None) -> np.ndarray:
    """Valida e valuta ``expression`` sul vettore dei tempi ``t``; restituisce I(t) in pA."""
    expression = expression.strip()
    if not expression:
        raise ExpressionError("L'espressione è vuota.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Espressione troppo lunga (max {MAX_EXPRESSION_LENGTH} caratteri).")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Errore di sintassi: {exc.msg}") from None

    helpers = _helpers(t, seed)
    safe_np = SimpleNamespace(
        **{name: getattr(np, name) for name in _NUMPY_FUNCTIONS + _NUMPY_CONSTANTS}
    )
    namespace = {
        "t": t.copy(),  # copia difensiva: l'espressione non può alterare il vettore dei tempi
        "dt": float(t[1] - t[0]) if t.size > 1 else 1.0,
        "T": float(t[-1] + (t[1] - t[0])) if t.size > 1 else 0.0,
        "np": safe_np,
        "pi": math.pi,
        "e": math.e,
        **{name: getattr(np, name) for name in _NUMPY_FUNCTIONS},
        **helpers,
    }
    _validate(tree, set(namespace))
    tree = ast.fix_missing_locations(_FloatConstants().visit(tree))

    try:
        with np.errstate(all="ignore"):
            result = eval(compile(tree, "<stimolo>", "eval"), {"__builtins__": {}}, namespace)
            current = np.broadcast_to(np.asarray(result, dtype=float), t.shape).copy()
    except ExpressionError:
        raise
    except Exception as exc:  # errori numerici o di tipo generati dall'espressione
        raise ExpressionError(f"{type(exc).__name__}: {exc}") from None

    if not np.all(np.isfinite(current)):
        raise ExpressionError("Il risultato contiene valori non finiti (inf/NaN): controlla divisioni, log e potenze.")
    return current
