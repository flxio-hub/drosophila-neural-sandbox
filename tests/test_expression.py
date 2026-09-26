import numpy as np
import pytest

from sandbox.stimuli import ExpressionError, evaluate_expression, time_vector

T = time_vector(100.0, 0.1)


@pytest.mark.parametrize("expr", [
    "15 * np.sin(2 * np.pi * 0.05 * t) * (t > 20)",
    "10 * step(t - 30) - 10 * step(t - 60)",
    "rect(t, 10, 20) * 50 + gauss(t, 50, 5) * 20",
    "where(t < 50, 10, -10)",
    "alpha(t, 10, 5) * 40 + pulses(t, 100, 2) * 30",
    "5 + noise(2)",
    "(t > 10) & (t < 20)",
    "42",
])
def test_valid_expressions_return_array(expr):
    out = evaluate_expression(expr, T, seed=0)
    assert out.shape == T.shape and out.dtype == float and np.all(np.isfinite(out))


def test_values_are_correct():
    out = evaluate_expression("15 * np.sin(2 * np.pi * 0.05 * t) * (t > 20)", T)
    expected = 15 * np.sin(2 * np.pi * 0.05 * T) * (T > 20)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('secret.txt')",
    "np.os",
    "np.load('x.npy')",
    "t.__class__",
    "().__class__.__bases__",
    "(lambda: 1)()",
    "[x for x in range(10)]",
    "t[0]",
    "'abc' * 10",
    "exec('1')",
    "np.sin(t, out=t)",
    "sin.__call__(t)",
    "globals()",
])
def test_unsafe_expressions_are_rejected(expr):
    with pytest.raises(ExpressionError):
        evaluate_expression(expr, T)


def test_huge_power_fails_fast_instead_of_hanging():
    with pytest.raises(ExpressionError):
        evaluate_expression("9**9**9**9", T)


def test_non_finite_result_is_rejected():
    with pytest.raises(ExpressionError):
        evaluate_expression("1 / (t - t)", T)


def test_length_limit():
    with pytest.raises(ExpressionError):
        evaluate_expression("+".join(["t"] * 500), T)


def test_time_vector_is_not_mutated():
    t = T.copy()
    evaluate_expression("t * 2", t)
    np.testing.assert_array_equal(t, T)
