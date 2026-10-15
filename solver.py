"""Symbolic engine for the Method of Undetermined Coefficients demo.

The UI is deliberately kept out of this module so that the calculations can be
tested with ordinary Python/pytest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

x = sp.symbols("x", real=True)
C1, C2 = sp.symbols("C_1 C_2")


@dataclass
class Solution:
    family: str
    forcing: sp.Expr
    raw_trial: sp.Expr
    trial: sp.Expr
    derivative_1: sp.Expr
    derivative_2: sp.Expr
    coefficients: dict[sp.Symbol, sp.Expr]
    particular: sp.Expr
    residual: sp.Expr
    resonance_power: int
    note: str
    general_solution: sp.Expr


def as_exact(value: Any) -> sp.Expr:
    """Convert a Streamlit numeric value to a small, readable exact number."""
    return sp.Rational(str(value)).limit_denominator(10_000)


def operator(y: sp.Expr) -> sp.Expr:
    """The fixed left-hand-side differential operator L[y]."""
    return sp.diff(y, x, 2) + 3 * sp.diff(y, x) + 2 * y


def resonance_power(rate: sp.Expr) -> int:
    """Multiplicity of an exponential rate in (m+1)(m+2)."""
    return int(sp.simplify(rate + 1) == 0) + int(sp.simplify(rate + 2) == 0)


def polynomial_symbols(degree: int, prefix: str = "A") -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(f"{prefix}_{j}") for j in range(degree + 1))


def exp_polynomial_problem(
    amplitude: sp.Expr, rate: sp.Expr, degree: int, family: str, note: str = ""
) -> tuple[sp.Expr, sp.Expr, sp.Expr, tuple[sp.Symbol, ...], int, str]:
    """Build R and an exponential-polynomial trial, including resonance."""
    forcing = sp.expand(amplitude * x**degree * sp.exp(rate * x))
    power = resonance_power(rate)
    unknowns = polynomial_symbols(degree)
    polynomial = sum(symbol * x**j for j, symbol in enumerate(unknowns))
    raw_trial = sp.exp(rate * x) * polynomial
    trial = sp.expand(x**power * raw_trial)
    if power:
        resonance_note = (
            f"The rate {sp.latex(rate)} is a root of the characteristic equation, "
            f"so the usual trial overlaps with the complementary solution. "
            f"Multiplying by x^{power} makes it linearly independent."
        )
        note = f"{note} {resonance_note}".strip()
    return forcing, raw_trial, trial, unknowns, power, note


def build_problem(config: dict[str, Any]) -> tuple[sp.Expr, sp.Expr, sp.Expr, tuple[sp.Symbol, ...], int, str]:
    """Construct R(x) and a suitable trial form for a UI configuration."""
    family = config["family"]
    k = as_exact(config.get("k", 1))
    a = as_exact(config.get("a", 1))
    b = as_exact(config.get("b", 0))
    c = as_exact(config.get("c", 0))
    degree = int(config.get("degree", 0))
    if not 0 <= degree <= 8:
        raise ValueError("The polynomial degree must be between 0 and 8.")

    if family == "Exponential":
        return exp_polynomial_problem(k, a, 0, family)
    if family == "Polynomial":
        return exp_polynomial_problem(k, sp.Integer(0), degree, family)
    if family == "Exponential × polynomial":
        return exp_polynomial_problem(k, a, degree, family)

    trig_is_sine = family in {"Sine", "Exponential × sine", "Polynomial × sine"}
    trig = sp.sin if trig_is_sine else sp.cos
    has_exp = family in {"Exponential × sine", "Exponential × cosine"}
    has_poly = family in {"Polynomial × sine", "Polynomial × cosine"}
    rate = c if has_exp else sp.Integer(0)
    poly_degree = degree if has_poly else 0

    # At zero frequency, sine/cosine is a constant and the usual A sin+B cos
    # trial becomes linearly dependent.  Reduce it to an exp-polynomial case.
    if a == 0:
        reduced_amplitude = sp.simplify(k * trig(b))
        reduction = (
            "Zero frequency makes the trigonometric factor a constant; the forcing "
            "has been reduced to the corresponding exponential-polynomial form."
        )
        if reduced_amplitude == 0:
            return sp.Integer(0), sp.Integer(0), sp.Integer(0), tuple(), 0, (
                reduction + " The forcing is zero, so y_p=0 is a valid particular solution."
            )
        return exp_polynomial_problem(reduced_amplitude, rate, poly_degree, family, reduction)

    forcing = sp.expand(k * x**poly_degree * sp.exp(rate * x) * trig(a * x + b))
    a_symbols = polynomial_symbols(poly_degree, "A")
    b_symbols = polynomial_symbols(poly_degree, "B")
    combined = sum(
        x**j * (A * sp.sin(a * x + b) + B * sp.cos(a * x + b))
        for j, (A, B) in enumerate(zip(a_symbols, b_symbols))
    )
    raw_trial = sp.exp(rate * x) * combined
    # This test is deliberately general: a complex rate would have to equal a
    # characteristic root for a trig trial to resonate.  For nonzero real a it
    # evaluates to zero, but keeping it here documents the general rule.
    power = resonance_power(rate + sp.I * a)
    trial = sp.expand(x**power * raw_trial)
    note = ""
    if power:
        note = (
            "The complex exponential rate underlying this trigonometric forcing "
            "overlaps with a characteristic root, so the trial was multiplied by x."
        )
    return forcing, raw_trial, trial, a_symbols + b_symbols, power, note


def solve_config(config: dict[str, Any]) -> Solution:
    """Find the undetermined coefficients and return every teaching step value."""
    forcing, raw_trial, trial, unknowns, power, note = build_problem(config)
    if forcing == 0:
        particular = sp.Integer(0)
        coeffs: dict[sp.Symbol, sp.Expr] = {}
    else:
        equation = sp.expand_trig(sp.expand(operator(trial) - forcing))
        coeffs = sp.solve_undetermined_coeffs(equation, unknowns, x) if unknowns else {}
        if coeffs is None:
            raise ValueError("SymPy could not determine the trial coefficients for these inputs.")
        # solve_undetermined_coeffs sometimes omits a coefficient whose value is zero.
        coeffs = {symbol: sp.simplify(coeffs.get(symbol, 0)) for symbol in unknowns}
        particular = sp.simplify(trial.subs(coeffs))

    d1 = sp.simplify(sp.diff(trial, x))
    d2 = sp.simplify(sp.diff(trial, x, 2))
    residual = sp.simplify(sp.expand_trig(operator(particular) - forcing))
    if residual != 0:
        raise ValueError("The symbolic residual did not simplify to zero; please change the inputs.")
    general = sp.simplify(C1 * sp.exp(-x) + C2 * sp.exp(-2 * x) + particular)
    return Solution(
        family=config["family"], forcing=forcing, raw_trial=raw_trial, trial=trial,
        derivative_1=d1, derivative_2=d2, coefficients=coeffs, particular=particular,
        residual=residual, resonance_power=power, note=note, general_solution=general,
    )


def apply_initial_conditions(solution: Solution, y0: Any, v0: Any) -> tuple[dict[sp.Symbol, sp.Expr], sp.Expr]:
    """Solve C1 and C2 from y(0)=y0 and y'(0)=v0."""
    y0_exact, v0_exact = as_exact(y0), as_exact(v0)
    equations = [
        sp.Eq(solution.general_solution.subs(x, 0), y0_exact),
        sp.Eq(sp.diff(solution.general_solution, x).subs(x, 0), v0_exact),
    ]
    constants = sp.solve(equations, (C1, C2), dict=True)
    if not constants:
        raise ValueError("The initial conditions could not be solved.")
    values = constants[0]
    return values, sp.simplify(solution.general_solution.subs(values))
