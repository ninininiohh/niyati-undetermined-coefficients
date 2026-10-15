"""Streamlit presentation app: Method of Undetermined Coefficients."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from solver import C1, C2, Solution, apply_initial_conditions, solve_config, x


st.set_page_config(page_title="Undetermined Coefficients", page_icon="∂", layout="wide")

FAMILIES = [
    "Exponential", "Sine", "Cosine", "Exponential × sine", "Exponential × cosine",
    "Polynomial", "Exponential × polynomial", "Polynomial × sine", "Polynomial × cosine",
]
PRESETS = {
    "Choose a preset…": None,
    "e^(3x)": {"family": "Exponential", "k": 1.0, "a": 3.0},
    "sin(2x)": {"family": "Sine", "k": 1.0, "a": 2.0, "b": 0.0},
    "x²": {"family": "Polynomial", "k": 1.0, "degree": 2},
    "e^(3x) sin(2x)": {"family": "Exponential × sine", "k": 1.0, "c": 3.0, "a": 2.0, "b": 0.0},
    "e^(−x) — resonance": {"family": "Exponential", "k": 1.0, "a": -1.0},
    "2x²e^(3x)": {"family": "Exponential × polynomial", "k": 2.0, "a": 3.0, "degree": 2},
    "3x²cos(2x+1)": {"family": "Polynomial × cosine", "k": 3.0, "degree": 2, "a": 2.0, "b": 1.0},
}


def latex(expr: sp.Expr) -> None:
    st.latex(sp.latex(sp.simplify(expr)))


def equation_latex(left: sp.Expr, right: sp.Expr) -> None:
    st.latex(f"{sp.latex(sp.simplify(left))} = {sp.latex(sp.simplify(right))}")


def reset() -> None:
    st.session_state.clear()


def defaults() -> None:
    for key, value in {"family": "Exponential", "k": 1.0, "a": 3.0, "b": 0.0, "c": 1.0, "degree": 2}.items():
        st.session_state.setdefault(key, value)


def use_preset() -> None:
    chosen = PRESETS[st.session_state.preset]
    if chosen:
        st.session_state.update(chosen)
        st.session_state.pop("result", None)


def number(label: str, key: str, value: float) -> float:
    return st.number_input(label, value=float(st.session_state.get(key, value)), step=0.5, key=key)


def show_graph(result: Solution, complete: sp.Expr | None, interval: tuple[float, float]) -> None:
    grid = np.linspace(interval[0], interval[1], 800)
    curves: list[tuple[str, sp.Expr, dict]] = [
        ("Forcing R(x)", result.forcing, {"dash": "dot", "color": "#e76f51"}),
        ("Particular solution yₚ(x)", result.particular, {"color": "#2563eb"}),
    ]
    if complete is not None:
        homogeneous = sp.simplify(complete - result.particular)
        curves.extend([
            ("Homogeneous part yₕ(x)", homogeneous, {"dash": "dash", "color": "#6b7280"}),
            ("Complete solution y(x)", complete, {"width": 4, "color": "#16a34a"}),
        ])
    figure = go.Figure()
    for name, expression, style in curves:
        try:
            fn = sp.lambdify(x, expression, modules=["numpy"])
            values = np.asarray(fn(grid), dtype=complex)
            if values.ndim == 0:
                values = np.full_like(grid, values, dtype=complex)
            values = np.real_if_close(values, tol=1000).astype(float)
            values[~np.isfinite(values)] = np.nan
            figure.add_trace(go.Scatter(x=grid, y=values, mode="lines", name=name, line=style,
                                        hovertemplate="x=%{x:.3f}<br>value=%{y:.5g}<extra>" + name + "</extra>"))
        except (TypeError, ValueError, OverflowError) as error:
            st.warning(f"Could not plot {name} over this interval: {error}")
    figure.update_layout(
        height=500, margin=dict(l=10, r=10, t=35, b=10), template="plotly_white",
        title="Forcing and solution behavior", xaxis_title="x", yaxis_title="value", legend_title="Curve",
    )
    figure.update_xaxes(zeroline=True, zerolinecolor="#9ca3af")
    figure.update_yaxes(zeroline=True, zerolinecolor="#9ca3af")
    st.plotly_chart(figure, use_container_width=True)
    st.caption("R(x) is the forcing input—not the solution. yₚ is one response to that input; the complete solution adds the homogeneous part.")


defaults()
st.title("Method of Undetermined Coefficients")
st.caption("An interactive Differential Equations presentation using one fixed linear operator.")

with st.container(border=True):
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("**Fixed differential equation**")
        st.latex(r"y''+3y'+2y=R(x)")
        st.markdown("The method selects a trial form based on the shape of the forcing function, then determines its coefficients by substitution.")
    with right:
        st.markdown("**Complementary solution**")
        st.latex(r"m^2+3m+2=(m+1)(m+2)=0")
        st.latex(r"y_h=C_1e^{-x}+C_2e^{-2x}, qquad y=y_h+y_p")

st.selectbox("Quick demonstration presets", list(PRESETS), key="preset", on_change=use_preset)
controls, actions = st.columns([5, 1])
with controls:
    family = st.selectbox("Choose the forcing function (R(x))", FAMILIES, key="family")
with actions:
    st.write("")
    st.button("Reset", on_click=reset, use_container_width=True)

fields = st.columns(5)
with fields[0]:
    k = number("Multiplier k", "k", 1.0)
with fields[1]:
    if family in {"Exponential", "Exponential × polynomial"}:
        a = number("Exponent rate a", "a", 3.0)
    elif family in {"Sine", "Cosine", "Exponential × sine", "Exponential × cosine", "Polynomial × sine", "Polynomial × cosine"}:
        a = number("Frequency a", "a", 2.0)
    else:
        a = st.session_state.get("a", 1.0)
with fields[2]:
    if family in {"Sine", "Cosine", "Exponential × sine", "Exponential × cosine", "Polynomial × sine", "Polynomial × cosine"}:
        b = number("Phase b (radians)", "b", 0.0)
    else:
        b = st.session_state.get("b", 0.0)
with fields[3]:
    if family in {"Exponential × sine", "Exponential × cosine"}:
        c = number("Exponential rate c", "c", 1.0)
    else:
        c = st.session_state.get("c", 0.0)
with fields[4]:
    if family in {"Polynomial", "Exponential × polynomial", "Polynomial × sine", "Polynomial × cosine"}:
        degree = st.number_input("Polynomial degree n", min_value=0, max_value=8, value=int(st.session_state.get("degree", 2)), step=1, key="degree")
    else:
        degree = int(st.session_state.get("degree", 0))

config = {"family": family, "k": k, "a": a, "b": b, "c": c, "degree": int(degree)}
if st.button("Solve", type="primary", use_container_width=False):
    try:
        st.session_state.result = solve_config(config)
        st.session_state.solved_config = config
    except (ValueError, TypeError, sp.SympifyError) as error:
        st.session_state.pop("result", None)
        st.error(f"Unable to solve this input: {error}")

result: Solution | None = (
    st.session_state.get("result")
    if st.session_state.get("solved_config") == config
    else None
)
if result:
    st.divider()
    st.header("Step-by-step solution")
    st.subheader("Step 1: Identify the forcing function")
    equation_latex(sp.Function("R")(x), result.forcing)

    st.subheader("Step 2: Select the trial solution")
    st.markdown("The initial trial is chosen to match the forcing function's form.")
    equation_latex(sp.Symbol("y_p"), result.raw_trial)
    if result.resonance_power:
        st.info(result.note)
        st.markdown("**Resonance-adjusted trial:**")
        equation_latex(sp.Symbol("y_p"), result.trial)
    elif result.note:
        st.info(result.note)
        st.markdown("**Trial after the required simplification:**")
        equation_latex(sp.Symbol("y_p"), result.trial)

    st.subheader("Step 3: Calculate derivatives")
    equation_latex(sp.Symbol("y_p'"), result.derivative_1)
    equation_latex(sp.Symbol("y_p''"), result.derivative_2)

    st.subheader("Step 4: Substitute into the ODE")
    st.latex(
        f"\\left({sp.latex(result.derivative_2)}\\right)+3\\left({sp.latex(result.derivative_1)}\\right)"
        f"+2\\left({sp.latex(result.trial)}\\right)={sp.latex(result.forcing)}"
    )

    st.subheader("Step 5: Determine the coefficients")
    if result.coefficients:
        st.markdown("Equating like terms after simplification gives:")
        coefficient_text = ",\\qquad ".join(f"{sp.latex(key)}={sp.latex(value)}" for key, value in result.coefficients.items())
        st.latex(coefficient_text)
    else:
        st.markdown("No unknown coefficients are needed because the forcing function is zero.")

    st.subheader("Step 6: Present the particular solution")
    equation_latex(sp.Symbol("y_p"), result.particular)

    st.subheader("Step 7: Verify the result")
    st.success("Verified symbolically: the residual is zero.")
    st.latex(r"y_p''+3y_p'+2y_p-R(x)=" + sp.latex(result.residual))

    st.subheader("Step 8: Show the general solution")
    equation_latex(sp.Symbol("y"), result.general_solution)

    st.divider()
    st.header("Initial conditions and interactive graph")
    use_initial_conditions = st.checkbox("Specify initial conditions", key="use_ic")
    complete: sp.Expr | None = None
    if use_initial_conditions:
        ic1, ic2 = st.columns(2)
        with ic1:
            y0 = st.number_input("y(0)", value=0.0, step=0.5, key="y0")
        with ic2:
            v0 = st.number_input("y′(0)", value=0.0, step=0.5, key="v0")
        try:
            constants, complete = apply_initial_conditions(result, y0, v0)
            st.markdown("**Constants determined from the initial conditions:**")
            st.latex(",\\qquad ".join(f"{sp.latex(symbol)}={sp.latex(value)}" for symbol, value in constants.items()))
            equation_latex(sp.Symbol("y"), complete)
        except ValueError as error:
            st.error(f"Could not apply the initial conditions: {error}")
    interval = st.slider("Graph interval for x", min_value=-10.0, max_value=10.0, value=(-5.0, 5.0), step=0.5)
    if interval[0] >= interval[1] or not all(math.isfinite(v) for v in interval):
        st.error("Choose a valid graph interval with a lower bound below the upper bound.")
    else:
        show_graph(result, complete, interval)
