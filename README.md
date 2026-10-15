# Method of Undetermined Coefficients presentation app

A classroom-friendly Streamlit app for the fixed equation

\[
y''+3y'+2y=R(x).
\]

It uses SymPy to construct a trial particular solution, solve its undetermined coefficients, verify the residual, and handle resonance with the homogeneous roots \(-1\) and \(-2\). Plotly provides zoomable solution graphs.

## Install and run (Windows PowerShell)

From this folder, run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once for the current terminal and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then open the local URL Streamlit displays (normally `http://localhost:8501`).

## Structure

- `app.py` — Streamlit presentation interface, dynamic controls, educational steps, and Plotly chart.
- `solver.py` — testable symbolic engine. It defines the fixed operator, chooses trials, detects resonance, solves coefficients with `sympy.solve_undetermined_coeffs`, and applies optional initial conditions.
- `test_solver.py` — residual tests for core, resonance, polynomial-exponential, and polynomial-trigonometric examples.

## Test command

With the virtual environment active:

```powershell
pytest -q
```

The test suite checks that substitution gives zero residual for these presentation presets: `e^(3x)`, `sin(2x)`, `x^2`, `e^(3x)sin(2x)`, and resonant `e^(-x)`. It also covers `2x^2e^(3x)`, `3x^2cos(2x+1)`, and a zero-frequency reduction.

## Verification performed

Before delivery, the symbolic engine was executed directly against all eight cases listed above. Every case passed the assertion `simplify(L(y_p) - R(x)) == 0`; the resonant `e^(-x)` case produced `y_p = xe^(-x)`. `app.py`, `solver.py`, and `test_solver.py` also passed Python syntax compilation. The full `pytest -q` command is provided above for local repeatability.

## Expected preset behavior

- `e^(3x)`: the trial is `Ae^(3x)`; no resonance.
- `sin(2x)`: the trial combines sine and cosine.
- `x²`: the trial is a quadratic polynomial.
- `e^(3x) sin(2x)`: the trial is an exponential times combined sine/cosine.
- `e^(−x) — resonance`: `Ae^(−x)` overlaps with `y_h`, so the app uses `Axe^(−x)` and finds `y_p=xe^(−x)`.
