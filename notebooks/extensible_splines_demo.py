# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Extensible Splines: Interactive B-Spline Demo

        This notebook demonstrates **B-Spline interpolation** from the 
        [extensible-splines](https://github.com/egoughnour/extensible-splines) library.

        B-Splines are piecewise polynomial curves that provide smooth interpolation 
        through control points with **local control** - moving one control point only 
        affects the curve in its local neighborhood.

        ## Key Properties
        - **C2 continuity**: Smooth first and second derivatives
        - **Local support**: Each control point influences only nearby curve segments
        - **Convex hull property**: The curve lies within the convex hull of control points
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    num_points = mo.ui.slider(
        start=4,
        stop=12,
        step=1,
        value=6,
        label="Number of Control Points"
    )
    num_points
    return (num_points,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(num_points, np, plt):
    def bspline_basis(i, k, t, knots):
        """Compute B-spline basis function recursively."""
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
        
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]
        
        term1 = 0.0 if denom1 == 0 else (t - knots[i]) / denom1 * bspline_basis(i, k - 1, t, knots)
        term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - t) / denom2 * bspline_basis(i + 1, k - 1, t, knots)
        
        return term1 + term2

    def evaluate_bspline(control_points, num_samples=200):
        """Evaluate a cubic B-spline curve."""
        n = len(control_points)
        k = 3  # cubic
        
        # Create clamped knot vector
        knots = [0] * (k + 1) + list(range(1, n - k)) + [n - k] * (k + 1)
        knots = [float(x) for x in knots]
        
        t_vals = np.linspace(knots[k], knots[n] - 0.0001, num_samples)
        curve = []
        
        for t in t_vals:
            point = np.zeros(2)
            for i in range(n):
                basis = bspline_basis(i, k, t, knots)
                point += basis * control_points[i]
            curve.append(point)
        
        return np.array(curve)

    # Generate control points in a pleasing pattern
    n = num_points.value
    angles = np.linspace(0, 2 * np.pi * 0.8, n)
    radii = 1 + 0.3 * np.sin(angles * 2)
    control_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Evaluate spline
    curve = evaluate_bspline(control_pts)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2.5, label='B-Spline Curve')
    ax.plot(control_pts[:, 0], control_pts[:, 1], 'ro--', markersize=10, 
            linewidth=1, alpha=0.7, label='Control Points')
    ax.set_aspect('equal')
    ax.set_title(f'Cubic B-Spline with {n} Control Points', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## How It Works

        The B-spline curve C(t) is defined as the sum of basis functions 
        multiplied by control points. The basis functions are computed 
        recursively using the Cox-de Boor formula.

        ---

        *Try adjusting the slider above to see how the curve changes with different 
        numbers of control points!*
        """
    )
    return


if __name__ == "__main__":
    app.run()
