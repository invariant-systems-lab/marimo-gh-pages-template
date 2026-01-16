# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Interactive B-Spline Editor

        Use the sliders below to adjust control point positions.
        The B-spline curve updates in real-time.

        *Powered by [extensible-splines](https://github.com/egoughnour/extensible-splines)*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    num_points = mo.ui.slider(
        start=4,
        stop=10,
        step=1,
        value=6,
        label="Number of Control Points"
    )
    num_points
    return (num_points,)


@app.cell(hide_code=True)
def _(mo, num_points):
    # Create sliders for X and Y coordinates of each control point
    n = num_points.value

    # Default positions in a circular pattern
    import numpy as np
    angles = np.linspace(0, 2 * np.pi * 0.85, n)
    default_x = 350 + 200 * np.cos(angles)
    default_y = 300 + 200 * np.sin(angles)

    x_sliders = [
        mo.ui.slider(start=50, stop=650, value=int(default_x[i]), label=f"P{i+1} X")
        for i in range(n)
    ]
    y_sliders = [
        mo.ui.slider(start=50, stop=550, value=int(default_y[i]), label=f"P{i+1} Y")
        for i in range(n)
    ]

    # Display sliders in a compact grid
    slider_rows = []
    for i in range(n):
        slider_rows.append(
            mo.hstack([
                mo.md(f"**Point {i+1}:**"),
                x_sliders[i],
                y_sliders[i]
            ], gap="1rem", align="center")
        )

    mo.vstack(slider_rows, gap="0.5rem")
    return x_sliders, y_sliders, n


@app.cell(hide_code=True)
def _(np, x_sliders, y_sliders, n):
    def bspline_basis(i, k, t, knots):
        """Compute B-spline basis function recursively (Cox-de Boor)."""
        if k == 0:
            return 1.0 if knots[i] <= t < knots[i + 1] else 0.0

        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = 0.0 if denom1 == 0 else (t - knots[i]) / denom1 * bspline_basis(i, k - 1, t, knots)
        term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - t) / denom2 * bspline_basis(i + 1, k - 1, t, knots)

        return term1 + term2

    def evaluate_bspline(control_points, num_samples=150):
        """Evaluate a cubic B-spline curve."""
        num_pts = len(control_points)
        if num_pts < 4:
            return []

        k = 3  # cubic
        knots = [0] * (k + 1) + list(range(1, num_pts - k)) + [num_pts - k] * (k + 1)
        knots = [float(x) for x in knots]

        t_vals = np.linspace(knots[k], knots[num_pts] - 0.0001, num_samples)
        curve = []

        for t in t_vals:
            point = np.zeros(2)
            for i in range(num_pts):
                basis = bspline_basis(i, k, t, knots)
                point += basis * control_points[i]
            curve.append(point)

        return curve

    # Get control points from sliders
    control_points = np.array([
        [x_sliders[i].value, y_sliders[i].value]
        for i in range(n)
    ])

    curve_points = evaluate_bspline(control_points)
    return control_points, curve_points, bspline_basis, evaluate_bspline


@app.cell(hide_code=True)
def _(mo, control_points, curve_points, n):
    # Generate SVG path for the curve
    curve_path = ""
    if len(curve_points) > 1:
        curve_path = f"M {curve_points[0][0]:.1f} {curve_points[0][1]:.1f} "
        for pt in curve_points[1:]:
            curve_path += f"L {pt[0]:.1f} {pt[1]:.1f} "

    # Generate control polygon path
    polygon_path = ""
    if len(control_points) > 1:
        polygon_path = f"M {control_points[0][0]:.1f} {control_points[0][1]:.1f} "
        for pt in control_points[1:]:
            polygon_path += f"L {pt[0]:.1f} {pt[1]:.1f} "

    # Generate control point circles
    circles = "".join(
        f'<circle cx="{control_points[i][0]:.1f}" cy="{control_points[i][1]:.1f}" r="8" fill="#14b8a6" stroke="#fff" stroke-width="2"/><text x="{control_points[i][0]:.1f}" y="{control_points[i][1] - 15:.1f}" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="Inter, sans-serif">P{i+1}</text>'
        for i in range(n)
    )

    canvas_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem; margin-top: 1rem;">
        <svg width="700" height="600" style="background: #1e293b; border-radius: 12px;">
            <!-- Grid -->
            <defs>
                <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                    <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#334155" stroke-width="0.5"/>
                </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)"/>

            <!-- Control polygon -->
            <path d="{polygon_path}" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="5,5"/>

            <!-- B-spline curve -->
            <path d="{curve_path}" fill="none" stroke="#6366f1" stroke-width="3" stroke-linecap="round"/>

            <!-- Control points -->
            {circles}
        </svg>

        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="color: #94a3b8; font-size: 14px;">Control Points: <strong style="color: #14b8a6;">{n}</strong></span>
            <span style="color: #94a3b8; font-size: 14px;">|</span>
            <span style="color: #94a3b8; font-size: 14px;">Curve: <strong style="color: #6366f1;">Cubic B-Spline (C² continuous)</strong></span>
        </div>
    </div>
    """

    mo.Html(canvas_html)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## About B-Splines

        B-splines (Basis splines) are piecewise polynomial curves with several important properties:

        - **Local control**: Moving one control point only affects the curve nearby
        - **Smoothness**: Cubic B-splines have C² continuity (smooth curvature)
        - **Convex hull**: The curve stays within the convex hull of control points

        The curve is computed using the **Cox-de Boor recursion formula**, which efficiently
        evaluates the basis functions at each parameter value.

        [View on GitHub](https://github.com/egoughnour/extensible-splines) |
        [Back to Portfolio](https://erikgoughnour.com)
        """
    )
    return


if __name__ == "__main__":
    app.run()
