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
    import json
    return mo, np, json


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Interactive B-Spline Editor

        **Click** on the canvas to add control points. **Drag** points to move them.
        **Double-click** a point to delete it. The B-spline curve updates in real-time.

        *Powered by [extensible-splines](https://github.com/egoughnour/extensible-splines)*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Initialize with some default points
    default_points = [
        {"x": 100, "y": 300},
        {"x": 200, "y": 100},
        {"x": 400, "y": 100},
        {"x": 500, "y": 300},
        {"x": 400, "y": 500},
        {"x": 200, "y": 500},
    ]

    points_state = mo.state(default_points)
    return default_points, points_state


@app.cell(hide_code=True)
def _(mo, np, json, points_state):
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
        n = len(control_points)
        if n < 4:
            return []

        k = 3  # cubic
        knots = [0] * (k + 1) + list(range(1, n - k)) + [n - k] * (k + 1)
        knots = [float(x) for x in knots]

        t_vals = np.linspace(knots[k], knots[n] - 0.0001, num_samples)
        curve = []

        for t in t_vals:
            point = np.zeros(2)
            for i in range(n):
                basis = bspline_basis(i, k, t, knots)
                point += basis * np.array([control_points[i]["x"], control_points[i]["y"]])
            curve.append({"x": point[0], "y": point[1]})

        return curve

    points = points_state.value
    curve_points = evaluate_bspline(points)

    # Generate SVG path for the curve
    curve_path = ""
    if len(curve_points) > 1:
        curve_path = f"M {curve_points[0]['x']} {curve_points[0]['y']} "
        for pt in curve_points[1:]:
            curve_path += f"L {pt['x']} {pt['y']} "

    # Generate control polygon path
    polygon_path = ""
    if len(points) > 1:
        polygon_path = f"M {points[0]['x']} {points[0]['y']} "
        for pt in points[1:]:
            polygon_path += f"L {pt['x']} {pt['y']} "

    # The interactive canvas using HTML/JS
    canvas_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem;">
        <svg id="spline-canvas" width="700" height="600" style="background: #1e293b; border-radius: 12px; cursor: crosshair;">
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
            <g id="control-points">
                {"".join(f'<circle cx="{pt["x"]}" cy="{pt["y"]}" r="10" fill="#14b8a6" stroke="#fff" stroke-width="2" class="control-point" data-index="{i}" style="cursor: grab;"/>' for i, pt in enumerate(points))}
            </g>

            <!-- Instructions overlay -->
            <text x="350" y="580" text-anchor="middle" fill="#64748b" font-size="12" font-family="Inter, sans-serif">
                Click to add points | Drag to move | Double-click to delete
            </text>
        </svg>

        <div style="display: flex; gap: 1rem; align-items: center;">
            <span style="color: #94a3b8; font-size: 14px;">Control Points: <strong style="color: #14b8a6;">{len(points)}</strong></span>
            <span style="color: #94a3b8; font-size: 14px;">|</span>
            <span style="color: #94a3b8; font-size: 14px;">Curve: <strong style="color: #6366f1;">Cubic B-Spline</strong></span>
            <span style="color: #94a3b8; font-size: 14px;">|</span>
            <span style="color: #94a3b8; font-size: 14px; opacity: {1 if len(points) >= 4 else 0.5};">{"Curve visible" if len(points) >= 4 else "Need 4+ points for curve"}</span>
        </div>
    </div>

    <script>
    (function() {{
        const svg = document.getElementById('spline-canvas');
        const pointsGroup = document.getElementById('control-points');
        let points = {json.dumps(points)};
        let dragIndex = -1;
        let dragOffset = {{x: 0, y: 0}};
        let lastClickTime = 0;

        function getSVGPoint(e) {{
            const rect = svg.getBoundingClientRect();
            return {{
                x: e.clientX - rect.left,
                y: e.clientY - rect.top
            }};
        }}

        function updateMarimo() {{
            // Send updated points back to marimo via the anywidget protocol
            const event = new CustomEvent('marimo:state-update', {{
                detail: {{ points: points }},
                bubbles: true
            }});
            svg.dispatchEvent(event);
        }}

        svg.addEventListener('mousedown', function(e) {{
            const target = e.target;
            const now = Date.now();

            if (target.classList.contains('control-point')) {{
                const index = parseInt(target.dataset.index);

                // Check for double-click (delete)
                if (now - lastClickTime < 300 && points.length > 2) {{
                    points.splice(index, 1);
                    updateMarimo();
                    location.reload(); // Trigger re-render
                    return;
                }}
                lastClickTime = now;

                // Start dragging
                dragIndex = index;
                const pt = getSVGPoint(e);
                dragOffset = {{
                    x: points[index].x - pt.x,
                    y: points[index].y - pt.y
                }};
                target.style.cursor = 'grabbing';
                e.preventDefault();
            }}
        }});

        svg.addEventListener('mousemove', function(e) {{
            if (dragIndex >= 0) {{
                const pt = getSVGPoint(e);
                points[dragIndex].x = Math.max(10, Math.min(690, pt.x + dragOffset.x));
                points[dragIndex].y = Math.max(10, Math.min(590, pt.y + dragOffset.y));

                // Update circle position directly for smooth feedback
                const circle = pointsGroup.children[dragIndex];
                circle.setAttribute('cx', points[dragIndex].x);
                circle.setAttribute('cy', points[dragIndex].y);
            }}
        }});

        svg.addEventListener('mouseup', function(e) {{
            if (dragIndex >= 0) {{
                const circle = pointsGroup.children[dragIndex];
                circle.style.cursor = 'grab';
                dragIndex = -1;
                updateMarimo();
                location.reload(); // Trigger re-render with new curve
            }}
        }});

        svg.addEventListener('click', function(e) {{
            // Only add point if clicking on empty space
            if (!e.target.classList.contains('control-point') && dragIndex < 0) {{
                const pt = getSVGPoint(e);
                points.push({{x: pt.x, y: pt.y}});
                updateMarimo();
                location.reload(); // Trigger re-render
            }}
        }});
    }})();
    </script>
    """

    mo.Html(canvas_html)
    return


@app.cell(hide_code=True)
def _(mo, points_state):
    # Buttons for reset and clear
    def reset_points():
        points_state.set_value([
            {"x": 100, "y": 300},
            {"x": 200, "y": 100},
            {"x": 400, "y": 100},
            {"x": 500, "y": 300},
            {"x": 400, "y": 500},
            {"x": 200, "y": 500},
        ])

    def clear_points():
        points_state.set_value([])

    reset_btn = mo.ui.button(label="Reset to Default", on_click=lambda _: reset_points())
    clear_btn = mo.ui.button(label="Clear All", on_click=lambda _: clear_points())

    mo.hstack([reset_btn, clear_btn], justify="center", gap="1rem")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## About B-Splines

        B-splines (Basis splines) are piecewise polynomial curves with several important properties:

        - **Local control**: Moving one control point only affects the curve nearby
        - **Smoothness**: Cubic B-splines have C2 continuity (smooth curvature)
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
