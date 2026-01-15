# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Temporal OMP for Frame Reconstruction

        This interactive demo shows how **Orthogonal Matching Pursuit (OMP)** with temporal augmentation
        can reconstruct a target video frame as a sparse combination of basis frames.

        ## The Problem
        Given a sequence of N video frames (each represented as a YCbCr triplet), we want to represent
        a target frame as a sparse linear combination of the basis frames, with a preference for
        temporally adjacent frames.

        ## The Algorithm
        The key insight is to **augment** the dictionary with temporal tags that encourage OMP to
        select frames that are close in time. The temporal tags use a forward-biased weighting
        scheme w_t = 1 + rho*t that guarantees monotonic forward progress in the reconstruction.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.preprocessing import normalize
    import matplotlib.pyplot as plt
    return OrthogonalMatchingPursuit, normalize, np, plt


@app.cell(hide_code=True)
def _(np):
    def choose_gamma(y, X, delta_w, tau=0.05, safety=1.25, mode="global"):
        if mode == "empirical":
            Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
            M0 = float(np.max(np.abs(Xn.T @ y)))
        else:
            M0 = float(np.linalg.norm(y))
        gamma = np.sqrt((2.0 * M0 * safety) / (tau * delta_w))
        return float(gamma)

    def build_temporal_tags(N, rho=0.02):
        w = 1.0 + rho * np.arange(N + 1, dtype=float)
        V = np.zeros((N, N), dtype=float)
        for t in range(N):
            V[t, t] += np.sqrt(w[t])
            if t + 1 < N:
                V[t + 1, t] += np.sqrt(w[t + 1])
        return V, w

    def augment_dictionary_framewise(ycbcr_series, target_vec, rho=0.02, tau=0.05, safety=1.25, gamma=None, mode="empirical"):
        X = ycbcr_series.astype(float).T
        norms = np.linalg.norm(X, axis=0) + 1e-12
        X = X / norms
        N = X.shape[1]
        y = np.asarray(target_vec, dtype=float).reshape(3)
        y_norm = y / (np.linalg.norm(y) + 1e-12)
        V, w = build_temporal_tags(N, rho=rho)
        delta_w = float(np.min(np.diff(w)))
        if gamma is None:
            gamma = choose_gamma(y_norm, X, delta_w, tau=tau, safety=safety, mode=mode)
        X_aug = np.vstack([X, gamma * V])
        y_aug = np.concatenate([y_norm, np.zeros(N)])
        X_aug = X_aug / (np.linalg.norm(X_aug, axis=0, keepdims=True) + 1e-12)
        return X_aug, y_aug, float(gamma), dict(N=N, rho=rho, V=V, w=w, norms=norms)
    return augment_dictionary_framewise, build_temporal_tags, choose_gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Interactive Parameters")
    return


@app.cell(hide_code=True)
def _(mo):
    n_frames_slider = mo.ui.slider(10, 50, value=20, step=5, label="Number of frames (N)")
    target_pos_slider = mo.ui.slider(0.0, 1.0, value=0.26, step=0.02, label="Target position (0-1)")
    rho_slider = mo.ui.slider(0.005, 0.1, value=0.02, step=0.005, label="Temporal weight increment")
    n_coefs_slider = mo.ui.slider(1, 6, value=3, step=1, label="Max nonzero coefficients")
    mo.vstack([mo.hstack([n_frames_slider, target_pos_slider]), mo.hstack([rho_slider, n_coefs_slider])])
    return n_coefs_slider, n_frames_slider, rho_slider, target_pos_slider


@app.cell(hide_code=True)
def _(OrthogonalMatchingPursuit, augment_dictionary_framewise, n_coefs_slider, n_frames_slider, np, rho_slider, target_pos_slider):
    rng = np.random.default_rng(7)
    N = n_frames_slider.value
    series = rng.normal(size=(N, 3))
    for i in range(1, N):
        series[i] = 0.7 * series[i] + 0.3 * series[i-1]
    pos = target_pos_slider.value
    idx = min(int(pos * (N - 1)), N - 2)
    frac = pos * (N - 1) - idx
    target = series[idx] + frac * (series[idx + 1] - series[idx])
    X_aug, y_aug, gamma, meta = augment_dictionary_framewise(series, target, rho=rho_slider.value, tau=0.05)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_coefs_slider.value)
    omp.fit(X_aug, y_aug)
    coef = omp.coef_
    support = np.flatnonzero(coef)
    return N, coef, gamma, idx, series, support, target


@app.cell(hide_code=True)
def _(coef, gamma, idx, mo, support):
    mo.md(f"### Results
- Target: frames {idx}-{idx+1}
- Selected: {list(support)}
- Gamma: {gamma:.2f}")
    return


@app.cell(hide_code=True)
def _(N, coef, idx, np, plt, series, support, target):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    frames = np.arange(N)
    axes[0,0].plot(frames, series[:, 0], 'b-', label='Y')
    axes[0,0].scatter(support, series[support, 0], c='purple', s=150, zorder=5)
    axes[0,0].axvline(x=idx+0.5, color='orange', linestyle='--')
    axes[0,0].set_title('Frame Sequence')
    axes[0,0].legend()
    colors = ['purple' if i in support else 'lightgray' for i in range(N)]
    axes[0,1].bar(frames, np.abs(coef), color=colors)
    axes[0,1].set_title('Coefficients')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(series[:, 0], series[:, 1], series[:, 2], c=frames, cmap='viridis')
    ax3.scatter([target[0]], [target[1]], [target[2]], c='orange', s=200, marker='*')
    ax3.set_title('YCbCr Space')
    w = 1.0 + 0.02 * np.arange(N)
    axes[1,1].plot(frames, w[:-1], 'b-')
    axes[1,1].scatter(support, w[support], c='purple', s=150)
    axes[1,1].set_title('Temporal Weights')
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("*From [mosaic-basis](https://github.com/egoughnour/mosaic-basis)*")
    return


if __name__ == "__main__":
    app.run()
