# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
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
        scheme `w_t = 1 + ρt` that guarantees monotonic forward progress in the reconstruction.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    def orthogonal_matching_pursuit(X, y, n_nonzero_coefs):
        """
        Pure NumPy implementation of Orthogonal Matching Pursuit.

        Parameters:
        -----------
        X : ndarray of shape (n_features, n_atoms)
            Dictionary matrix (columns are atoms)
        y : ndarray of shape (n_features,)
            Target signal
        n_nonzero_coefs : int
            Maximum number of nonzero coefficients

        Returns:
        --------
        coef : ndarray of shape (n_atoms,)
            Sparse coefficient vector
        """
        n_features, n_atoms = X.shape
        coef = np.zeros(n_atoms)
        residual = y.copy()
        support = []

        for _ in range(n_nonzero_coefs):
            # Find atom most correlated with residual
            correlations = np.abs(X.T @ residual)
            correlations[support] = -np.inf  # Exclude already selected atoms
            best_atom = np.argmax(correlations)

            if correlations[best_atom] < 1e-10:
                break  # Residual is essentially zero

            support.append(best_atom)

            # Solve least squares on selected atoms
            X_support = X[:, support]
            coef_support = np.linalg.lstsq(X_support, y, rcond=None)[0]

            # Update residual
            residual = y - X_support @ coef_support

        # Fill in coefficients for selected atoms
        for i, idx in enumerate(support):
            coef[idx] = coef_support[i]

        return coef

    return np, orthogonal_matching_pursuit, plt


@app.cell(hide_code=True)
def _(np):
    # Core algorithm functions (from mosaic-basis)

    def choose_gamma(y, X, delta_w, tau=0.05, safety=1.25, mode="global"):
        """Compute gamma that guarantees forward step in OMP."""
        if mode == "empirical":
            Xn = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
            M0 = float(np.max(np.abs(Xn.T @ y)))
        else:
            M0 = float(np.linalg.norm(y))
        gamma = np.sqrt((2.0 * M0 * safety) / (tau * delta_w))
        return float(gamma)

    def build_temporal_tags(N, rho=0.02):
        """Build two-hot temporal tags with forward bias."""
        w = 1.0 + rho * np.arange(N + 1, dtype=float)
        V = np.zeros((N, N), dtype=float)
        for t in range(N):
            V[t, t] += np.sqrt(w[t])
            if t + 1 < N:
                V[t + 1, t] += np.sqrt(w[t + 1])
        return V, w

    def augment_dictionary_framewise(ycbcr_series, target_vec, rho=0.02, tau=0.05,
                                      safety=1.25, gamma=None, mode="empirical"):
        """Build augmented dictionary for OMP with temporal tags."""
        X = ycbcr_series.astype(float).T  # (3, N)
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

        # Renormalize columns
        X_aug = X_aug / (np.linalg.norm(X_aug, axis=0, keepdims=True) + 1e-12)

        return X_aug, y_aug, float(gamma), dict(N=N, rho=rho, V=V, w=w, norms=norms)

    return augment_dictionary_framewise, build_temporal_tags, choose_gamma


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Interactive Parameters""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Parameter controls
    n_frames_slider = mo.ui.slider(10, 50, value=20, step=5, label="Number of frames (N)")
    target_pos_slider = mo.ui.slider(0.0, 1.0, value=0.26, step=0.02, label="Target position (0-1)")
    rho_slider = mo.ui.slider(0.005, 0.1, value=0.02, step=0.005, label="Temporal weight increment (ρ)")
    tau_slider = mo.ui.slider(0.01, 0.2, value=0.05, step=0.01, label="Min coefficient threshold (τ)")
    n_coefs_slider = mo.ui.slider(1, 6, value=3, step=1, label="Max nonzero coefficients")
    seed_slider = mo.ui.slider(0, 100, value=7, step=1, label="Random seed")

    mo.vstack([
        mo.hstack([n_frames_slider, target_pos_slider], justify="start"),
        mo.hstack([rho_slider, tau_slider], justify="start"),
        mo.hstack([n_coefs_slider, seed_slider], justify="start"),
    ])
    return (
        n_coefs_slider,
        n_frames_slider,
        rho_slider,
        seed_slider,
        target_pos_slider,
        tau_slider,
    )


@app.cell(hide_code=True)
def _(
    augment_dictionary_framewise,
    n_coefs_slider,
    n_frames_slider,
    np,
    orthogonal_matching_pursuit,
    rho_slider,
    seed_slider,
    target_pos_slider,
    tau_slider,
):
    # Generate data and run OMP
    rng = np.random.default_rng(seed_slider.value)
    N = n_frames_slider.value
    series = rng.normal(size=(N, 3))

    # Smooth the series a bit to simulate real video frames
    for i in range(1, N):
        series[i] = 0.7 * series[i] + 0.3 * series[i-1]

    # Target position as fraction through the sequence
    pos = target_pos_slider.value
    idx = int(pos * (N - 1))
    idx = min(idx, N - 2)
    frac = pos * (N - 1) - idx
    target = series[idx] + frac * (series[idx + 1] - series[idx])

    # Run augmented OMP
    X_aug, y_aug, gamma, meta = augment_dictionary_framewise(
        series, target, rho=rho_slider.value, tau=tau_slider.value, safety=1.25, mode="empirical"
    )

    # Use our pure NumPy OMP implementation
    coef = orthogonal_matching_pursuit(X_aug, y_aug, n_nonzero_coefs=n_coefs_slider.value)
    support = np.flatnonzero(coef)

    # Reconstruct the target from chosen basis frames
    X_orig = series.T / (np.linalg.norm(series.T, axis=0, keepdims=True) + 1e-12)
    reconstruction = X_orig @ coef
    reconstruction = reconstruction * np.linalg.norm(target)  # Scale back

    reconstruction_error = np.linalg.norm(target - reconstruction)
    return (
        N,
        X_aug,
        X_orig,
        coef,
        frac,
        gamma,
        idx,
        meta,
        pos,
        reconstruction,
        reconstruction_error,
        rng,
        series,
        support,
        target,
        y_aug,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Results""")
    return


@app.cell(hide_code=True)
def _(coef, gamma, idx, mo, reconstruction_error, support):
    mo.md(
        f"""
        ### OMP Selection Results

        - **Target position:** Between frames {idx} and {idx + 1}
        - **Selected support (frame indices):** {list(support)}
        - **Coefficients:** {[f"{c:.3f}" for c in coef[support]]}
        - **Gamma (temporal scale):** {gamma:.2f}
        - **Reconstruction error:** {reconstruction_error:.4f}
        """
    )
    return


@app.cell(hide_code=True)
def _(N, coef, idx, np, plt, series, support, target):
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Frame series in YCbCr space (Y component)
    ax1 = axes[0, 0]
    frames = np.arange(N)
    ax1.plot(frames, series[:, 0], 'b-', label='Y channel', linewidth=2)
    ax1.plot(frames, series[:, 1], 'g-', label='Cb channel', linewidth=2, alpha=0.7)
    ax1.plot(frames, series[:, 2], 'r-', label='Cr channel', linewidth=2, alpha=0.7)
    ax1.axvline(x=idx + 0.5, color='orange', linestyle='--', linewidth=2, label='Target position')
    ax1.scatter([idx + 0.5], [target[0]], color='orange', s=100, zorder=5, marker='*')
    ax1.scatter(support, series[support, 0], color='purple', s=150, zorder=5, marker='o',
                edgecolors='black', linewidths=2, label='Selected frames')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('YCbCr Value')
    ax1.set_title('Frame Sequence with OMP Selection')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coefficient magnitudes
    ax2 = axes[0, 1]
    colors = ['purple' if i in support else 'lightgray' for i in range(N)]
    ax2.bar(frames, np.abs(coef), color=colors, edgecolor='black')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('|Coefficient|')
    ax2.set_title('OMP Coefficient Magnitudes')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: 3D visualization of frames and target
    ax3 = axes[1, 0]
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(series[:, 0], series[:, 1], series[:, 2], c=frames, cmap='viridis',
                s=50, alpha=0.6, label='Frames')
    ax3.scatter(series[support, 0], series[support, 1], series[support, 2],
                c='purple', s=200, marker='o', edgecolors='black', linewidths=2, label='Selected')
    ax3.scatter([target[0]], [target[1]], [target[2]], c='orange', s=300, marker='*',
                edgecolors='black', linewidths=2, label='Target')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Cb')
    ax3.set_zlabel('Cr')
    ax3.set_title('YCbCr Space')

    # Plot 4: Temporal weights
    ax4 = axes[1, 1]
    w = 1.0 + 0.02 * np.arange(N)
    ax4.plot(frames, w[:-1], 'b-', linewidth=2, label='Temporal weight w_t')
    ax4.fill_between(frames, 1, w[:-1], alpha=0.3)
    ax4.scatter(support, w[support], color='purple', s=150, zorder=5, marker='o',
                edgecolors='black', linewidths=2, label='Selected frames')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Weight w_t = 1 + ρt')
    ax4.set_title('Forward-Biased Temporal Weights')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig
    return ax1, ax2, ax3, ax4, axes, colors, fig, frames, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## How It Works

        1. **Dictionary Augmentation**: Each frame vector is augmented with temporal tags that encode
           its position in the sequence. These tags use a two-hot encoding with forward-biased weights.

        2. **Forward Bias**: The weight schedule `w_t = 1 + ρt` creates a monotonically increasing
           preference for later frames, which helps OMP make consistent forward progress.

        3. **Gamma Scaling**: The temporal tags are scaled by γ to balance the data fidelity term
           against the temporal regularization.

        4. **OMP Selection**: Standard OMP then selects the k most important atoms from this
           augmented dictionary, naturally preferring frames close to the target position.

        ---

        *Implementation from [mosaic-basis](https://github.com/egoughnour/mosaic-basis) -
        Forward-biased frame selection for OMP to reconstruct frame sequences.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
