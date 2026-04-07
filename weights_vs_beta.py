"""
weights_vs_beta.py

April 5, 2026

Plots GMV and LOMV portfolio weights against factor loadings (betas) for a
single-factor covariance matrix

    Sigma = sigma^2 * beta beta^T + delta^2 * I_p

Two portfolios are computed:
  GMV  (long-short): min w' Sigma w  s.t. w'1 = 1
  LOMV (long-only) : min w' Sigma w  s.t. w'1 = 1,  w_i >= 0 for all i

GMV weights are computed analytically via the Sherman-Morrison formula.
The result is exactly affine in beta_i:  w_i^GMV = A + B * beta_i.

LOMV weights are computed via Theorem 1 (active set K from the R_i sequence)
and Proposition 1 (GMV on the restricted k-asset submatrix, again by
Sherman-Morrison).  The result is zero outside K and affine in beta_i within K.

Output: weights_vs_beta.png
"""

import numpy as np
import matplotlib.pyplot as plt
import unittest


# ---------------------------------------------------------------------------
# Parameters  (adjust here)
# ---------------------------------------------------------------------------

SIGMA     = 1.0   # factor scale:  Sigma = sigma^2 * beta beta^T + delta^2 * I
DELTA     = 0.5   # idiosyncratic scale
P         = 5000  # number of assets
BETA_MEAN = 1.0   # mean of beta_i ~ N(beta_mean, beta_std^2)
BETA_STD  = 0.4   # std dev of beta distribution
SEED      = 42    # RNG seed for reproducibility

# Derived variance parameters used throughout
VAR_F   = SIGMA ** 2   # = 1.00  (factor return variance)
VAR_EPS = DELTA ** 2   # = 0.25  (idiosyncratic variance)


# ---------------------------------------------------------------------------
# GMV portfolio  (Sherman-Morrison)
# ---------------------------------------------------------------------------

def gmv_weights(beta, var_f, var_eps):
    """
    Global minimum variance (long-short) portfolio weights.

    Sigma = var_f * beta beta^T + var_eps * I.  By Sherman-Morrison:

        Sigma^{-1} 1 = (1/var_eps) * [1 - c * (beta^T 1) * beta],
        c = var_f / (var_eps + var_f * ||beta||^2).

    Weights are exactly affine in beta:  w_i = A + B * beta_i.

    Parameters
    ----------
    beta    : (p,) array
    var_f   : float  (sigma^2)
    var_eps : float  (delta^2)

    Returns
    -------
    w : (p,) array summing to 1 (may contain negative entries)
    """
    q    = np.dot(beta, beta)          # ||beta||^2
    s    = beta.sum()                  # 1^T beta
    c    = var_f / (var_eps + var_f * q)
    # Sigma^{-1} 1  (up to the 1/var_eps scalar, which cancels in the ratio)
    Sinv1 = np.ones(len(beta)) - c * s * beta
    return Sinv1 / Sinv1.sum()


# ---------------------------------------------------------------------------
# LOMV portfolio  (Theorem 1 + Proposition 1)
# ---------------------------------------------------------------------------

def lomv_weights(beta, var_f, var_eps):
    """
    Long-only minimum variance portfolio weights.

    Step 1 — Theorem 1: sort betas ascending (with sign convention) and find
    the active set K = {1, ..., k} via the sequence
        R_i = 1/var_f + (1/var_eps) * (S2_i - beta_i * S1_i),
        S1_i = sum_{j<i} beta_j,  S2_i = sum_{j<i} beta_j^2.
    k = max{i : R_i > 0}.

    Step 2 — Proposition 1: the LOMV weights on K are the GMV weights for
    the k-asset submatrix Sigma^K, computed by the same Sherman-Morrison
    formula.  All remaining weights are zero.

    Parameters
    ----------
    beta    : (p,) array
    var_f   : float
    var_eps : float

    Returns
    -------
    w : (p,) array, sums to 1, all entries >= 0 (original asset ordering)
    k : int, number of active assets
    """
    p         = len(beta)
    beta_work = beta.copy()

    # Sign convention: ensure sum(beta) >= 0 (does not change Sigma)
    if beta_work.sum() < 0:
        beta_work = -beta_work

    # Sort ascending: Theorem 1 requires beta_1 <= ... <= beta_p
    sort_idx = np.argsort(beta_work)
    beta_s   = beta_work[sort_idx]

    # Vectorised R_i sequence
    S1 = np.empty(p);  S1[0] = 0.0;  S1[1:] = np.cumsum(beta_s[:-1])
    S2 = np.empty(p);  S2[0] = 0.0;  S2[1:] = np.cumsum(beta_s[:-1] ** 2)
    R  = 1.0 / var_f + (S2 - beta_s * S1) / var_eps

    # k = last 1-based index with R_i > 0  (R[0] = 1/var_f > 0 always)
    k = int(np.where(R > 0)[0][-1]) + 1

    # GMV weights on the k-asset active submatrix (Sherman-Morrison)
    w_K = gmv_weights(beta_s[:k], var_f, var_eps)

    # Reconstruct p-vector in the original asset ordering
    w_sorted       = np.zeros(p)
    w_sorted[:k]   = w_K
    w_full         = np.zeros(p)
    w_full[sort_idx] = w_sorted

    return w_full, k


# ---------------------------------------------------------------------------
# Portfolio variance  (O(p), exploits single-factor structure)
# ---------------------------------------------------------------------------

def portfolio_variance(w, beta, var_f, var_eps):
    """
    w^T Sigma w = var_f * (beta^T w)^2 + var_eps * ||w||^2.
    Avoids forming the p x p matrix Sigma.
    """
    return var_f * np.dot(beta, w) ** 2 + var_eps * np.dot(w, w)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_weights_vs_beta(beta, w_gmv, w_lomv, k,
                         sigma, delta, beta_mean, beta_std):
    """
    Scatter plot of w_i vs beta_i for both portfolios.
    Blue dots: GMV.  Black dots: LOMV.
    """
    p       = len(beta)
    n_neg   = int((beta < 0).sum())
    pct_neg = 100.0 * n_neg / p

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(beta, w_gmv,  s=4, color="steelblue", alpha=0.55, zorder=3,
               label=f"GMV (long-short, all {p:,} assets)")
    ax.scatter(beta, w_lomv, s=4, color="black",     alpha=0.55, zorder=4,
               label=f"LOMV (long-only, {k} active assets)")

    # Invisible entries that add information to the legend box
    ax.plot([], [], " ", label=f"Negative betas: {n_neg} ({pct_neg:.2f}%)")

    n_gmv_pos = int((w_gmv > 0).sum())
    ax.plot([], [], " ", label=f"GMV positive weights: {n_gmv_pos} ({100.0*n_gmv_pos/p:.1f}%)")

    ax.axhline(0, color="dimgrey", linewidth=0.9, linestyle="--", zorder=2)

    ax.set_xlabel(r"Factor loading $\beta_i$", fontsize=12)
    ax.set_ylabel(r"Portfolio weight $w_i$", fontsize=12)
    ax.set_title(
        r"GMV vs LOMV portfolio weights against $\beta_i$"   "\n"
        fr"$\beta_i \sim \mathcal{{N}}({beta_mean},\,{beta_std}^2)$,  "
        fr"$\Sigma = {sigma}^2\,\beta\beta^\top + {delta}^2\,I$,  "
        fr"$p = {p:,}$",
        fontsize=11
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig("weights_vs_beta.png", dpi=150, bbox_inches="tight")
    print("Plot saved to weights_vs_beta.png")
    plt.show()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPortfolios(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.beta    = rng.normal(1.0, 0.4, size=300)
        self.var_f   = VAR_F
        self.var_eps = VAR_EPS

    def test_gmv_sums_to_one(self):
        w = gmv_weights(self.beta, self.var_f, self.var_eps)
        self.assertAlmostEqual(w.sum(), 1.0, places=10)

    def test_gmv_has_negative_weights(self):
        """With mean-positive betas, GMV goes short on high-beta assets."""
        w = gmv_weights(self.beta, self.var_f, self.var_eps)
        self.assertTrue(np.any(w < 0))

    def test_gmv_is_affine_in_beta(self):
        """Sherman-Morrison gives w_i = A + B*beta_i exactly."""
        w = gmv_weights(self.beta, self.var_f, self.var_eps)
        X = np.column_stack([np.ones_like(self.beta), self.beta])
        coeffs, *_ = np.linalg.lstsq(X, w, rcond=None)
        residuals = w - X @ coeffs
        self.assertLess(np.max(np.abs(residuals)), 1e-12)

    def test_lomv_sums_to_one(self):
        w, _ = lomv_weights(self.beta, self.var_f, self.var_eps)
        self.assertAlmostEqual(w.sum(), 1.0, places=10)

    def test_lomv_non_negative(self):
        w, _ = lomv_weights(self.beta, self.var_f, self.var_eps)
        self.assertTrue(np.all(w >= -1e-12))

    def test_lomv_active_count_matches_positive_weights(self):
        """k equals the number of strictly positive weights."""
        w, k = lomv_weights(self.beta, self.var_f, self.var_eps)
        self.assertEqual(k, int(np.sum(w > 1e-12)))

    def test_lomv_variance_geq_gmv_variance(self):
        """Long-only constraint => LOMV variance >= GMV variance."""
        w_g = gmv_weights(self.beta, self.var_f, self.var_eps)
        w_l, _ = lomv_weights(self.beta, self.var_f, self.var_eps)
        var_g = portfolio_variance(w_g, self.beta, self.var_f, self.var_eps)
        var_l = portfolio_variance(w_l, self.beta, self.var_f, self.var_eps)
        self.assertGreaterEqual(var_l, var_g - 1e-12)

    def test_sign_invariance(self):
        """Flipping beta leaves Sigma unchanged, so LOMV weights are the same."""
        w1, k1 = lomv_weights( self.beta, self.var_f, self.var_eps)
        w2, k2 = lomv_weights(-self.beta, self.var_f, self.var_eps)
        self.assertEqual(k1, k2)
        np.testing.assert_allclose(w1, w2, atol=1e-10)

    def test_portfolio_variance_formula(self):
        """Variance via factor formula matches direct quadratic form for small p."""
        p    = 10
        rng  = np.random.default_rng(99)
        beta = rng.normal(1.0, 0.4, size=p)
        w    = gmv_weights(beta, self.var_f, self.var_eps)
        Sigma = self.var_f * np.outer(beta, beta) + self.var_eps * np.eye(p)
        var_direct = float(w @ Sigma @ w)
        var_fast   = portfolio_variance(w, beta, self.var_f, self.var_eps)
        self.assertAlmostEqual(var_direct, var_fast, places=12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== GMV vs LOMV Portfolio Weights ===")
    print(f"p = {P},  beta ~ N({BETA_MEAN}, {BETA_STD}^2)")
    print(f"sigma = {SIGMA},  delta = {DELTA}  "
          f"=>  var_f = {VAR_F},  var_eps = {VAR_EPS}\n")

    rng  = np.random.default_rng(SEED)
    beta = rng.normal(BETA_MEAN, BETA_STD, size=P)
    print(f"Beta summary:  mean={beta.mean():.4f},  std={beta.std():.4f},  "
          f"min={beta.min():.4f},  max={beta.max():.4f}")
    print(f"Fraction negative: {(beta < 0).mean():.4f}\n")

    print("Computing GMV weights ...")
    w_gmv = gmv_weights(beta, VAR_F, VAR_EPS)
    n_neg = (w_gmv < 0).sum()
    print(f"  sum={w_gmv.sum():.8f},  min={w_gmv.min():.6f},  "
          f"max={w_gmv.max():.6f},  short positions={n_neg} ({100*n_neg/P:.1f}%)")

    print("Computing LOMV weights ...")
    w_lomv, k = lomv_weights(beta, VAR_F, VAR_EPS)
    print(f"  sum={w_lomv.sum():.8f},  active assets k={k} ({100*k/P:.2f}%),  "
          f"max weight={w_lomv.max():.6f}")

    var_gmv  = portfolio_variance(w_gmv,  beta, VAR_F, VAR_EPS)
    var_lomv = portfolio_variance(w_lomv, beta, VAR_F, VAR_EPS)
    print(f"\nPortfolio variance:")
    print(f"  GMV  = {var_gmv:.6e}")
    print(f"  LOMV = {var_lomv:.6e}  (ratio LOMV/GMV = {var_lomv/var_gmv:.4f})")

    plot_weights_vs_beta(beta, w_gmv, w_lomv, k,
                         SIGMA, DELTA, BETA_MEAN, BETA_STD)

    print("\n--- Running unit tests ---")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestPortfolios)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    main()
