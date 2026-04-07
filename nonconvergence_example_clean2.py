"""
nonconvergence_example_clean.py


Output
------
  nonconvergence_example_clean.png  – combined figure (boxplots + G function)

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import unittest


# ---------------------------------------------------------------------------
# Distribution definition  (Section 5.2)
# ---------------------------------------------------------------------------

SUPPORT = np.array([-1.0, 1.0, 2.0, 5.0])
PROBS   = np.array([ 0.05, 0.15, 0.30, 0.50])

BETA_STAR      = 2.0
F_MINUS        = 0.20   # liminf = F(beta*-)
F_STAR         = 0.50   # limsup = F(beta*)
MODE_THRESHOLD = (F_MINUS + F_STAR) / 2.0   # = 0.35


# ---------------------------------------------------------------------------
# G function
# ---------------------------------------------------------------------------

def G_discrete(y, support, probs):
    """
    G(y) = sum_i  p_i * beta_i * (beta_i - y) * 1{beta_i <= y}
    """
    mask = support <= y
    return float(np.sum(probs[mask] * support[mask] * (support[mask] - y)))


def find_beta_star(support, probs, y_max=20.0, n_grid=10_000):
    """Numerical beta* = sup{y >= 0 : G(y) = 0}."""
    ys    = np.linspace(0.0, y_max, n_grid)
    Gvals = np.array([G_discrete(y, support, probs) for y in ys])
    nonneg_idx = np.where(Gvals >= -1e-12)[0]
    return float(ys[nonneg_idx[-1]]) if len(nonneg_idx) > 0 else 0.0


def f_cdf(x, support, probs):
    """F(x) = P(beta <= x)."""
    return float(np.sum(probs[support <= x]))


def f_cdf_minus(x, support, probs):
    """F(x-) = P(beta < x)."""
    return float(np.sum(probs[support < x]))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_active_k(beta, var_f, var_eps):
    """
    Return k_p (number of active LOMV assets) for a given beta vector.
    Implements Theorem 1.
    """
    beta = beta.copy()
    if beta.sum() < 0:
        beta = -beta

    beta_s = np.sort(beta)
    S1 = np.empty(len(beta_s));  S1[0] = 0.0;  S1[1:] = np.cumsum(beta_s[:-1])
    S2 = np.empty(len(beta_s));  S2[0] = 0.0;  S2[1:] = np.cumsum(beta_s[:-1]**2)
    R  = 1.0 / var_f + (S2 - beta_s * S1) / var_eps
    k  = int(np.where(R > 0)[0][-1]) + 1
    return k


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_trials(p_values, var_f, var_eps, support, probs,
               num_trials=400, base_seed=0):
    """Run num_trials for each p; return dict p -> array of k_p/p."""
    results = {}
    for pi, p in enumerate(p_values):
        cell_seed = base_seed + pi * num_trials
        ratios = np.empty(num_trials)
        for t in range(num_trials):
            rng    = np.random.default_rng(cell_seed + t)
            beta   = rng.choice(support, size=p, p=probs)
            ratios[t] = compute_active_k(beta, var_f, var_eps) / p
        results[p] = ratios
        n_high = np.sum(ratios > MODE_THRESHOLD)
        print(f"  p={p:6d}: mean={ratios.mean():.4f}  std={ratios.std():.4f}  "
              f"high-mode={n_high}/{num_trials} ({100*n_high/num_trials:.0f}%)")
    return results


# ---------------------------------------------------------------------------
# Combined figure  (minimal in-figure text)
# ---------------------------------------------------------------------------

def make_figure(results, p_values, num_trials, f_minus, f_star):
    """
    Single figure with two rows:
      Row 1 – one boxplot panel per p value.
      Row 2 – G(y) curve spanning the full width.
    All explanatory text is omitted from the figure; it belongs in the LaTeX
    caption (written by write_latex_figure()).
    """
    rng_jitter = np.random.default_rng(42)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, len(p_values), figure=fig,
                            hspace=0.40, wspace=0.18,
                            height_ratios=[1.3, 1.0])

    # ------------------------------------------------------------------
    # Row 1: boxplots
    # ------------------------------------------------------------------
    for col, p in enumerate(p_values):
        ax     = fig.add_subplot(gs[0, col])
        ratios = results[p]
        low    = ratios[ratios <  MODE_THRESHOLD]
        high   = ratios[ratios >= MODE_THRESHOLD]

        ax.boxplot(ratios, positions=[1], vert=True, patch_artist=True,
                   widths=0.4,
                   boxprops=dict(facecolor="lightyellow", color="grey",
                                 linewidth=1.2),
                   medianprops=dict(linewidth=0),
                   whiskerprops=dict(color="grey"),
                   capprops=dict(color="grey"),
                   flierprops=dict(marker="", linestyle="none"))

        jw = 0.12
        def jitter(n): return rng_jitter.uniform(-jw, jw, size=n)

        ax.scatter(1 + jitter(len(low)),  low,  s=8, color="steelblue",
                   alpha=0.55, zorder=4)
        ax.scatter(1 + jitter(len(high)), high, s=8, color="tomato",
                   alpha=0.55, zorder=4)

        x0, x1 = 0.55, 1.45
        ax.axhspan(f_minus, f_star, xmin=0, xmax=1, color="gold", alpha=0.15)
        ax.hlines(f_minus, x0, x1, colors="darkorange", linewidths=2,
                  linestyles="--", zorder=6,
                  label=r"$\liminf$" if col == 0 else None)
        ax.hlines(f_star,  x0, x1, colors="darkred",   linewidths=2,
                  linestyles="-.", zorder=6,
                  label=r"$\limsup$" if col == 0 else None)

        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([])
        ax.set_title(f"$p = {p:,}$", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        # Trial counts per mode in matching colours, inside the top of the panel
        ax.text(0.36, 0.97, f"\u25cf {len(low)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="steelblue")
        ax.text(0.64, 0.97, f"\u25cf {len(high)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, color="tomato")

        if col == 0:
            ax.set_ylabel(r"$k_p\,/\,p$", fontsize=11)
            ax.legend(loc="upper right", fontsize=8, handlelength=1.5)

    # ------------------------------------------------------------------
    # Row 2: G(y) curve
    # ------------------------------------------------------------------
    ax_g = fig.add_subplot(gs[1, :])

    ys    = np.linspace(0, 6.5, 2000)
    Gvals = np.array([G_discrete(y, SUPPORT, PROBS) for y in ys])

    ax_g.plot(ys, Gvals, color="steelblue", linewidth=2, label=r"$G(y)$")
    ax_g.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_g.axvline(BETA_STAR, color="darkorange", linewidth=1.5, linestyle="--",
                 label=fr"$\beta^*={int(BETA_STAR)}$")
    ax_g.plot(BETA_STAR, 0, "o", color="darkorange", markersize=8, zorder=5)

    mask_pos = Gvals > 0
    ax_g.fill_between(ys, Gvals, 0, where=mask_pos,
                      color="lightgreen", alpha=0.30, label=r"$G>0$")
    ax_g.fill_between(ys, Gvals, 0, where=~mask_pos,
                      color="lightsalmon", alpha=0.30, label=r"$G<0$")

    ax_g.set_xlabel(r"$y$", fontsize=11)
    ax_g.set_ylabel(r"$G(y)$", fontsize=11)
    ax_g.legend(fontsize=9, loc="upper right")
    ax_g.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax_g.set_xlim(0, 6.5)

    png_path = "nonconvergence_example_clean.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {png_path}")
    plt.show()
    return png_path




# ---------------------------------------------------------------------------
# Unit tests  (identical to nonconvergence_example.py)
# ---------------------------------------------------------------------------

class TestNonConvergence(unittest.TestCase):

    def test_G_at_beta_star_is_zero(self):
        g = G_discrete(BETA_STAR, SUPPORT, PROBS)
        self.assertAlmostEqual(g, 0.0, places=10,
                               msg=f"G(beta*)={g:.2e}, expected 0")

    def test_G_positive_before_beta_star(self):
        for y in [0.1, 0.5, 1.0, 1.5, 1.99]:
            g = G_discrete(y, SUPPORT, PROBS)
            self.assertGreater(g, 0.0, msg=f"G({y})={g:.4f} should be > 0")

    def test_G_negative_after_beta_star(self):
        for y in [2.01, 3.0, 4.0, 5.0, 6.0]:
            g = G_discrete(y, SUPPORT, PROBS)
            self.assertLess(g, 0.0, msg=f"G({y})={g:.4f} should be < 0")

    def test_piecewise_formulas_match_paper(self):
        y = 0.7
        self.assertAlmostEqual(G_discrete(y, SUPPORT, PROBS),
                               0.05 * (1 + y), places=10)
        y = 1.4
        self.assertAlmostEqual(G_discrete(y, SUPPORT, PROBS),
                               0.20 - 0.10 * y, places=10)
        y = 3.5
        self.assertAlmostEqual(G_discrete(y, SUPPORT, PROBS),
                               1.40 - 0.70 * y, places=10)

    def test_theoretical_limits_match_paper(self):
        self.assertAlmostEqual(f_cdf_minus(BETA_STAR, SUPPORT, PROBS), F_MINUS, places=10)
        self.assertAlmostEqual(f_cdf(BETA_STAR, SUPPORT, PROBS),       F_STAR,  places=10)

    def test_find_beta_star_numeric(self):
        self.assertAlmostEqual(find_beta_star(SUPPORT, PROBS), BETA_STAR, delta=0.01)

    def test_k_in_valid_range(self):
        rng = np.random.default_rng(0)
        for p in [100, 500]:
            for _ in range(10):
                beta = rng.choice(SUPPORT, size=p, p=PROBS)
                k = compute_active_k(beta, var_f=1.0, var_eps=0.1)
                self.assertGreaterEqual(k, 1)
                self.assertLessEqual(k, p)

    def test_active_ratio_in_correct_range(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            beta  = rng.choice(SUPPORT, size=2000, p=PROBS)
            ratio = compute_active_k(beta, var_f=1.0, var_eps=0.1) / 2000
            self.assertGreater(ratio, F_MINUS - 0.05)
            self.assertLess(ratio,    F_STAR  + 0.05)

    def test_bimodality_at_large_p(self):
        rng = np.random.default_rng(2)
        in_gap = 0
        n = 200
        for _ in range(n):
            beta  = rng.choice(SUPPORT, size=5000, p=PROBS)
            ratio = compute_active_k(beta, var_f=1.0, var_eps=0.1) / 5000
            if 0.25 < ratio < 0.45:
                in_gap += 1
        self.assertLess(in_gap / n, 0.15,
                        msg=f"{in_gap}/{n} ratios in gap, expected few")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    var_f      = 1.0
    var_eps    = 0.1
    num_trials = 400
    p_values   = [500, 3000, 10_000]

    print("=== Non-Convergence Example (Section 5.2) ===\n")
    print("Beta distribution:")
    for b, p in zip(SUPPORT, PROBS):
        print(f"  P(beta = {b:+.0f}) = {p}")
    print(f"\nE[beta] = {np.dot(SUPPORT, PROBS):.2f}")
    print(f"P(beta < 0) = {PROBS[SUPPORT < 0].sum():.2f}")

    print(f"\nTheoretical quantities:")
    print(f"  beta*        = {BETA_STAR}")
    print(f"  G(beta*)     = {G_discrete(BETA_STAR, SUPPORT, PROBS):.6f}  (should be 0)")
    print(f"  liminf k_p/p = F(beta*-) = {F_MINUS:.4f}")
    print(f"  limsup k_p/p = F(beta*)  = {F_STAR:.4f}")
    print(f"  atom mass at beta* = {PROBS[SUPPORT == BETA_STAR][0]:.2f}")

    print(f"\nRunning {num_trials} trials for each p value...")
    results = run_trials(p_values, var_f, var_eps, SUPPORT, PROBS,
                         num_trials=num_trials, base_seed=0)

    print(f"\nSummary:")
    print(f"  {'p':>7}  {'mean':>7}  {'std':>7}  {'% high':>8}  "
          f"{'liminf':>8}  {'limsup':>8}")
    for p in p_values:
        r    = results[p]
        pct  = 100 * np.mean(r >= MODE_THRESHOLD)
        print(f"  {p:>7,}  {r.mean():>7.4f}  {r.std():>7.4f}  "
              f"{pct:>7.1f}%  {F_MINUS:>8.4f}  {F_STAR:>8.4f}")

    print()
    make_figure(results, p_values, num_trials, F_MINUS, F_STAR)


    print("\n--- Running unit tests ---")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestNonConvergence)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    main()
