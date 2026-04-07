"""
active_ratio_2.py

Computes the LOMV active ratio k_p / p as a function of p and the standard
deviation of the beta distribution, using the true population covariance

    Sigma = var_f * beta beta^T + var_eps * I_p

via Theorem 1 of the paper (the R_i sequence determines the active set K).
Weights are not needed here, so Proposition 1 is not called.

Each panel shows the theoretical limiting value F(beta*) as p -> infinity,
where beta* is the root of G(beta*) = 0:

    G(y) = (mu*(mu - y) + sigma^2) * Phi(alpha) - sigma*mu * phi(alpha),
    alpha = (y - mu) / sigma,   beta ~ N(mu, sigma^2).

Outputs (produced for each of var_eps in {0.5, 0.1})
------------------------------------------------------
  active_ratio_delta0.5.png / active_ratio_delta0.1.png
      Boxplot figure: no figure-level title or sub-panel captions.
      Axis labels and per-panel headings (beta ~ N(...)) are retained.
      Legend inside each panel shows mean marker, F(beta*) line,
      beta* value, and percentage of negative betas.

  active_ratio_delta0.5.txt / active_ratio_delta0.1.txt
      Plain-text file containing the caption / description information
      that was removed from the figure.

  active_ratio_table.tex
      LaTeX table summarising mean +/- std of k_p/p for all 18 cells
      (3 sigma values x 3 p values x 2 var_eps values).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import unittest
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Core computation: active set size only (Theorem 1, no weights)
# ---------------------------------------------------------------------------

def compute_active_k(beta, var_f, var_eps):
    """
    Return k_p (number of active LOMV assets) for a given beta vector.

    Theorem 1: sort betas ascending, apply sign convention, then find
    k = max{i : R_i > 0} where
        R_i = 1/var_f + (1/var_eps) * (S2_i - beta_i * S1_i),
        S1_i = sum_{j<i} beta_j,  S2_i = sum_{j<i} beta_j^2.
    """
    beta = beta.copy()
    if beta.sum() < 0:
        beta = -beta
    beta_s = np.sort(beta)

    S1 = np.empty(len(beta_s));  S1[0] = 0.0;  S1[1:] = np.cumsum(beta_s[:-1])
    S2 = np.empty(len(beta_s));  S2[0] = 0.0;  S2[1:] = np.cumsum(beta_s[:-1] ** 2)
    R  = 1.0 / var_f + (S2 - beta_s * S1) / var_eps

    k = int(np.where(R > 0)[0][-1]) + 1
    return k


# ---------------------------------------------------------------------------
# Theoretical limiting active ratio F(beta*)
# ---------------------------------------------------------------------------

def compute_theoretical_limit(beta_mean, beta_std):
    """
    Compute beta* (root of G) and F(beta*) = Phi((beta* - mu) / sigma).

    G(y) = (mu*(mu-y) + sigma^2)*Phi(alpha) - sigma*mu*phi(alpha),
    alpha = (y - mu)/sigma.

    Returns (beta_star, F_limit).  If G(0) <= 0 returns (0, F(0)).
    Note: beta* and F(beta*) depend only on the beta distribution F,
    not on var_f or var_eps.
    """
    mu, sigma = beta_mean, beta_std

    def G(y):
        alpha = (y - mu) / sigma
        return (mu * (mu - y) + sigma ** 2) * norm.cdf(alpha) \
               - sigma * mu * norm.pdf(alpha)

    g0 = G(0.0)
    if g0 <= 0.0:
        return 0.0, float(norm.cdf(-mu / sigma))

    y_hi = max(mu + 10.0 * sigma, 1.0)
    while G(y_hi) > 0.0:
        y_hi *= 2.0

    beta_star = brentq(G, 0.0, y_hi, xtol=1e-12, rtol=1e-12)
    F_limit   = float(norm.cdf((beta_star - mu) / sigma))
    return beta_star, F_limit


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def compute_ratios_grid(p_values, beta_stds, beta_mean, var_f, var_eps,
                        num_trials=400, base_seed=0):
    """
    For each (p, beta_std) cell draw num_trials beta vectors and compute k_p/p.

    Returns dict mapping (p, beta_std) -> (num_trials,) array of k_p/p.
    """
    results = {}
    for si, std in enumerate(beta_stds):
        for pi, p in enumerate(p_values):
            cell_seed = base_seed + si * len(p_values) * num_trials + pi * num_trials
            ratios = np.empty(num_trials)
            for t in range(num_trials):
                rng  = np.random.default_rng(cell_seed + t)
                beta = rng.normal(loc=beta_mean, scale=std, size=p)
                ratios[t] = compute_active_k(beta, var_f, var_eps) / p
            results[(p, std)] = ratios
            print(f"  std={std:.2f}, p={p:6d}: mean={ratios.mean():.5f}  "
                  f"std={ratios.std():.5f}")
    return results


# ---------------------------------------------------------------------------
# Figure: boxplots (no figure title, no sub-panel captions)
# ---------------------------------------------------------------------------

def plot_active_ratio_grid(results, p_values, beta_stds, beta_mean,
                           var_f, var_eps, num_trials, theoretical_limits,
                           png_filename="active_ratio_boxplots.png"):
    """
    One figure with len(beta_stds) panels.  Each panel:
      - Side-by-side boxplots of k_p/p for the three p values.
      - Blue diamond: mean across trials.
      - Orange dashed line: theoretical limit F(beta*).
      - Legend inside panel also shows beta* and % negative betas.
    No figure-level suptitle and no sub-panel text captions are added;
    those go to the companion .txt file instead.
    """
    n_panels = len(beta_stds)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    x_labels = [f"p = {p:,}" for p in p_values]
    positions = np.arange(1, len(p_values) + 1)

    for ax, std in zip(axes, beta_stds):
        data = [results[(p, std)] for p in p_values]

        ax.boxplot(data, positions=positions, vert=True, patch_artist=True,
                   widths=0.5,
                   boxprops=dict(facecolor="lightgreen", color="darkgreen"),
                   medianprops=dict(color="red", linewidth=2),
                   whiskerprops=dict(color="darkgreen"),
                   capprops=dict(color="darkgreen"),
                   flierprops=dict(marker="o", markerfacecolor="gray",
                                   markersize=3, linestyle="none"))

        # Mean markers
        means = [r.mean() for r in data]
        ax.plot(positions, means, "D", color="blue", markersize=5,
                label="mean", zorder=5)

        # Theoretical limit line
        beta_star, F_limit = theoretical_limits[std]
        x_lo = positions[0] - 0.45
        x_hi = positions[-1] + 0.45
        ax.hlines(F_limit, x_lo, x_hi,
                  colors="darkorange", linewidths=1.8, linestyles="--",
                  label=fr"$F(\beta^*) = {F_limit:.5f}$", zorder=6)

        # Extra legend entries: beta* and % negative betas
        frac_neg = float(norm.cdf(-beta_mean / std))
        ax.plot([], [], " ", label=fr"$\beta^* = {beta_star:.5f}$")
        ax.plot([], [], " ", label=fr"{100 * frac_neg:.4f}% negative betas")

        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel(r"Active ratio $k_p\,/\,p$", fontsize=11)
        ax.set_title(fr"$\beta \sim \mathcal{{N}}({beta_mean},\, {std}^2)$",
                     fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.5f"))
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(png_filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {png_filename}")
    plt.show()


# ---------------------------------------------------------------------------
# Text caption file
# ---------------------------------------------------------------------------

def write_caption_text(results, p_values, beta_stds, beta_mean,
                       var_f, var_eps, num_trials, theoretical_limits,
                       txt_filename="active_ratio_caption.txt"):
    """
    Write a plain-text description of the figure to txt_filename.
    This contains the information that would otherwise appear as figure
    titles and sub-panel captions.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("LOMV Active Ratio Study — Figure Description")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Figure: Distribution of LOMV active ratio k_p / p")
    lines.append(f"  beta_i ~ N(mu={beta_mean}, sigma^2),  var_f={var_f},  "
                 f"var_eps={var_eps}")
    lines.append(f"  p values : {p_values}")
    lines.append(f"  trials   : {num_trials} independent draws of beta per cell")
    lines.append("")
    lines.append("Each panel shows side-by-side boxplots of k_p/p for the three")
    lines.append("p values.  The orange dashed line marks the theoretical limit")
    lines.append("F(beta*) = lim_{p->inf} k_p/p (Theorem 2 of the paper),")
    lines.append("where beta* is the unique positive root of")
    lines.append("  G(y) = E[beta*(beta-y)*1_{beta<=y}] = 0.")
    lines.append("Blue diamonds mark the sample mean across trials.")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Per-panel summary")
    lines.append("-" * 70)

    for std in beta_stds:
        beta_star, F_limit = theoretical_limits[std]
        frac_neg = float(norm.cdf(-beta_mean / std))
        lines.append("")
        lines.append(f"Panel: beta ~ N({beta_mean}, {std}^2)")
        lines.append(f"  beta*           = {beta_star:.6f}")
        lines.append(f"  F(beta*)        = {F_limit:.6f}  "
                     f"(theoretical limit of k_p/p as p -> inf)")
        lines.append(f"  % negative betas approx {100 * frac_neg:.4f}%  "
                     f"[= Phi(-{beta_mean}/{std})]")
        lines.append("")
        lines.append(f"  {'p':>8}  {'mean k/p':>12}  {'std k/p':>12}  "
                     f"{'median k/p':>12}")
        for p in p_values:
            r = results[(p, std)]
            lines.append(f"  {p:>8,}  {r.mean():>12.6f}  {r.std():>12.6f}  "
                         f"{np.median(r):>12.6f}")

    lines.append("")
    lines.append("=" * 70)

    with open(txt_filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Caption text saved to {txt_filename}")


# ---------------------------------------------------------------------------
# LaTeX summary table
# ---------------------------------------------------------------------------

def generate_latex_table(all_results, p_values, beta_stds, beta_mean,
                         var_f, theoretical_limits,
                         tex_filename="active_ratio_table.tex"):
    """
    Write a LaTeX table summarising mean +/- std of k_p/p for all 18 cells:
      3 sigma values  x  3 p values  x  2 var_eps values.

    Rows are grouped by sigma (with two rows per sigma, one per var_eps).
    Columns are p values.  F(beta*) (which depends only on the beta
    distribution, not on var_eps) is shown once per sigma group.
    """
    var_eps_values = sorted(all_results.keys(), reverse=True)   # 0.5 then 0.1

    lines = []
    lines.append(r"% Active ratio summary table — generated by active_ratio_2.py")
    lines.append(r"% Requires \usepackage{booktabs} and \usepackage{multirow}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Mean $\pm$ standard deviation of the LOMV active ratio "
        r"$k_p/p$ over 400 trials. "
        r"$\beta_i \sim \mathcal{N}(\mu=" + str(beta_mean) + r",\,\sigma^2)$, "
        r"$\mathrm{var\_f}=" + str(var_f) + r"$. "
        r"$F(\beta^*)$ is the theoretical limit as $p\to\infty$ "
        r"(depends on $\sigma$ only, not on $\delta^2$).}"
    )
    lines.append(r"\label{tab:active_ratio}")

    n_p = len(p_values)
    # Column spec: sigma | delta^2 | F(beta*) | p columns
    col_spec = "ll" + "c" + "|" + "c" * n_p
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row
    p_headers = " & ".join(
        r"$p=" + f"{p:,}".replace(",", r"{,}") + r"$" for p in p_values
    )
    lines.append(
        r"$\sigma$ & $\delta^2$ & $F(\beta^*)$ & " + p_headers + r" \\"
    )
    lines.append(r"\midrule")

    for si, std in enumerate(beta_stds):
        beta_star, F_limit = theoretical_limits[std]
        if si > 0:
            lines.append(r"\midrule")
        for ri, var_eps in enumerate(var_eps_values):
            results = all_results[var_eps]

            # sigma and F(beta*) shown only in first row of each group
            if ri == 0:
                sigma_cell   = f"{std}"
                F_limit_cell = f"{F_limit:.5f}"
            else:
                sigma_cell   = ""
                F_limit_cell = ""

            delta_sq_cell = f"{var_eps}"

            data_cells = " & ".join(
                f"${results[(p, std)].mean():.4f} \\pm {results[(p, std)].std():.4f}$"
                for p in p_values
            )
            lines.append(
                f"{sigma_cell} & {delta_sq_cell} & {F_limit_cell} & "
                + data_cells + r" \\"
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines) + "\n"

    with open(tex_filename, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {tex_filename}")
    print("\n" + latex_str)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestActiveRatio(unittest.TestCase):

    def test_k_always_positive(self):
        """k must be at least 1."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            beta = rng.normal(1.0, 0.4, size=50)
            k = compute_active_k(beta, var_f=1.0, var_eps=0.5)
            self.assertGreaterEqual(k, 1)

    def test_k_at_most_p(self):
        """k must not exceed p."""
        rng = np.random.default_rng(1)
        for p in [10, 100, 500]:
            beta = rng.normal(1.0, 0.4, size=p)
            k = compute_active_k(beta, var_f=1.0, var_eps=0.5)
            self.assertLessEqual(k, p)

    def test_ratio_in_unit_interval(self):
        """k_p/p must lie in (0, 1]."""
        rng = np.random.default_rng(2)
        for p in [100, 500]:
            for _ in range(5):
                beta = rng.normal(1.0, 0.3, size=p)
                k = compute_active_k(beta, var_f=1.0, var_eps=0.5)
                ratio = k / p
                self.assertGreater(ratio, 0.0)
                self.assertLessEqual(ratio, 1.0 + 1e-10)

    def test_sign_invariance(self):
        """k must be the same whether beta or -beta is passed."""
        rng = np.random.default_rng(3)
        beta = rng.normal(1.0, 0.4, size=100)
        k_pos = compute_active_k( beta, var_f=1.0, var_eps=0.5)
        k_neg = compute_active_k(-beta, var_f=1.0, var_eps=0.5)
        self.assertEqual(k_pos, k_neg)

    def test_all_positive_betas_small_ratio(self):
        """With all-positive betas (mean=1, small std), k/p should be small."""
        rng = np.random.default_rng(4)
        beta = np.abs(rng.normal(1.0, 0.1, size=1000))
        k = compute_active_k(beta, var_f=1.0, var_eps=0.5)
        self.assertLess(k / len(beta), 0.1)

    def test_compute_ratios_grid_shape(self):
        """Result dict must have one entry per (p, std) pair."""
        res = compute_ratios_grid(p_values=[50, 100], beta_stds=[0.4],
                                  beta_mean=1.0, var_f=1.0, var_eps=0.5,
                                  num_trials=5, base_seed=0)
        self.assertIn((50,  0.4), res)
        self.assertIn((100, 0.4), res)
        self.assertEqual(len(res[(50, 0.4)]), 5)

    def test_larger_p_tighter_distribution(self):
        """Larger p should give less variable active ratios."""
        res = compute_ratios_grid(p_values=[200, 2000], beta_stds=[0.4],
                                  beta_mean=1.0, var_f=1.0, var_eps=0.5,
                                  num_trials=50, base_seed=99)
        self.assertLess(res[(2000, 0.4)].std(), res[(200, 0.4)].std())


class TestTheoreticalLimit(unittest.TestCase):

    def test_F_limit_in_unit_interval(self):
        """F(beta*) must lie in [0, 1]."""
        for mu, sigma in [(1.0, 0.4), (1.0, 0.25), (1.0, 0.1)]:
            _, F_lim = compute_theoretical_limit(mu, sigma)
            self.assertGreaterEqual(F_lim, 0.0)
            self.assertLessEqual(F_lim, 1.0)

    def test_G_vanishes_at_beta_star(self):
        """G(beta*) should be (nearly) zero at the computed root."""
        for mu, sigma in [(1.0, 0.4), (1.0, 0.25)]:
            beta_star, _ = compute_theoretical_limit(mu, sigma)
            if beta_star > 0:
                alpha = (beta_star - mu) / sigma
                G_val = (mu * (mu - beta_star) + sigma**2) * norm.cdf(alpha) \
                        - sigma * mu * norm.pdf(alpha)
                self.assertAlmostEqual(G_val, 0.0, places=8)

    def test_larger_std_gives_larger_limit(self):
        """More spread in betas -> larger F(beta*)."""
        _, F_lo = compute_theoretical_limit(beta_mean=1.0, beta_std=0.1)
        _, F_hi = compute_theoretical_limit(beta_mean=1.0, beta_std=0.4)
        self.assertLess(F_lo, F_hi)

    def test_limit_consistent_with_large_p_simulation(self):
        """For large p, simulated mean k_p/p should be close to F(beta*)."""
        mu, sigma = 1.0, 0.4
        _, F_lim = compute_theoretical_limit(mu, sigma)
        rng = np.random.default_rng(777)
        ratios = [compute_active_k(rng.normal(mu, sigma, 5000),
                                   var_f=1.0, var_eps=0.5) / 5000
                  for _ in range(30)]
        self.assertAlmostEqual(np.mean(ratios), F_lim, delta=0.003)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    var_f      = 1.0
    beta_mean  = 1.0
    num_trials = 400
    p_values   = [500, 3_000, 10_000]
    beta_stds  = [0.4, 0.25, 0.1]

    # Theoretical limits depend only on the beta distribution (not var_eps)
    print("Computing theoretical limits F(beta*) ...")
    theoretical_limits = {}
    for std in beta_stds:
        beta_star, F_lim = compute_theoretical_limit(beta_mean, std)
        theoretical_limits[std] = (beta_star, F_lim)
        print(f"  sigma={std:.2f}:  beta*={beta_star:.6f},  F(beta*)={F_lim:.6f}")

    all_results = {}

    for var_eps, label in [(0.5, "delta0.5"), (0.1, "delta0.1")]:
        print(f"\n=== var_eps = {var_eps} ===")
        results = compute_ratios_grid(p_values, beta_stds, beta_mean,
                                      var_f, var_eps,
                                      num_trials=num_trials, base_seed=0)
        all_results[var_eps] = results

        png_file = f"active_ratio_{label}.png"
        txt_file = f"active_ratio_{label}.txt"

        plot_active_ratio_grid(results, p_values, beta_stds, beta_mean,
                               var_f, var_eps, num_trials, theoretical_limits,
                               png_filename=png_file)

        write_caption_text(results, p_values, beta_stds, beta_mean,
                           var_f, var_eps, num_trials, theoretical_limits,
                           txt_filename=txt_file)

    # LaTeX table combining both var_eps experiments
    generate_latex_table(all_results, p_values, beta_stds, beta_mean,
                         var_f, theoretical_limits,
                         tex_filename="active_ratio_table.tex")

    # Unit tests
    print("\n--- Running unit tests ---")
    loader = unittest.TestLoader()
    for cls in [TestActiveRatio, TestTheoreticalLimit]:
        suite = loader.loadTestsFromTestCase(cls)
        unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    main()
