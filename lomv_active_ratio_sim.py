"""
lomv_active_ratio_sim.py

Figure 3.1 and Table 3.1 of

    A. Kercheval and O. Sowunmi,
    "The Long-Only Minimum Variance Portfolio in a One-Factor Market:
     Theory and Asymptotics."
(Figure, table, theorem and section numbers refer to the current manuscript.)

Monte Carlo study of the active ratio k_p/p of the long-only minimum
variance (LOMV) portfolio, i.e. the fraction of the p assets that receive
positive weight in

    minimize w' Sigma w  subject to  sum(w) = 1, w >= 0,

under the one-factor covariance model

    Sigma = sigma^2 * beta beta^T + delta^2 * I,
    sigma = 0.15, delta = 0.30   (Clarke, de Silva & Thorley, 2011).

Two distributions F for the iid betas:
    "normal"    : beta_i ~ N(1, 0.4^2)                 (P(beta < 0) ~ 0.62%)
    "truncated" : beta_i ~ N(1, 0.4^2) conditioned on beta_i >= 0
                  (implemented by redrawing every negative draw).
For each distribution and each p in {500, 3000, 10000}: 400 independent trials.

Active set.  By Theorem 2.2 of the paper (explicit LOMV solution), with the
betas sorted increasingly and delta_i = delta,
    R_i = 1/sigma^2 + sum_{j<=i} (beta_j/delta^2)(beta_j - beta_i),
    k_p = max{ i : R_i > 0 },
and the active set consists of the k_p assets with the smallest betas.
Each trial is also verified independently against the KKT conditions of the
LOMV problem (function kkt_check), which does not use Theorem 2.2.

Population limits (dashed lines in the figure).  With
    G(y) = int_{x <= y} x (x - y) dF(x),
Theorem 2.5 of the paper (asymptotic active ratio) gives k_p/p -> F(beta*) a.s.:
    "normal"    : Case 1, beta* = unique positive zero of G  (~0.28842),
                  F(beta*) ~ 0.03762;
    "truncated" : Case 3, beta* = 0 and F(beta*) = 0.

Finite-p predictor (Table 3.1; a heuristic, *not* a theorem).  Let y_p > beta*
solve G(y_p) = -delta^2 / (p sigma^2) and predict k_p/p by F(y_p).  The
script compares F(y_p) with the simulated mean active ratio.

Outputs (written to OUTDIR):
    active_ratio_cst_combined.png   two-panel boxplot figure (Figure 3.1)
    active_ratio_cst_predictor.tex  LaTeX tabular for Table 3.1
    active_ratio_cst_trials.csv     every trial's active ratio
and summary statistics printed to stdout.

Requires numpy, scipy, matplotlib (>= 3.9, for the boxplot tick_labels
keyword).  Runtime: well under a minute.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SIGMA = 0.15                   # factor volatility
DELTA = 0.30                   # specific volatility (same for all assets)
MU_B, S_B = 1.0, 0.4           # beta ~ N(MU_B, S_B^2), possibly truncated at 0
P_LIST = [500, 3000, 10000]    # numbers of assets
N_TRIALS = 400                 # trials per (distribution, p)
SEED = 20261022                # master seed; all trials use one generator
DISTS = ["normal", "truncated"]
OUTDIR = "."

# Note: the LOMV portfolio is unchanged if Sigma is multiplied by a positive
# constant, so k_p depends on (sigma, delta) only through delta^2/sigma^2 = 4.


# ---------------------------------------------------------------------------
# Beta sampling
# ---------------------------------------------------------------------------
def draw_betas(rng, p, dist):
    """Draw p iid betas from N(MU_B, S_B^2); for dist == 'truncated',
    redraw negatives until none remain (= conditioning on beta >= 0)."""
    b = rng.normal(MU_B, S_B, size=p)
    if dist == "truncated":
        neg = b < 0
        while neg.any():
            b[neg] = rng.normal(MU_B, S_B, size=neg.sum())
            neg = b < 0
    return b


# ---------------------------------------------------------------------------
# Active set via Theorem 2.2
# ---------------------------------------------------------------------------
def active_count(beta, sigma, delta):
    """Return (k, sorted betas), k = max{i : R_i > 0}, for constant delta."""
    if beta.sum() < 0:                 # sign convention: sum beta_i/delta_i^2 >= 0
        beta = -beta                   # (Sigma is unchanged by beta -> -beta)
    b = np.sort(beta)
    S1 = np.cumsum(b)                  # sum_{j<=i} beta_j
    S2 = np.cumsum(b * b)              # sum_{j<=i} beta_j^2
    R = 1.0 / sigma**2 + (S2 - b * S1) / delta**2
    pos = np.nonzero(R > 0)[0]         # nonempty, since R_1 = 1/sigma^2 > 0
    return pos[-1] + 1, b


def kkt_check(b_sorted, k, sigma, delta, rtol=1e-9):
    """Certify that the portfolio supported on the first k sorted assets is the
    LOMV solution.  The candidate is the long-short minimum variance portfolio
    of the k-asset subproblem, w_K = Sigma_K^{-1} 1 / (1' Sigma_K^{-1} 1).
    Since the problem is convex, the KKT conditions are sufficient:
        w_K > 0,  and  (Sigma w)_i >= lambda = w' Sigma w  for every i > k
    (equality on K holds by construction).  Sigma_K^{-1} 1 is computed by
    Sherman-Morrison; delta is constant."""
    bK = b_sorted[:k]
    s2, d2 = sigma**2, delta**2
    # Sigma_K^{-1} 1 = (1/d2) [1 - c bK],  c = (s2 sum(bK)/d2) / (1 + s2 |bK|^2/d2)
    c = s2 * bK.sum() / d2 / (1.0 + s2 * (bK @ bK) / d2)
    x = (1.0 - c * bK) / d2
    wK = x / x.sum()
    lam = 1.0 / x.sum()                          # = w' Sigma w
    pos_ok = np.all(wK > 0)
    exposure = bK @ wK                           # beta' w
    Sw_inactive = s2 * b_sorted[k:] * exposure   # (Sigma w)_i for i > k, where w_i = 0
    inact_ok = np.all(Sw_inactive >= lam * (1 - rtol))
    return bool(pos_ok and inact_ok)


# ---------------------------------------------------------------------------
# Population quantities: F, G, beta*, F(beta*)
# ---------------------------------------------------------------------------
def _partial_moments(y, m=MU_B, s=S_B):
    """E[X; X<=y] and E[X^2; X<=y] for X ~ N(m, s^2) (standard closed forms)."""
    z = (y - m) / s
    Phi, phi = norm.cdf(z), norm.pdf(z)
    EX1 = m * Phi - s * phi
    EX2 = (m**2 + s**2) * Phi - s * (m + y) * phi
    return EX1, EX2


def F_cdf(y, dist):
    """CDF of the beta distribution."""
    if dist == "normal":
        return norm.cdf(y, MU_B, S_B)
    q0 = norm.cdf(0, MU_B, S_B)                  # mass removed by truncation
    return max(norm.cdf(y, MU_B, S_B) - q0, 0.0) / (1.0 - q0)


def G(y, dist):
    """G(y) = int_{x<=y} (x^2 - y x) dF(x), in closed form."""
    EX1, EX2 = _partial_moments(y)
    if dist == "normal":
        return EX2 - y * EX1
    # truncated: integrate over [0, y] only, renormalized by P(N >= 0)
    if y <= 0:
        return 0.0
    EX1_0, EX2_0 = _partial_moments(0.0)
    q0 = norm.cdf(0, MU_B, S_B)
    return ((EX2 - EX2_0) - y * (EX1 - EX1_0)) / (1.0 - q0)


def G_quad(y, dist):
    """G by numerical quadrature; used only to cross-check the closed form."""
    if dist == "normal":
        f = lambda x: (x * x - y * x) * norm.pdf(x, MU_B, S_B)
        return quad(f, -np.inf, y)[0]
    q0 = norm.cdf(0, MU_B, S_B)
    f = lambda x: (x * x - y * x) * norm.pdf(x, MU_B, S_B) / (1.0 - q0)
    return quad(f, 0.0, y)[0] if y > 0 else 0.0


def population_limits(dist):
    """Return (beta*, F(beta*), P(beta < 0))."""
    if dist == "normal":                         # Case 1 of Theorem 2.5
        bstar = brentq(lambda y: G(y, dist), 1e-8, 10.0, xtol=1e-14)
        assert abs(G_quad(bstar, dist)) < 1e-10
        return bstar, F_cdf(bstar, dist), norm.cdf(0, MU_B, S_B)
    return 0.0, 0.0, 0.0                         # Case 3: beta* = 0


def predictor(p, dist):
    """Heuristic finite-p prediction: y_p > beta* with G(y_p) = -delta^2/(p sigma^2),
    returns (y_p, F(y_p)).  G is strictly decreasing on [beta*, inf) here."""
    bstar = population_limits(dist)[0]
    c = DELTA**2 / (p * SIGMA**2)
    yp = brentq(lambda y: G(y, dist) + c, bstar + 1e-10, 10.0, xtol=1e-14)
    assert abs(G_quad(yp, dist) + c) < 1e-9
    return yp, F_cdf(yp, dist)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(SEED)
    results = {d: {p: np.empty(N_TRIALS) for p in P_LIST} for d in DISTS}
    kkt_fail = 0
    for d in DISTS:
        for p in P_LIST:
            for t in range(N_TRIALS):
                beta = draw_betas(rng, p, d)
                k, b_sorted = active_count(beta, SIGMA, DELTA)
                if not kkt_check(b_sorted, k, SIGMA, DELTA):
                    kkt_fail += 1
                results[d][p][t] = k / p
    print(f"KKT verification failures: {kkt_fail} of "
          f"{len(DISTS) * len(P_LIST) * N_TRIALS}")
    return results


# ---------------------------------------------------------------------------
# Figure 3.1
# ---------------------------------------------------------------------------
TITLES = {
    "normal": r"$\beta \sim \mathcal{N}(1.0,\, 0.4^2)$",
    "truncated": r"$\beta \sim \mathcal{N}(1.0,\, 0.4^2)$ truncated at 0",
}


def draw_panel(ax, data, dist):
    """Boxplots of k_p/p over trials for each p, with means (diamonds) and the
    limit F(beta*) (dashed line)."""
    bstar, Fbstar, pneg = population_limits(dist)
    labels = [f"p = {p:,}" for p in P_LIST]
    ax.boxplot([data[p] for p in P_LIST], tick_labels=labels, widths=0.45,
               patch_artist=True,
               boxprops=dict(facecolor="lightgreen", color="darkgreen"),
               whiskerprops=dict(color="darkgreen"),
               capprops=dict(color="darkgreen"),
               medianprops=dict(color="red", linewidth=2),
               flierprops=dict(marker="o", markersize=3,
                               markerfacecolor="none", markeredgecolor="black"))
    ax.plot(range(1, len(P_LIST) + 1), [data[p].mean() for p in P_LIST],
            "D", color="blue", markersize=6, label="mean", zorder=5)
    ax.axhline(Fbstar, color="darkorange", linestyle="--", linewidth=2,
               label=rf"$F(\beta^*) = {Fbstar:.5f}$")
    ax.plot([], [], " ", label=rf"$\beta^* = {bstar:.5f}$")
    ax.plot([], [], " ", label=f"{100 * pneg:.4f}% negative betas")
    ax.set_title(TITLES[dist], fontsize=12)
    ax.set_ylabel(r"Active ratio $k_p\,/\,p$", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="upper right", fontsize=9)


def make_figure(results):
    # Both panels share one vertical scale so the distributions can be compared.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, d in zip(axes, DISTS):
        draw_panel(ax, results[d], d)
    axes[1].tick_params(labelleft=True)   # sharey hides these by default
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/active_ratio_cst_combined.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table 3.1 (predictor vs simulation) and per-trial CSV
# ---------------------------------------------------------------------------
def make_table(results):
    names = {"normal": r"$\mathcal{N}(1,0.4^2)$",
             "truncated": r"$\mathcal{N}(1,0.4^2)$, truncated"}
    blocks = []
    for d in DISTS:
        pred = [predictor(p, d) for p in P_LIST]
        yp = " & ".join(f"{y:.4f}" for y, _ in pred)
        Fy = " & ".join(f"{f:.4f}" for _, f in pred)
        sm = " & ".join(f"{results[d][p].mean():.4f}" for p in P_LIST)
        blocks.append(f"{names[d]} & $y_p$ & {yp} \\\\\n"
                      f" & $F(y_p)$ & {Fy} \\\\\n"
                      f" & simulated mean & {sm} \\\\")
    tex = (r"\begin{tabular}{lc|ccc}" "\n" r"\toprule" "\n"
           r"Beta distribution & & $p=500$ & $p=3{,}000$ & $p=10{,}000$ \\" "\n"
           r"\midrule" "\n"
           + ("\n" r"\midrule" "\n").join(blocks)
           + "\n" r"\bottomrule" "\n" r"\end{tabular}")
    with open(f"{OUTDIR}/active_ratio_cst_predictor.tex", "w") as f:
        f.write(tex + "\n")
    with open(f"{OUTDIR}/active_ratio_cst_trials.csv", "w") as f:
        f.write("distribution,p,trial,active_ratio\n")
        for d in DISTS:
            for p in P_LIST:
                for t, v in enumerate(results[d][p]):
                    f.write(f"{d},{p},{t},{v:.6f}\n")
    return tex


if __name__ == "__main__":
    res = run()
    make_figure(res)
    print(make_table(res))
    for d in DISTS:
        bstar, Fbstar, pneg = population_limits(d)
        print(f"{d}: beta* = {bstar:.5f}, F(beta*) = {Fbstar:.5f}, "
              f"P(beta<0) = {pneg:.5f}")
        for p in P_LIST:
            x = res[d][p]
            print(f"  p={p:6d}  mean={x.mean():.5f}  sd={x.std(ddof=1):.5f}  "
                  f"median={np.median(x):.5f}  min={x.min():.5f}  max={x.max():.5f}")
