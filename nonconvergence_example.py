"""
nonconvergence_example.py

Figure 3.2 of

    A. Kercheval and O. Sowunmi,
    "The Long-Only Minimum Variance Portfolio in a One-Factor Market:
     Theory and Asymptotics."
(Figure, table, theorem and section numbers refer to the current manuscript.)

Illustrates that the active ratio k_p/p of the long-only minimum variance
(LOMV) portfolio need not converge when beta* is an atom of the beta
distribution F (Case 1 of Theorem 2.5 of the paper; example of Section 2.3).

Beta distribution (four atoms):
    P(beta = -1) = 0.05,  P(beta = 1) = 0.15,
    P(beta =  2) = 0.30,  P(beta = 5) = 0.50.
Covariance:  Sigma = sigma^2 * beta beta^T + delta^2 * I,
             sigma^2 = 1, delta^2 = 0.1.

Population function
    G(y) = sum_a P(beta = a) * a * (a - y) * 1{a <= y}
has its unique zero at beta* = 2, an atom of mass 0.30, and Theorem 2.5 gives,
almost surely,
    liminf k_p/p = F(beta*-) = 0.20,     limsup k_p/p = F(beta*) = 0.50.

Why the trials are bimodal.  Betas tied at 2 contribute nothing to R_i at
beta_i = 2, so all assets in the beta = 2 block share the value
    R_block = 1/sigma^2 + (1/delta^2)(3 N_{-1} - N_1),
where N_a is the number of draws equal to a.  The whole block is therefore
either active (R_block > 0; k_p/p ~ 0.50, "high mode", red) or inactive
(k_p/p ~ 0.20, "low mode", blue).  Since E[3 N_{-1} - N_1] = 0, both occur
with substantial probability at every p.  (The fraction of trials in each
mode is a numerical observation; the theorem concerns a single realization
as p -> infinity.)

For each p in {500, 3000, 10000}, 400 independent trials.  k_p is computed by
Theorem 2.2 of the paper (explicit LOMV solution) and every trial is checked
against the KKT conditions of the LOMV problem.

Output: nonconvergence_example_clean.png (top: k_p/p by trial for each p;
bottom: the population function G), and per-p mode counts on stdout.

Requires numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
ATOMS = np.array([-1.0, 1.0, 2.0, 5.0])
PROBS = np.array([0.05, 0.15, 0.30, 0.50])
SIGMA2 = 1.0                  # sigma^2
DELTA2 = 0.1                  # delta^2 (same for all assets)
P_LIST = [500, 3000, 10000]
N_TRIALS = 400
SEED = 2026
BSTAR = 2.0                   # unique zero of G (verified below)
F_LEFT = PROBS[ATOMS < BSTAR].sum()    # F(beta*-) = 0.20  (liminf)
F_RIGHT = PROBS[ATOMS <= BSTAR].sum()  # F(beta*)  = 0.50  (limsup)


# ---------------------------------------------------------------------------
# Population function G and its zero
# ---------------------------------------------------------------------------
def G(y):
    """G(y) = sum_a P(beta=a) a (a - y) 1{a <= y}; vectorized in y."""
    y = np.asarray(y, dtype=float)[..., None]
    return np.sum(PROBS * ATOMS * (ATOMS - y) * (ATOMS <= y), axis=-1)


# G is piecewise linear with G(0) > 0 and G(y) -> -infinity; check that
# beta* = 2 is its only zero on (0, infinity) on a fine grid.
_yy = np.linspace(1e-6, 20, 200001)
_g = G(_yy)
assert abs(G(BSTAR)) < 1e-14
assert np.all(_g[_yy < BSTAR - 1e-9] > 0) and np.all(_g[_yy > BSTAR + 1e-9] < 0)


# ---------------------------------------------------------------------------
# LOMV active set via Theorem 2.2, and KKT check
# ---------------------------------------------------------------------------
def active_count(beta):
    """k = max{i : R_i > 0} for sorted betas and constant delta.
    We compute delta^2 * R_i = delta^2/sigma^2 + sum_{j<=i} beta_j (beta_j - beta_i),
    which has the same sign as R_i.  With integer-valued betas, the sums are
    computed exactly in floating point, so ties (the beta = 2 block) are
    handled without rounding error."""
    b = np.sort(beta)
    S1, S2 = np.cumsum(b), np.cumsum(b * b)
    Rs = DELTA2 / SIGMA2 + (S2 - b * S1)
    pos = np.nonzero(Rs > 0)[0]            # nonempty: first entry is delta^2/sigma^2 > 0
    return pos[-1] + 1, b


def kkt_check(b_sorted, k, rtol=1e-9):
    """Certify that the minimum variance portfolio of the first k sorted assets
    is the LOMV solution: its weights are positive and (Sigma w)_i >= w' Sigma w
    for all inactive i (KKT conditions, sufficient by convexity)."""
    bK = b_sorted[:k]
    c = SIGMA2 * bK.sum() / DELTA2 / (1.0 + SIGMA2 * (bK @ bK) / DELTA2)
    x = (1.0 - c * bK) / DELTA2               # Sigma_K^{-1} 1 (Sherman-Morrison)
    wK = x / x.sum()
    lam = 1.0 / x.sum()                        # w' Sigma w
    Sw_inactive = SIGMA2 * b_sorted[k:] * (bK @ wK)
    return bool(np.all(wK > 0) and np.all(Sw_inactive >= lam * (1 - rtol)))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run():
    rng = np.random.default_rng(SEED)
    ratio = {p: np.empty(N_TRIALS) for p in P_LIST}
    high = {p: np.empty(N_TRIALS, dtype=bool) for p in P_LIST}
    for p in P_LIST:
        for t in range(N_TRIALS):
            beta = rng.choice(ATOMS, size=p, p=PROBS)
            if beta.sum() < 0:                 # sign convention of Section 2.1 (essentially never
                beta = -beta                   # here: E[beta] = 3.2 > 0, so this is
                                               # astronomically unlikely)
            k, b = active_count(beta)
            assert kkt_check(b, k)
            # The beta* block is all-or-nothing: the last active asset either
            # has beta < 2 (block inactive) or is the last asset with beta = 2.
            n_block = int((b == BSTAR).sum())
            block_active = n_block > 0 and b[k - 1] == BSTAR
            assert (not block_active) or k == int((b <= BSTAR).sum())
            ratio[p][t] = k / p
            high[p][t] = block_active
    return ratio, high


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
BLUE, RED = "steelblue", "tomato"


def make_figure(ratio, high):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.45, wspace=0.18)
    jit = np.random.default_rng(0)             # jitter only; does not affect results

    # --- Top row: k_p/p across trials, one panel per p ---
    for j, p in enumerate(P_LIST):
        ax = fig.add_subplot(gs[0, j])
        x, h = ratio[p], high[p]
        # gold band: the non-convergence interval [F(beta*-), F(beta*)]
        ax.axhspan(F_LEFT, F_RIGHT, color="gold", alpha=0.15, zorder=0)
        ax.boxplot(x, positions=[0], widths=0.4, whis=(0, 100), showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="lightyellow", color="gray"),
                   whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
                   medianprops=dict(color="gray"))
        xs = jit.uniform(-0.15, 0.15, size=x.size)
        ax.scatter(xs[~h], x[~h], s=12, color=BLUE, alpha=0.6, zorder=3)
        ax.scatter(xs[h], x[h], s=12, color=RED, alpha=0.6, zorder=3)
        ax.axhline(F_LEFT, color="darkorange", ls="--", lw=2, label="liminf")
        ax.axhline(F_RIGHT, color="darkred", ls="-.", lw=2, label="limsup")
        # mode counts at the top of each panel
        ytop = x.max()
        ax.plot(-0.25, ytop, "o", color=BLUE, ms=6, clip_on=False)
        ax.text(-0.21, ytop, f"{(~h).sum()}", color=BLUE, va="center", fontsize=9)
        ax.plot(0.15, ytop, "o", color=RED, ms=6, clip_on=False)
        ax.text(0.19, ytop, f"{h.sum()}", color=RED, va="center", fontsize=9)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(rf"$p = {p:,}$".replace(",", "{,}"), fontsize=12)
        ax.grid(axis="y", ls=":", lw=0.5)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        if j == 0:
            ax.set_ylabel(r"$k_p\,/\,p$", fontsize=12)
            ax.legend(loc="center right", fontsize=8)

    # --- Bottom row: the population function G ---
    ax = fig.add_subplot(gs[1, :])
    y = np.linspace(0, 6.5, 2001)
    g = G(y)
    ax.plot(y, g, color="steelblue", lw=2, label=r"$G(y)$")
    ax.fill_between(y, g, 0, where=g > 0, color="limegreen", alpha=0.2,
                    label=r"$G>0$")
    ax.fill_between(y, g, 0, where=g < 0, color="salmon", alpha=0.25,
                    label=r"$G<0$")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.axvline(BSTAR, color="darkorange", ls="--", lw=1.5,
               label=rf"$\beta^* = {BSTAR:g}$")
    ax.plot(BSTAR, 0, "o", color="darkorange", ms=8, zorder=4)
    ax.set_xlim(0, 6.5)
    ax.set_xlabel(r"$y$", fontsize=12)
    ax.set_ylabel(r"$G(y)$", fontsize=12)
    ax.grid(ls=":", lw=0.5)
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig("nonconvergence_example_clean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ratio, high = run()
    print(f"liminf F(beta*-) = {F_LEFT:.2f}, limsup F(beta*) = {F_RIGHT:.2f}")
    for p in P_LIST:
        h = high[p]
        print(f"p={p:6d}: low mode {(~h).sum():3d}, high mode {h.sum():3d}, "
              f"low-mode mean {ratio[p][~h].mean():.4f}, "
              f"high-mode mean {ratio[p][h].mean():.4f}")
    make_figure(ratio, high)
