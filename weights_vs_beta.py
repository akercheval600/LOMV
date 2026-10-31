"""
weights_vs_beta.py

Figure 1.1 of

    A. Kercheval and O. Sowunmi,
    "The Long-Only Minimum Variance Portfolio in a One-Factor Market:
     Theory and Asymptotics."
(Figure, table, theorem and section numbers refer to the current manuscript.)

Compares, asset by asset, the weights of two minimum variance portfolios
under the one-factor covariance model

    Sigma = sigma^2 * beta beta^T + Delta,    Delta = diag(delta_i^2):

  * GMV  (long-short):  minimize w' Sigma w  subject to  sum(w) = 1;
  * LOMV (long-only):   minimize w' Sigma w  subject to  sum(w) = 1, w >= 0.

Setup used in the paper:
    p = 5000 assets, beta_i iid N(1, 0.4^2), delta_i = 0.3 for all i,
    sigma = 0.15 (values from Clarke, de Silva & Thorley, 2011),
    random seed 42 (numpy.random.default_rng).

Both portfolios are computed in closed form; no numerical optimizer is used.

  GMV.  By the Sherman-Morrison formula,
        (Sigma^{-1} 1)_i = (1/delta_i^2) * (1 - beta_i / tau_GMV),
        tau_GMV = (1/sigma^2 + S2) / S1,
        S1 = sum_j beta_j/delta_j^2,   S2 = sum_j beta_j^2/delta_j^2,
     and w_GMV = Sigma^{-1} 1 / (1' Sigma^{-1} 1).

  LOMV. Theorem 2.2 of the paper. Sort the betas in increasing order and set
        R_i = 1/sigma^2 + sum_{j<=i} (beta_j/delta_j^2)(beta_j - beta_i).
     The active set is K = {1, ..., k} with k = max{i : R_i > 0}, and on K
     the weights are the long-short minimum variance weights of the
     k-asset subproblem (Proposition 2.1), which by the same Sherman-Morrison
     computation are
        w_i proportional to (1/delta_i^2) * (1 - beta_i / tau),
        tau = (1/sigma^2 + sum_{j<=k} beta_j^2/delta_j^2)
              / (sum_{j<=k} beta_j/delta_j^2),
     and w_i = 0 for i > k.

As an independent check (not relying on Theorem 2.2), the script verifies the
KKT conditions of the convex LOMV problem: with lambda = w' Sigma w,
(Sigma w)_i = lambda for i in K and (Sigma w)_i >= lambda for i not in K.

Output: weights_vs_beta.png, and a one-line summary printed to stdout
(k = 319 active LOMV assets, 34 negative betas, 3304 positive GMV weights).

Requires numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
P = 5000                  # number of assets
MU_B, SD_B = 1.0, 0.4     # beta_i ~ N(MU_B, SD_B^2)
SIGMA = 0.15              # factor volatility
DELTA = 0.30              # specific volatility (same for all assets)
SEED = 42

rng = np.random.default_rng(SEED)
beta = rng.normal(MU_B, SD_B, P)
d2 = np.full(P, DELTA**2)          # delta_i^2
s2 = SIGMA**2                      # sigma^2

# Sign convention of the paper (Section 2.1): sum_i beta_i/delta_i^2 >= 0.
# (Replacing beta by -beta leaves Sigma unchanged.)  Satisfied here.
assert np.sum(beta / d2) >= 0

# ---------------------------------------------------------------------------
# GMV (long-short) weights via Sherman-Morrison
# ---------------------------------------------------------------------------
S1, S2 = np.sum(beta / d2), np.sum(beta**2 / d2)
w_gmv = (1.0 / d2) * (1.0 - s2 * beta * S1 / (1.0 + s2 * S2))
w_gmv /= w_gmv.sum()

# ---------------------------------------------------------------------------
# LOMV (long-only) weights via Theorem 2.2
# ---------------------------------------------------------------------------
order = np.argsort(beta)               # indices sorting beta increasingly
bs, ds = beta[order], d2[order]        # sorted betas and delta^2
c1 = np.cumsum(bs / ds)                # sum_{j<=i} beta_j/delta_j^2
c2 = np.cumsum(bs**2 / ds)             # sum_{j<=i} beta_j^2/delta_j^2
R = 1.0 / s2 + c2 - bs * c1            # R_i of Theorem 2.2
k = int(np.nonzero(R > 0)[0][-1]) + 1  # k = max{i : R_i > 0}  (R_1 > 0 always)

tau = (1.0 / s2 + c2[k - 1]) / c1[k - 1]
active = np.zeros(P, dtype=bool)
active[order[:k]] = True               # K = the k assets with smallest beta
w_lomv = np.where(active, (1.0 / d2) * (1.0 - beta / tau), 0.0)
w_lomv /= w_lomv.sum()

# ---------------------------------------------------------------------------
# Independent KKT check of the LOMV solution
# ---------------------------------------------------------------------------
Sw = s2 * beta * (beta @ w_lomv) + d2 * w_lomv     # Sigma w, without forming Sigma
lam = w_lomv @ Sw                                  # = w' Sigma w
assert np.all(w_lomv[active] > 0)
assert np.allclose(Sw[active], lam, rtol=1e-10, atol=0)
assert np.all(Sw[~active] >= lam * (1 - 1e-12))

n_neg = int((beta < 0).sum())
n_gmv_pos = int((w_gmv > 0).sum())
print(f"k = {k}, tau = {tau:.4f}, negative betas = {n_neg}, "
      f"GMV positive = {n_gmv_pos} ({100 * n_gmv_pos / P:.1f}%), "
      f"GMV nonzero = {int((w_gmv != 0).sum())}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.scatter(beta, w_gmv, s=4, color="steelblue", alpha=0.6,
           label=f"GMV (long-short, all {P:,} assets)")
ax.scatter(beta, w_lomv, s=4, color="black", alpha=0.6,
           label=f"LOMV (long-only, {k} active assets)")
# Invisible handles, used only to put the summary counts in the legend.
ax.plot([], [], " ", label=f"Negative betas: {n_neg} ({100 * n_neg / P:.2f}%)")
ax.plot([], [], " ", label=f"GMV positive weights: {n_gmv_pos} "
                           f"({100 * n_gmv_pos / P:.1f}%)")
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.grid(axis="y", ls=":", lw=0.5)
ax.set_xlabel(r"Factor loading $\beta_i$", fontsize=12)
ax.set_ylabel(r"Portfolio weight $w_i$", fontsize=12)
ax.set_title(r"GMV vs LOMV portfolio weights against $\beta_i$" "\n"
             rf"$\beta_i \sim \mathcal{{N}}({MU_B},\ {SD_B}^2)$,  "
             rf"$\Sigma = {SIGMA}^2\,\beta\beta^\top + {DELTA}^2\,I$,  "
             rf"$p = {P // 1000}{{,}}{P % 1000:03d}$",
             fontsize=11)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout()
fig.savefig("weights_vs_beta.png", dpi=150)
