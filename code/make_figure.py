"""Render the mu-sweep trade-off figure (two stacked panels, shared mu axis;
no dual axis). Top: identified rho(mu) with the non-identified interval at mu=0.
Bottom: routing distance(mu). Saves PDF (for LaTeX) and PNG (to inspect)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIGS, exist_ok=True)

data = json.load(open(os.path.join(RESULTS, "mu_sweep.json")))
mus = data["mu_grid"]
rho = data["rho_mean"]
dist = data["dist_mean"]
lo, hi = data["rho0_interval_mean"]

# identified rho points (mu > 0)
mx = [m for m in mus if m > 0]
ry = [rho[str(m)] for m in mx]
dx = mus
dy = [dist[str(m)] for m in mus]

INK = "#1b3a5b"        # single, print-safe, colorblind-safe dark blue
BAND = "#c8d6e5"
GRID = "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.8, "font.family": "serif"})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

# --- Panel A: rho(mu)
ax1.axvspan(-0.6, 0.6, ymin=0, ymax=1, color="none")  # keep margins
ax1.fill_between([-0.4, 0.4], [lo, lo], [hi, hi], color=BAND, zorder=1)
ax1.plot([0, 0], [lo, hi], color="#7089a5", lw=2, zorder=2)
ax1.annotate("not identified\nat $\\mu=0$", xy=(0, hi), xytext=(1.6, hi + 0.006),
             fontsize=8, color="#4a4a4a", va="bottom")
ax1.plot(mx, ry, "-o", color=INK, lw=2, ms=5, zorder=3)
ax1.set_ylabel(r"return-flow ratio $\rho$")
ax1.grid(True, color=GRID, lw=0.7)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

# --- Panel B: distance(mu)
ax2.plot(dx, dy, "-o", color=INK, lw=2, ms=5, zorder=3)
ax2.set_ylabel("routing distance")
ax2.set_xlabel(r"deferral penalty $\mu$")
ax2.grid(True, color=GRID, lw=0.7)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

ax1.set_xlim(-0.8, 20.8)
fig.align_ylabels([ax1, ax2])
fig.savefig(os.path.join(FIGS, "mu_tradeoff.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "mu_tradeoff.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(FIGS, "mu_tradeoff.pdf"))
