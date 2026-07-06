"""
Conceptual schematic (two panels) of the multi-period locker mechanism.
(a) A pickup whose destination is visited LATER on the route is completed same-day.
(b) A pickup whose destination was ALREADY visited cannot be delivered; it returns
    to the depot and is dispatched again the next day (overnight recirculation).
This order-dependence is the source of recirculation and of the identifiability
result. Saves PDF (LaTeX) + PNG (inspect).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch

FIGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIGS, exist_ok=True)

INK = "#1b3a5b"
ROUTE = "#9aa7b4"
SAMEDAY = "#2e7d32"
RETURN = "#c0392b"
plt.rcParams.update({"font.size": 10, "font.family": "serif"})

pos = {0: (0.6, 0.4), "A": (1.4, 2.4), "B": (3.4, 2.2)}
lab = {0: "depot", "A": r"$L_a$", "B": r"$L_b$"}


def draw_nodes(ax):
    for i, (x, y) in pos.items():
        if i == 0:
            ax.add_patch(FancyBboxPatch((x - 0.45, y - 0.26), 0.9, 0.52,
                         boxstyle="round,pad=0.02", fc="white", ec=INK, lw=1.6, zorder=3))
        else:
            ax.add_patch(Circle((x, y), 0.32, fc="white", ec=INK, lw=1.6, zorder=3))
        ax.text(x, y, lab[i], ha="center", va="center", zorder=4)


def arr(ax, a, b, color, style="-", lw=1.8, rad=0.0, z=2):
    ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=13, lw=lw, ls=style, color=color,
                 shrinkA=17, shrinkB=17, zorder=z))


fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax in (axa, axb):
    ax.set_xlim(-0.2, 4.4); ax.set_ylim(-0.4, 3.6); ax.axis("off")
    for a, b in [(0, "A"), ("A", "B")]:      # day-1 route
        arr(ax, a, b, ROUTE, lw=2.4, z=1)
    arr(ax, "B", 0, ROUTE, lw=2.4, rad=-0.35, z=1)
    draw_nodes(ax)

# (a) same-day: pickup at La, destination Lb (visited later)
arr(axa, "A", "B", SAMEDAY, lw=2.0, rad=0.45, z=2)
axa.text(2.4, 3.35, "pickup at $L_a$ for $L_b$", color=SAMEDAY, fontsize=8.5, ha="center")
axa.text(2.4, 2.98, "(visited later) $\\Rightarrow$ same-day", color=SAMEDAY, fontsize=8.5, ha="center")
axa.set_title("(a) same-day delivery", fontsize=10, color=INK)

# (b) return: pickup at Lb, destination La (already visited) -> depot, then next day
arr(axb, "B", 0, RETURN, style="--", lw=2.0, rad=0.42, z=2)
arr(axb, 0, "A", RETURN, style=":", lw=2.0, rad=-0.30, z=2)
axb.text(2.5, 3.35, "pickup at $L_b$ for $L_a$", color=RETURN, fontsize=8.5, ha="center")
axb.text(2.5, 2.98, "(already visited) $\\Rightarrow$ returns", color=RETURN, fontsize=8.5, ha="center")
axb.text(0.15, 1.5, "day 2", color=RETURN, fontsize=8, rotation=63, ha="left")
axb.set_title("(b) overnight recirculation", fontsize=10, color=INK)

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "mechanism.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "mechanism.png"), dpi=150, bbox_inches="tight")
print("wrote mechanism.pdf")
