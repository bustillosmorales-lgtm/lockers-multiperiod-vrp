"""Scaling figure for David's Kandoo-derived 48-instance benchmark, in the repo
style. Panel A: exact vs matheuristic solve time by size (reliable regime).
Panel B: matheuristic optimality gap by size. n>=20 shaded as the open-source
tractability frontier."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))

rows = json.load(open(os.path.join(RESULTS, "benchmark_david.json")))["rows"]

INK = "#1b3a5b"
INK2 = "#a34a2a"       # second series (heuristic)
BAND = "#eee2d8"
GRID = "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#888888",
                     "axes.linewidth": 0.8, "font.family": "serif"})

sizes = sorted(set(r["n"] for r in rows))


def agg(field, pred):
    xs, ys = [], []
    for n in sizes:
        vals = [r[field] for r in rows if r["n"] == n and pred(r)
                and r[field] is not None]
        if vals:
            xs.append(n)
            ys.append(sum(vals) / len(vals))
    return xs, ys


# reliable = n<=15; frontier = n>=20
ex_x, ex_y = agg("exact_time_s", lambda r: r["n"] <= 15 and not r["cbc_flagged"]
                 and abs(r["exact_obj"] - round(r["exact_obj"])) < 1e-6)
he_x, he_y = agg("heur_time_s", lambda r: r["heur_obj"] is not None)
gap_x, gap_y = agg("heur_vs_exact_pct", lambda r: r["n"] <= 15)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 4.8), sharex=True,
                               gridspec_kw={"height_ratios": [1.15, 1],
                                            "hspace": 0.14})

FR = 17.5   # frontier boundary between n=15 and n=20
for ax in (ax1, ax2):
    ax.axvspan(FR, 32, color=BAND, zorder=0)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# Panel A: solve time
ax1.plot(ex_x, ex_y, "-o", color=INK, lw=2, ms=5, label="exact MILP (CBC)")
ax1.plot(he_x, he_y, "-s", color=INK2, lw=2, ms=5, label="matheuristic")
ax1.set_ylabel("mean solve time (s)")
ax1.set_yscale("log")
ax1.legend(frameon=False, fontsize=8, loc="upper left")
ax1.annotate("open-source\nfrontier ($n\\geq20$)", xy=(25, ax1.get_ylim()[1]),
             xytext=(19.5, 1.2), fontsize=8, color="#7a6a5a", va="top")

# Panel B: heuristic gap
ax2.plot(gap_x, gap_y, "-s", color=INK2, lw=2, ms=5)
ax2.set_ylabel("mean heur. gap (%)")
ax2.set_xlabel("number of lockers $n$")
ax2.set_ylim(bottom=-0.15)

ax1.set_xlim(3, 32)
fig.align_ylabels([ax1, ax2])
fig.savefig(os.path.join(FIGS, "benchmark_david.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIGS, "benchmark_david.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(FIGS, "benchmark_david.pdf"))
print("exact:", list(zip(ex_x, [round(y, 1) for y in ex_y])))
print("heur :", list(zip(he_x, [round(y, 1) for y in he_y])))
print("gap  :", list(zip(gap_x, [round(y, 2) for y in gap_y])))
