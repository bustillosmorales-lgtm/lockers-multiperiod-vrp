"""
Data figures for the paper, from results/*.json. Okabe-Ito colorblind-safe
palette; no dual axes (panels instead); legend for >=2 series; thin marks.
Renders PDF (LaTeX) + PNG (inspect) for each.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIGS, exist_ok=True)

# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"
GRID = "#e6e6e6"
plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.edgecolor": "#888888", "axes.linewidth": 0.8})


def load(name):
    return json.load(open(os.path.join(RES, name)))


def finish(ax):
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ---- 1. Non-identifiability of rho (C1): interval per instance + reported point
def fig_nonident():
    cert = load("rho_certificate.json")
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    for i, c in enumerate(cert):
        lo, hi = c["rho_identified_set_[rho_min,rho_max]"]
        y = i
        ax.plot([lo, hi], [y, y], color=BLUE, lw=8, solid_capstyle="round", alpha=0.35)
        ax.plot([lo, hi], [y, y], "|", color=BLUE, ms=14, mew=2)
        # reported endpoint: min delivery (Ymin) -> max rho; max delivery -> min rho
        ymin, ymax = c["delivery_interval_[Ymin,Ymax]"]
        rep = hi if c["paper_reported_total_delivery"] == ymin else lo
        ax.plot([rep], [y], "o", color=VERM, ms=9, zorder=5)
        name = "small (5 lockers)" if "small" in c["instance"] else "case study (15 lockers)"
        ax.text(lo - 0.004, y, name, ha="right", va="center", fontsize=9)
    ax.plot([], [], "o", color=VERM, label="value reported by the base paper")
    ax.plot([], [], color=BLUE, lw=8, alpha=0.35, label=r"identified set of $\rho$ (equal cost)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=1, frameon=False, fontsize=8.5)
    ax.set_yticks([]); ax.set_ylim(-0.6, len(cert) - 0.4)
    ax.set_xlabel(r"return-flow ratio $\rho$")
    finish(ax); ax.spines["left"].set_visible(False)
    save(fig, "nonident")


# ---- 2. rho by size: base (non-identified) vs identified (service objective)
def fig_rho_size():
    base = {5: .3012, 8: .3324, 10: .3592, 12: .3855, 15: .4115, 20: .5251, 25: .5132, 30: .5007}
    bs = load("benchmark_service.json")["by_size"]
    ident = {int(k): v["avg_rho"] for k, v in bs.items()}
    sizes = sorted(ident)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(sorted(base), [base[s] for s in sorted(base)], "--o", color=ORANGE,
            lw=2, ms=5, label="base objective (not identified)")
    ax.plot(sizes, [ident[s] for s in sizes], "-o", color=BLUE, lw=2, ms=5,
            label="service objective (identified)")
    ax.set_xlabel("number of lockers"); ax.set_ylabel(r"return-flow ratio $\rho$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    finish(ax); save(fig, "rho_size")


# ---- 3. identified rho by scenario (highlight reduced vehicle capacity)
def fig_rho_scenario():
    bsc = load("benchmark_service.json")["by_scenario"]
    order = ["low demand", "base", "high demand", "reduced fleet",
             "reduced vehicle capacity", "reduced locker capacity"]
    short = {"low demand": "low dem.", "base": "base", "high demand": "high dem.",
             "reduced fleet": "red. fleet", "reduced vehicle capacity": "red. veh. cap.",
             "reduced locker capacity": "red. lock. cap."}
    vals = [bsc[s]["avg_rho"] for s in order]
    colors = [VERM if s == "reduced vehicle capacity" else BLUE for s in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar([short[s] for s in order], vals, color=colors, width=0.66)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel(r"identified $\rho$"); ax.set_ylim(0, max(vals) * 1.18)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.5)
    finish(ax); save(fig, "rho_scenario")


# ---- 4. value of integration by capacity regime (C2)
def fig_voi_regime():
    s = load("voi_summary.json")["summary"]
    cells = [("loose_mu0", "loose\n$\\mu=0$"), ("loose_mu5", "loose\n$\\mu=5$"),
             ("tight_mu0", "tight\n$\\mu=0$"), ("tight_mu5", "tight\n$\\mu=5$")]
    frac = [100 * s[k]["frac_VoI_positive"] for k, _ in cells]
    colors = [GRAY, GRAY, BLUE, BLUE]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar([lab for _, lab in cells], frac, color=colors, width=0.62)
    for i, v in enumerate(frac):
        ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=8.5)
    ax.set_ylabel("instances with positive\nvalue of integration (%)")
    ax.set_ylim(0, max(frac) * 1.25 + 5)
    finish(ax); save(fig, "voi_regime")


# ---- 5. VoI at scale (prevalence by size) + horizon effect
def fig_voi_scale():
    sm = load("voi_scaled_summary.json")["summary"]
    sizes = sorted(int(k[1:]) for k in sm)
    frac = [100 * sm[f"n{n}"]["frac_VoI_positive"] for n in sizes]
    mx = [sm[f"n{n}"]["max_VoI_pct"] for n in sizes]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax1.bar([str(n) for n in sizes], frac, color=BLUE, width=0.6)
    ax1.set_ylabel("instances with VoI $>0$ (%)"); ax1.set_xlabel("number of lockers")
    ax1.set_ylim(0, 100); finish(ax1)
    # horizon effect
    tau = load("voi_tau_comparison.json")
    def frac_pos(rows):
        return 100 * sum(1 for r in rows if r["VoI"] > 1e-6) / len(rows)
    ax2.bar(["$\\tau=2$", "$\\tau=3$"], [frac_pos(tau["tau2"]), frac_pos(tau["tau3"])],
            color=[GRAY, BLUE], width=0.5)
    ax2.set_ylabel("instances with VoI $>0$ (%)"); ax2.set_ylim(0, 100)
    ax2.set_title("horizon effect (tight)", fontsize=9)
    finish(ax2); save(fig, "voi_scale")


# ---- 6. scaling: matheuristic vs exact wall (C3), fair 600 s budget
def fig_scaling():
    data = load("scaling_highs.json")
    rows = data["rows"]
    tl = data.get("highs_time_limit_s", 300)
    n = [r["n"] for r in rows]
    ht = [r["heur_time_s"] for r in rows]
    et = [r["highs_time_s"] for r in rows]
    ho = [r["heur_obj"] for r in rows]
    eo = [r["highs_obj"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 4.6), sharex=True,
                                   gridspec_kw={"hspace": 0.14})
    ax1.plot(n, ht, "-o", color=BLUE, lw=2, ms=5, label="matheuristic")
    en = [x for x, t in zip(n, et) if t is not None]
    ev = [t for t in et if t is not None]
    ax1.plot(en, ev, "--s", color=ORANGE, lw=2, ms=5, label="HiGHS solver")
    ax1.axhline(tl, color=GRAY, lw=1, ls=":")
    ax1.text(n[0], tl * 1.02, f"{tl}s time limit", color=GRAY, fontsize=8, va="bottom")
    ax1.set_ylabel("runtime (s)"); ax1.legend(frameon=False, fontsize=8.5, loc="center left")
    finish(ax1)
    ax2.plot(n, ho, "-o", color=BLUE, lw=2, ms=5, label="matheuristic (seconds)")
    en2 = [x for x, o in zip(n, eo) if o is not None]
    ev2 = [o for o in eo if o is not None]
    ax2.plot(en2, ev2, "--s", color=ORANGE, lw=2, ms=5, label="HiGHS (proven / 300 s incumbent)")
    ax2.set_ylabel("objective"); ax2.set_xlabel("number of lockers")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    finish(ax2); save(fig, "scaling")


# ---- 7. lower-bound tightening ladder (representative instance, mu=5)
def fig_bound():
    # n=5 seed1: exact=657; gaps to exact (%)
    labels = ["LP\nrelaxation", "+ subtour\ncuts", "+ branch\n-and-price",
              "+ 2-cycle\ninequalities", "+ 3-cycle\ninequalities"]
    gaps = [17.4, 16.0, 16.0, 4.6, 2.3]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    colors = [GRAY, GRAY, BLUE, BLUE, GREEN]
    ax.bar(labels, gaps, color=colors, width=0.62)
    for i, v in enumerate(gaps):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8.5)
    ax.set_ylabel("gap to exact optimum (%)"); ax.set_ylim(0, 20)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    finish(ax); save(fig, "bound_ladder")


# ---- 2b. within-instance rho: distance-only interval vs identified (same instances)
def fig_rho_within():
    bs = load("within_instance_rho.json")["by_size"]
    sizes = sorted(int(k) for k in bs)
    lo = [bs[str(n)]["rho_min_mean"] for n in sizes]
    hi = [bs[str(n)]["rho_max_mean"] for n in sizes]
    ident = [bs[str(n)]["rho_identified_mean"] for n in sizes]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.fill_between(sizes, lo, hi, color=ORANGE, alpha=0.25,
                    label="distance-only $\\rho$ (not identified)")
    ax.plot(sizes, lo, color=ORANGE, lw=1, alpha=0.6)
    ax.plot(sizes, hi, color=ORANGE, lw=1, alpha=0.6)
    ax.plot(sizes, ident, "-o", color=BLUE, lw=2, ms=5,
            label="service-aware $\\rho$ (identified)")
    ax.set_xlabel("number of lockers"); ax.set_ylabel(r"return-flow ratio $\rho$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    finish(ax); save(fig, "rho_within")


# ---- 4b. controlled VoI vs capacity threshold (exact joint vs exact myopic)
def fig_voi_controlled():
    bc = load("voi_controlled.json")["by_capacity_factor"]
    keys = sorted(bc, key=float, reverse=True)            # loose -> tight
    facs = [float(k) for k in keys]
    mean = [bc[k]["mean_VoI_pct"] for k in keys]
    mx = [bc[k]["max_VoI_pct"] for k in keys]
    x = list(range(len(facs)))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(x, mx, "--o", color=ORANGE, lw=1.6, ms=4, label="max VoI")
    ax.plot(x, mean, "-o", color=BLUE, lw=2, ms=5, label="mean VoI")
    ax.set_xticks(x); ax.set_xticklabels([f"{f:.2f}" for f in facs])
    ax.set_xlabel(r"vehicle capacity ($\times$ reference flow) --- tighter $\rightarrow$")
    ax.set_ylabel("value of integration (\\%)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    finish(ax); save(fig, "voi_controlled")


# ---- 4c. VoI MAGNITUDE by capacity and by horizon (conditional on positive)
def fig_voi_magnitude():
    d = load("voi_regime.json")
    bf = d["by_factor"]; bt = d["by_tau"]
    keys = sorted(bf, key=float)                          # tight -> loose
    facs = [float(k) for k in keys]
    cond = [bf[k]["cond_mean_pct"] for k in keys]
    mx = [bf[k]["max_pct"] for k in keys]
    x = list(range(len(facs)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                   gridspec_kw={"width_ratios": [3, 2]})
    ax1.plot(x, mx, "--o", color=ORANGE, lw=1.6, ms=4, label="ceiling (max)")
    ax1.plot(x, cond, "-o", color=BLUE, lw=2, ms=5, label="mean $|$ positive")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{f:.2f}" for f in facs], fontsize=8)
    ax1.set_xlabel(r"capacity ($\times$ ref. flow) --- tighter $\leftarrow$")
    ax1.set_ylabel("value of integration (\\%)")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    finish(ax1)
    taus = sorted(int(k) for k in bt)
    condt = [bt[str(t)]["cond_mean_pct"] for t in taus]
    mxt = [bt[str(t)]["max_pct"] for t in taus]
    xt = list(range(len(taus)))
    w = 0.36
    ax2.bar([i - w / 2 for i in xt], condt, width=w, color=BLUE, label="mean $|$ positive")
    ax2.bar([i + w / 2 for i in xt], mxt, width=w, color=ORANGE, label="ceiling (max)")
    ax2.set_xticks(xt); ax2.set_xticklabels([f"$\\tau={t}$" for t in taus])
    ax2.set_ylabel("value of integration (\\%)")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.set_title("horizon effect", fontsize=9)
    finish(ax2); save(fig, "voi_magnitude")


# ---- 6. real-geography benchmark: rho non-identifiability persists on real streets
def fig_real_benchmark():
    d = load("real_benchmark.json")
    rows = sorted(d["rows"], key=lambda r: r["rho_width"])
    widths = [r["rho_width"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                   gridspec_kw={"width_ratios": [3, 2]})
    # left: rho interval per instance (sorted by width), all non-degenerate
    y = list(range(len(rows)))
    for yi, r in zip(y, rows):
        ax1.plot([r["rho_lo"], r["rho_hi"]], [yi, yi], "-", color=BLUE, lw=1.4)
    ax1.plot([r["rho_lo"] for r in rows], y, ".", color=BLUE, ms=3)
    ax1.plot([r["rho_hi"] for r in rows], y, ".", color=BLUE, ms=3)
    ax1.axvspan(0, 0, color=GRAY)  # no-op keep axes
    ax1.set_xlabel(r"identified set of $\rho$ (real instances)")
    ax1.set_ylabel("instance (sorted by width)")
    ax1.set_xlim(0.28, 0.58); finish(ax1)
    # right: width distribution vs the synthetic reference (~0.025)
    ax2.hist(widths, bins=8, color=BLUE, edgecolor="white")
    ax2.axvline(0.025, color=VERM, lw=2, ls="--")
    ax2.text(0.03, ax2.get_ylim()[1] * 0.9, "synthetic\n$\\approx0.025$",
             color=VERM, fontsize=8, va="top")
    ax2.set_xlabel(r"interval width $\Delta\rho$")
    ax2.set_ylabel("real instances")
    finish(ax2); save(fig, "real_benchmark")


if __name__ == "__main__":
    fig_nonident(); fig_rho_size(); fig_rho_scenario(); fig_voi_regime()
    fig_voi_scale(); fig_scaling(); fig_bound(); fig_rho_within()
    fig_voi_controlled(); fig_voi_magnitude(); fig_real_benchmark()
