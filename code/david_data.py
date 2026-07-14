"""
Adapter for David's real data (delivered 2026-07: `nueva info/`).

Two sources, both mapped to the internal instance dict used by model_full /
matheuristic / voi (schema: n, T=[1..tau], d={(i,j):dist}, p={t:{j:{k}}},
g={t:{k}}, C, Q, H):

  1. build_case_study()      -- the real Kandoo Smart Locker network (Guayaquil,
     15 lockers). Distances read from DatosCasoEstudio.xlsx (rounded driven
     distances); depot dispatch g and locker-to-locker demand p taken from the
     manuscript's case-study tables (identical to the Excel OD block). This is
     the instance the manuscript solved with Gurobi (obj 270, gap 0).

  2. parse_instance_txt(path) -- one of the 48 benchmark scenario files in
     nueva info/instancias_txt/ (schema V/K/T/C/Q/H/g/p_day_t/d). Handles the
     asymmetric distance matrix, fractional demand, and either g orientation
     (rows=days or rows=nodes), and remaps David's 0-based T to internal 1-based.
"""
import ast
import os

# ---------------------------------------------------------------------------
# Prefer the in-repo copy (self-contained reproduction); fall back to the
# original delivery folder if the repo copy is absent.
_REPO = os.path.join(os.path.dirname(__file__), "..", "data")
_NUEVA = os.path.join(os.path.dirname(__file__), "..", "..", "nueva info")
if os.path.isdir(_REPO):
    CASE_XLSX = os.path.join(_REPO, "DatosCasoEstudio.xlsx")
    INSTANCES_DIR = os.path.join(_REPO, "instancias_txt")
else:
    CASE_XLSX = os.path.join(_NUEVA, "DatosCasoEstudio.xlsx")
    INSTANCES_DIR = os.path.join(_NUEVA, "instancias_txt", "instancias_txt")


# ---- 1. real case study ---------------------------------------------------
# g_k^1 (depot dispatch, period 1); period 2 is all zero (manuscript Table).
_CASE_G1 = [6, 8, 5, 4, 7, 3, 5, 6, 4, 5, 3, 4, 6, 2, 5]

# p_jk^1 and p_jk^2, lockers 1..15 (manuscript Tables 9-10 / Excel OD block).
_CASE_P1 = [
    [0, 1, 1, 1, 1, 1, 0, 2, 1, 3, 0, 1, 3, 0, 0],
    [0, 0, 3, 1, 3, 0, 0, 3, 2, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 2, 2, 1, 1, 2, 0, 1, 2, 0, 1, 0, 0],
    [2, 1, 1, 0, 2, 2, 2, 3, 2, 1, 1, 2, 1, 1, 1],
    [0, 1, 1, 2, 0, 1, 2, 0, 2, 3, 2, 0, 0, 0, 0],
    [0, 2, 3, 0, 2, 0, 1, 0, 2, 0, 2, 1, 1, 1, 2],
    [2, 1, 2, 1, 1, 0, 0, 0, 2, 0, 3, 2, 1, 2, 0],
    [2, 1, 0, 0, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 2],
    [2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 2],
    [2, 3, 0, 1, 2, 1, 2, 1, 2, 0, 0, 1, 3, 1, 0],
    [3, 1, 0, 0, 1, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1],
    [2, 2, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0],
    [3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 2, 0, 0, 1, 2],
    [1, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0, 0, 3],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 1, 0],
]
_CASE_P2 = [
    [0, 1, 1, 1, 1, 1, 0, 2, 1, 3, 0, 1, 3, 0, 0],
    [0, 0, 3, 1, 2, 0, 0, 3, 1, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 2, 2, 1, 1, 2, 0, 1, 2, 0, 1, 0, 0],
    [2, 1, 1, 0, 2, 2, 2, 3, 1, 1, 1, 2, 1, 1, 1],
    [0, 1, 1, 2, 0, 1, 2, 0, 2, 3, 2, 0, 0, 0, 0],
    [0, 2, 3, 0, 2, 0, 1, 0, 1, 0, 2, 1, 1, 1, 2],
    [2, 1, 2, 1, 1, 0, 0, 0, 2, 0, 3, 2, 1, 2, 0],
    [2, 1, 0, 0, 2, 2, 2, 0, 1, 0, 0, 0, 2, 1, 2],
    [2, 1, 0, 1, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 2],
    [2, 3, 0, 1, 2, 1, 2, 1, 1, 0, 0, 1, 3, 1, 0],
    [2, 1, 0, 0, 1, 0, 1, 2, 1, 2, 0, 0, 2, 0, 1],
    [2, 2, 0, 1, 1, 1, 1, 0, 2, 0, 0, 0, 1, 1, 0],
    [3, 2, 2, 0, 3, 0, 1, 1, 3, 0, 2, 0, 0, 1, 2],
    [1, 2, 1, 2, 1, 0, 3, 0, 2, 1, 2, 1, 0, 0, 3],
    [2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 2, 2, 1, 0],
]


def _case_distance_matrix():
    """16x16 rounded driven distances (depot 0 + 15 lockers) from the Excel."""
    import openpyxl
    wb = openpyxl.load_workbook(CASE_XLSX, data_only=True)
    ws = wb["Datos"]
    rows = list(ws.iter_rows(min_row=3, max_row=18, min_col=3, max_col=18,
                             values_only=True))
    return [[0 if v is None else round(float(v)) for v in r] for r in rows]


def build_case_study():
    n = 15
    D = _case_distance_matrix()
    d = {(i, j): D[i][j] for i in range(n + 1) for j in range(n + 1) if i != j}
    p = {
        1: {j: {k: _CASE_P1[j - 1][k - 1] for k in range(1, n + 1)}
            for j in range(1, n + 1)},
        2: {j: {k: _CASE_P2[j - 1][k - 1] for k in range(1, n + 1)}
            for j in range(1, n + 1)},
    }
    g = {1: {k: _CASE_G1[k - 1] for k in range(1, n + 1)},
         2: {k: 0 for k in range(1, n + 1)}}
    return {"n": n, "T": [1, 2], "d": d, "p": p, "g": g,
            "C": 5, "Q": 125, "H": 30,
            "meta": {"name": "Kandoo Guayaquil", "source": "case study"}}


# ---- 2. benchmark scenario files -----------------------------------------
def _parse_kv_blocks(text):
    """Parse David's `key = value` file into a dict; values are python literals
    (ints/floats/nested lists) possibly spanning multiple lines."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    out, i = {}, 0
    body = "\n".join(lines)
    # split on top-level `name =` assignments
    import re
    parts = re.split(r"(?m)^([A-Za-z_]\w*)\s*=\s*", body)
    # parts = ['', name1, val1, name2, val2, ...]
    it = iter(parts[1:])
    for name, val in zip(it, it):
        val = val.strip()
        # cut off trailing tokens that belong to the next assignment (already split)
        out[name] = ast.literal_eval(val)
    return out


def parse_instance_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        kv = _parse_kv_blocks(f.read())
    V = kv["V"]
    Traw = kv["T"]
    n = len(V) - 1                      # exclude depot 0
    tau = len(Traw)
    T = list(range(1, tau + 1))         # internal 1-based periods
    # distance matrix (asymmetric) over V
    Dm = kv["d"]
    d = {(i, j): Dm[i][j] for i in range(n + 1) for j in range(n + 1) if i != j}

    # g: detect orientation by shape. rows==#nodes -> g[node][day]; rows==#days -> g[day][node]
    gm = kv["g"]
    n_nodes = n + 1
    if len(gm) == n_nodes:                       # [node][day]
        g = {T[t]: {k: gm[k][t] for k in range(1, n + 1)} for t in range(tau)}
    elif len(gm) == tau:                          # [day][node]
        g = {T[t]: {k: gm[t][k] for k in range(1, n + 1)} for t in range(tau)}
    else:
        raise ValueError(f"{path}: g has {len(gm)} rows, expected {n_nodes} or {tau}")

    # p: one matrix per day, key p_day_<t> in David's 0-based day index
    p = {}
    for t_int, t_raw in zip(T, Traw):
        pm = kv[f"p_day_{t_raw}"]
        p[t_int] = {j: {k: pm[j][k] for k in range(1, n + 1)}
                    for j in range(1, n + 1)}

    inst = {"n": n, "T": T, "d": d, "p": p, "g": g,
            "C": kv["C"], "Q": kv["Q"], "H": kv["H"],
            "meta": {"name": os.path.basename(path)}}
    return inst


def all_instance_paths():
    fs = [f for f in os.listdir(INSTANCES_DIR) if f.endswith(".txt")]
    return [os.path.join(INSTANCES_DIR, f) for f in sorted(fs)]


if __name__ == "__main__":
    inst = build_case_study()
    print("CASE:", inst["meta"]["name"], "| n=", inst["n"], "C=", inst["C"],
          "Q=", inst["Q"], "H=", inst["H"], "| flow p1=",
          sum(inst["p"][1][j][k] for j in inst["p"][1] for k in inst["p"][1][j]),
          "g1=", sum(inst["g"][1].values()))

    # verify the manuscript's stated optimal routes distance == 270
    day1 = [0, 5, 4, 6, 2, 3, 11, 10, 8, 9, 15, 13, 14, 1, 12, 7, 0]
    day2 = [0, 5, 6, 4, 2, 3, 10, 11, 9, 8, 15, 13, 14, 1, 12, 7, 0]
    d = inst["d"]
    dist = lambda r: sum(d[(r[i], r[i + 1])] for i in range(len(r) - 1))
    print(f"manuscript routes: day1={dist(day1)}  day2={dist(day2)}  "
          f"total={dist(day1) + dist(day2)}  (manuscript reports 270)")

    paths = all_instance_paths()
    print(f"\n{len(paths)} benchmark instances found")
    for pth in paths[:2]:
        ii = parse_instance_txt(pth)
        print(" ", ii["meta"]["name"], "n=", ii["n"], "T=", ii["T"],
              "C=", ii["C"], "Q=", ii["Q"], "H=", ii["H"],
              "sampleD=", ii["d"][(0, 1)], ii["d"][(1, 0)])
