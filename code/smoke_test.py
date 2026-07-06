"""Quick correctness smoke test for model_full on a tiny synthetic instance."""
import math
import random
import model_full as M


def make_instance(n=5, tau=2, seed=0, tight=1.0):
    rng = random.Random(seed)
    coords = {0: (50, 50)}
    for k in range(1, n + 1):
        coords[k] = (rng.uniform(0, 100), rng.uniform(0, 100))
    d = {}
    for i in coords:
        for j in coords:
            if i != j:
                d[(i, j)] = round(math.dist(coords[i], coords[j]))
    T = list(range(1, tau + 1))
    p = {t: {j: {k: (rng.randint(0, 3) if j != k else 0)
                 for k in range(1, n + 1)} for j in range(1, n + 1)} for t in T}
    g = {t: {k: rng.randint(0, 3) for k in range(1, n + 1)} for t in T}
    total_demand = sum(p[1][j][k] for j in p[1] for k in p[1][j]) + sum(g[1].values())
    Q = max(10, int(total_demand / tight))   # tight<1 -> looser; tight>1 -> tighter
    return {"n": n, "T": T, "d": d, "p": p, "g": g, "C": 2, "Q": Q, "H": 40}


if __name__ == "__main__":
    inst = make_instance(n=5, tau=2, seed=1, tight=0.8)
    print("Q =", inst["Q"])
    for mu in (0, 3):
        j = M.solve_joint(inst, mu, time_limit=60)
        m = M.solve_myopic(inst, mu, time_limit=60)
        print(f"\nmu={mu}")
        print(f"  joint : status={j['status']} dist={j['total_distance']} R={j['total_R']} obj={j['objective']}")
        print(f"  myopic: dist={m['total_distance']} R={m['total_R']}")
        print("  joint validate:", M.validate(inst, j) or "OK")
        print("  myopic validate:", M.validate(inst, m) or "OK")
