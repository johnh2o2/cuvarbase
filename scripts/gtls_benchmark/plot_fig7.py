#!/usr/bin/env python
"""Reproduce GTLS paper (arXiv:2607.00348) Fig. 7 apples-to-apples on one GPU,
plus an SDE-parity panel. Merges any number of results_*.json files (e.g.
results_cuv.json results_gtls.json). Writes fig7_reproduction.png.

Usage: python plot_fig7.py out.png results_cuv.json results_gtls.json ...
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig7_reproduction.png"
FILES = sys.argv[2:] or ["results_cuv.json", "results_gtls.json"]

# merge: baseline -> method -> result
merged, gpu, inj = {}, "GPU", {}
for fn in FILES:
    try:
        d = json.load(open(fn))
    except Exception:
        continue
    gpu = d.get("env", {}).get("gpu", gpu)
    inj = d.get("inj", inj)
    for b, row in d.get("results", {}).items():
        merged.setdefault(b, {})
        for m, r in row.get("methods", {}).items():
            merged[b][m] = r

bl = sorted(int(b) for b in merged)

STYLE = {  # colorblind-safe; grouped by family
    "gtls_full":        ("#d55e00", "o", "-",  "GTLS  full-T0 scan  (paper Fig.7 setting)"),
    "gtls_skip8":       ("#e69f00", "s", "-",  "GTLS  skip=8  (its efficient default)"),
    "cuv_bls_kunimoto": ("#cc79a7", "P", "-",  "cuvarbase BLS  (Kunimoto qmin=2e-4, nov=3 — paper's BLS cfg)"),
    "cuv_tls_matched":  ("#0072b2", "D", "-",  "cuvarbase TLS  (matched: grid+durations+epochs to GTLS)"),
    "cuv_tls_default":  ("#009e73", "^", "-",  "cuvarbase TLS  (survey default: t0os=3, 15 dur)"),
    "cuv_bls_matched":  ("#56b4e9", "v", "-",  "cuvarbase BLS  (sensible cfg: qmin=2e-3, fused nov=2)"),
}
ORDER = ["gtls_full", "gtls_skip8", "cuv_bls_kunimoto", "cuv_tls_matched",
         "cuv_tls_default", "cuv_bls_matched"]

def series(m, key):
    xs, ys = [], []
    for b in bl:
        v = merged[str(b)].get(m, {}).get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            xs.append(b); ys.append(v)
    return np.array(xs, float), np.array(ys, float)

# published paper anchors (single-LC; GTLS/BLS on RTX 4090, TLS on 7950X CPU)
PAPER = {"gtls": [(1500, 33.3), (3000, 138.0)], "bls": [(1500, 121.1)],
         "tls_cpu": [(1500, 522.0), (3000, 3289.0)]}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 10.2),
                               gridspec_kw={"height_ratios": [2.5, 1]})

for m in ORDER:
    c, mk, ls, lab = STYLE[m]
    x, y = series(m, "time_s")
    if len(x):
        ax1.plot(x, y, marker=mk, ls=ls, color=c, lw=2, ms=7, label=lab, zorder=4)
# GTLS full compile-subtracted (search only)
xs, ys = series("gtls_full", "search_s")
if len(xs):
    ax1.plot(xs, ys, ":", color=STYLE["gtls_full"][0], lw=1.3, alpha=.7,
             label="GTLS full, search only (JIT-compile subtracted)", zorder=3)
# published paper points
px, py = zip(*PAPER["gtls"]); ax1.scatter(px, py, marker="*", s=280,
            facecolor="none", edgecolor="#d55e00", linewidths=2, zorder=6)
bx, by = zip(*PAPER["bls"]); ax1.scatter(bx, by, marker="*", s=280,
            facecolor="none", edgecolor="#cc79a7", linewidths=2, zorder=6)
tx, ty = zip(*PAPER["tls_cpu"]); ax1.plot(tx, ty, "--", color="#7f7f7f", lw=1.6,
            marker="X", ms=10, label="reference TLS (CPU 7950X, paper)", zorder=3)
ax1.scatter([], [], marker="*", s=200, facecolor="none", edgecolor="k",
            linewidths=1.5, label="★ published paper value (RTX 4090)")

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("light-curve baseline  [days]   (30-min cadence)")
ax1.set_ylabel("search time per light curve  [s]")
snr = inj.get("period"), inj.get("depth")
ax1.set_title("GTLS Fig. 7 reproduced apples-to-apples on one GPU (%s)\n"
              "identical Ofir period grid · identical per-period duration window "
              "· matched epoch density · one injected transit" % gpu, fontsize=10.5)
ax1.grid(True, which="both", alpha=.25)
ax1.legend(fontsize=7.6, loc="lower right", framealpha=.96, ncol=1)
for b in (1500, 3000):
    ax1.axvline(b, color="k", alpha=.06, lw=10)

for m in ORDER:
    c, mk, ls, lab = STYLE[m]
    x, y = series(m, "sde_identical")
    if len(x):
        ax2.plot(x, y, marker=mk, color=c, lw=1.5, ms=6)
ax2.axhline(7, color="k", ls="--", lw=1, alpha=.6)
ax2.text(bl[0], 8, "SDE=7 detection threshold", fontsize=8, alpha=.7)
ax2.set_xscale("log")
ax2.set_xlabel("light-curve baseline  [days]")
ax2.set_ylabel("SDE  (one identical\nstatistic per spectrum)")
ax2.set_title("Detection significance is identical across all methods — "
              "the speed gap is not bought with sensitivity", fontsize=10)
ax2.grid(True, which="both", alpha=.25)
# explicit x ticks (log axis otherwise only labels 10^3)
ticks = [b for b in (200, 300, 500, 1000, 1500, 2000, 3000) if bl[0] <= b <= bl[-1]]
for ax in (ax1, ax2):
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(NullFormatter())
    ax.set_xlim(bl[0] * 0.9, bl[-1] * 1.12)
fig.tight_layout()
fig.savefig(OUT, dpi=145, bbox_inches="tight")
print("wrote", OUT)

# ---- text table + speedups ----
hdr = "baseline  " + "".join("%18s" % m.replace("cuv_", "").replace("_", " ")
                             for m in ORDER)
print("\n" + hdr)
for b in bl:
    r = "%6dd " % b
    for m in ORDER:
        t = merged[str(b)].get(m, {}).get("time_s")
        r += "%18s" % (("%.3f s" % t) if isinstance(t, (int, float)) else "-")
    print(r)
print("\nSDE (identical) per baseline:")
for b in bl:
    ss = [merged[str(b)].get(m, {}).get("sde_identical") for m in ORDER]
    ss = [s for s in ss if isinstance(s, (int, float))]
    print("  %6dd: %.1f–%.1f  (spread %.1f%%)" % (
        b, min(ss), max(ss), 100 * (max(ss) - min(ss)) / np.mean(ss)))
print("\nSpeedup cuvarbase-TLS-matched vs GTLS:")
for b in bl:
    M = merged[str(b)]
    cm = M.get("cuv_tls_matched", {}).get("time_s")
    gf = M.get("gtls_full", {}).get("time_s")
    gs = M.get("gtls_skip8", {}).get("time_s")
    if cm and (gf or gs):
        print("  %6dd: vs GTLS-full %-7s  vs GTLS-skip8 %-7s" % (
            b, ("%.0fx" % (gf / cm)) if gf else "-",
            ("%.0fx" % (gs / cm)) if gs else "-"))
