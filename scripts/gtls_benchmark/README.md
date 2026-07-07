# GTLS apples-to-apples benchmark

Reproduces Figure 7 of the GTLS paper (Hu, Ge, Jin & Willis, arXiv:2607.00348) —
single-light-curve search time vs light-curve baseline — with the search held
**fair** across implementations, on one GPU. Full analysis and results:
`analysis/GTLS_COMPARISON.md`.

## Files
- `bench_core.py` — GPU-independent core: light-curve injection (batman, Keplerian
  duration), the shared Ofir period grid, and the one identical SDE re-scorer.
- `gtls_apples_bench.py` — the runner. Sweeps baselines × methods (GTLS full/skip8,
  cuvarbase TLS matched/default, cuvarbase BLS kunimoto/sensible), matching period
  grid, per-period duration window, epoch density, and injected transit; writes JSON.
- `plot_fig7.py` — merges result JSONs and renders the reproduced figure + tables.

## Requirements (GPU host)
`cupy`, `pycuda`, `scikit-cuda`, `batman-package`, `numpy<2` (numba/gtls pin),
plus **both** cuvarbase feature branches merged:
- `feature/tls-fast-survey` — the improved TLS (`tls_search_batch`);
- `feature/bls-survey-speed` — the improved BLS (`eebls_gpu_batch`).

The TLS-vs-GTLS curves run on `feature/tls-fast-survey` alone; the **BLS** curves
require `feature/bls-survey-speed` (otherwise stock BLS is timed and the numbers
will differ from the writeup). GTLS = `pip install gputls` (v0.5.1) + cupy.

## Run
```bash
python gtls_apples_bench.py \
    --baselines 200,500,1000,1500,2000,3000 \
    --methods cuv_tls_matched,cuv_tls_default,cuv_bls_kunimoto,cuv_bls_matched \
    --cuv-reps 3 --out results_cuv.json
# GTLS (slow at long baselines — its runtime scales ~N^2.5):
python gtls_apples_bench.py --baselines 200,500,1000,1500 \
    --methods gtls_full,gtls_skip8 --gtls-reps 1 --out results_gtls.json
python plot_fig7.py fig7_reproduction.png results_cuv.json results_gtls.json
```

Result JSONs from the July 2026 A5000 run are in
`benchmarks/results/gtls_comparison_jul2026/`.
