"""PDM GPU-vs-CPU benchmark + correctness check (punchlist C1, issue #33).

Runs on a GPU machine. Compares cuvarbase's GPU PDM (PDMAsyncProcess)
against the CPU reference (pdm2_cpu) for (a) correctness -- the GPU and
CPU theta spectra must agree and recover an injected period -- and (b)
throughput across an (ndata x nfreq) grid. Writes a JSON report.

    python scripts/benchmark_pdm.py --tests-only        # correctness only
    python scripts/benchmark_pdm.py --output out.json   # + timing
"""
import argparse
import json
import time

import numpy as np

from cuvarbase.pdm import PDMAsyncProcess, pdm2_cpu
from cuvarbase.utils import weights


def make_lc(ndata, baseline, period, depth=0.1, noise=0.01, seed=42):
    rng = np.random.RandomState(seed)
    t = np.sort(baseline * rng.rand(ndata))
    y = 1.0 + depth * np.sin(2 * np.pi * t / period)
    y += noise * rng.randn(ndata)
    dy = noise * np.ones_like(y)
    return t.astype(np.float64), y.astype(np.float64), dy.astype(np.float64)


def gpu_pdm(proc, t, y, dy, freqs, kind='binned_linterp', nbins=10):
    res = proc.run([(t, y, dy)], freqs=[freqs.astype(np.float32)],
                   kind=kind, nbins=nbins)
    proc.finish()
    return np.asarray(res[0][1], dtype=np.float64)


def test_correctness():
    # The cuvarbase PDM kernels (and pdm2_cpu) return 1 - var/var_tot,
    # which PEAKS at the true period (maximize convention, like the
    # release gate's argmax) — NOT the classic minimize-theta PDM
    # statistic. Correctness = GPU matches CPU (correlation + same
    # argmax) AND the argmax recovers the injected period.
    print("=" * 60)
    print("PDM correctness: GPU PDM matches CPU reference (pdm2_cpu)")
    print("=" * 60)
    proc = PDMAsyncProcess()
    all_pass = True
    for ndata, baseline, period in [(300, 100.0, 2.5),
                                    (1000, 180.0, 5.0),
                                    (3000, 365.0, 10.0)]:
        t, y, dy = make_lc(ndata, baseline, period)
        w = weights(dy)
        fmin, fmax = 1.0 / (period * 2), 1.0 / (period / 2)
        freqs = np.linspace(fmin, fmax, 2000)

        gpu = gpu_pdm(proc, t, y, dy, freqs, nbins=10)
        cpu = np.asarray(pdm2_cpu(t, y, w, freqs, nbins=10, linterp=True),
                         dtype=np.float64)

        corr = np.corrcoef(gpu, cpu)[0, 1]
        same_argmax = int(np.argmax(gpu)) == int(np.argmax(cpu))
        f_best = freqs[np.argmax(gpu)]
        df = freqs[1] - freqs[0]
        recovers = abs(f_best - 1.0 / period) < 5 * df
        ok = corr > 0.999 and same_argmax and recovers
        all_pass = all_pass and ok
        print("  ndata=%-5d P=%4.1fd  corr=%.6f  argmax_match=%s  "
              "f_best=%.5f f_inj=%.5f recovers=%s  %s"
              % (ndata, period, corr, same_argmax, f_best, 1.0 / period,
                 recovers, "PASS" if ok else "FAIL"))
    print("  Overall:", "ALL PASS" if all_pass else "SOME FAILED")
    return all_pass


def benchmark(stamp):
    proc = PDMAsyncProcess()
    rows = []
    # Grid kept modest: the CPU reference (pure-Python pdm2_cpu) is the
    # slow side, so large nfreq*ndata cells dominate wall-clock. These
    # sizes still show the GPU speedup and write a representative JSON.
    for ndata in (1000, 5000):
        for nfreq in (2000, 10000):
            t, y, dy = make_lc(ndata, 365.0, 5.0)
            w = weights(dy)
            freqs = np.linspace(0.01, 2.0, nfreq)

            # warm up / compile
            gpu_pdm(proc, t, y, dy, freqs)
            tg = time.time()
            gpu_pdm(proc, t, y, dy, freqs)
            gpu_t = time.time() - tg

            tc = time.time()
            pdm2_cpu(t, y, w, freqs, nbins=10, linterp=True)
            cpu_t = time.time() - tc

            rows.append(dict(ndata=ndata, nfreq=nfreq,
                             gpu_s=gpu_t, cpu_s=cpu_t,
                             speedup=cpu_t / gpu_t if gpu_t else None))
            print("  ndata=%-6d nfreq=%-6d  gpu=%7.4fs cpu=%7.4fs  %6.1fx"
                  % (ndata, nfreq, gpu_t, cpu_t, rows[-1]['speedup']))
    return dict(timestamp=stamp, grid=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests-only', action='store_true')
    ap.add_argument('--output', default='benchmarks/results/benchmark_pdm.json')
    args = ap.parse_args()

    # Retain the CUDA context (lazy since v1.0) before querying the device.
    from cuvarbase.base import ensure_context
    ensure_context()
    import pycuda.driver as cuda
    try:
        dev = cuda.Context.get_device()
    except Exception:
        dev = None

    passed = test_correctness()
    out = dict(device=str(dev.name()) if dev else 'unknown',
               correctness_pass=bool(passed))

    if not args.tests_only:
        print("\nThroughput (GPU PDM vs pdm2_cpu):")
        out['benchmark'] = benchmark(time.strftime('%Y-%m-%dT%H:%M:%S'))
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print("\nwrote %s" % args.output)

    print("\nPDM tests:", "PASS" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
