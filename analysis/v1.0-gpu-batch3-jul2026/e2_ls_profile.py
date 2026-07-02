"""E2 diagnosis: why is LombScargleAsyncProcess.batched_run_const_nfreq
slower with batch_size > 1 (multi-stream) than with batch_size=1?

Stage-times the batched path for batch_size in (1, 2, 4, 8) on the same
32 light curves / shared frequency grid:
  - alloc:    LombScargleMemory construction + allocate (per call)
  - setdata:  host-side weighting + pinned-buffer fill + H2D enqueue
  - launch:   lomb_scargle_async (kernel/NFFT enqueue per LC)
  - finish:   context sync barrier per batch (GPU wait)

Prints a table + JSON_RESULT.
"""
import json
import time

import numpy as np

import cuvarbase.lombscargle as ls_mod
from cuvarbase.lombscargle import LombScargleAsyncProcess
from cuvarbase.memory import LombScargleMemory


def make_data(n_lcs=32, ndata=3000, seed=5):
    rand = np.random.RandomState(seed)
    data = []
    for i in range(n_lcs):
        t = np.sort(365.0 * rand.rand(ndata))
        y = np.cos(2 * np.pi * 3.0 * t) + 0.1 * rand.randn(ndata)
        data.append((t, y, 0.1 * np.ones(ndata)))
    return data


def freq_grid(nf=100000, samples_per_peak=3, baseline=365.0, fmin=0.05):
    df = 1.0 / (samples_per_peak * baseline)
    k0 = max(1, int(round(fmin / df)))
    return df * (k0 + np.arange(nf))


class StageTimers:
    def __init__(self):
        self.t = {'setdata': 0.0, 'launch': 0.0, 'finish': 0.0,
                  'alloc': 0.0}

    def patch(self, proc):
        timers = self.t

        orig_setdata = LombScargleMemory.setdata
        orig_allocate = LombScargleMemory.allocate
        orig_async = ls_mod.lomb_scargle_async
        orig_finish = proc.finish

        def timed_setdata(self, **kw):
            t0 = time.time()
            r = orig_setdata(self, **kw)
            timers['setdata'] += time.time() - t0
            return r

        def timed_allocate(self, **kw):
            t0 = time.time()
            r = orig_allocate(self, **kw)
            timers['alloc'] += time.time() - t0
            return r

        def timed_async(*a, **kw):
            t0 = time.time()
            r = orig_async(*a, **kw)
            timers['launch'] += time.time() - t0
            return r

        def timed_finish():
            t0 = time.time()
            r = orig_finish()
            timers['finish'] += time.time() - t0
            return r

        LombScargleMemory.setdata = timed_setdata
        LombScargleMemory.allocate = timed_allocate
        ls_mod.lomb_scargle_async = timed_async
        proc.finish = timed_finish
        return (orig_setdata, orig_allocate, orig_async, orig_finish)

    @staticmethod
    def unpatch(proc, originals):
        (LombScargleMemory.setdata, LombScargleMemory.allocate,
         ls_mod.lomb_scargle_async, _of) = originals
        proc.finish = _of


def main():
    data = make_data()
    freqs = freq_grid()
    n_lcs = len(data)

    proc = LombScargleAsyncProcess(use_double=False, sigma=4)
    # warm-up: compile + one small run
    _ = proc.batched_run_const_nfreq(data[:2], freqs=freqs[:1000])
    proc.finish()

    rows = []
    for bs in (1, 2, 4, 8):
        timers = StageTimers()
        originals = timers.patch(proc)
        t0 = time.time()
        res = proc.batched_run_const_nfreq(data, batch_size=bs,
                                           freqs=freqs)
        total = time.time() - t0
        StageTimers.unpatch(proc, originals)

        assert len(res) == n_lcs and all(np.all(np.isfinite(p))
                                         for _, p in res)
        row = dict(batch_size=bs, total_s=total,
                   ms_per_lc=1e3 * total / n_lcs, **timers.t)
        other = total - sum(timers.t.values())
        row['other_s'] = other
        rows.append(row)
        print("batch_size=%d total=%.3fs (%.1f ms/LC)  alloc=%.3f "
              "setdata=%.3f launch=%.3f finish=%.3f other=%.3f"
              % (bs, total, row['ms_per_lc'], timers.t['alloc'],
                 timers.t['setdata'], timers.t['launch'],
                 timers.t['finish'], other), flush=True)

    print("JSON_RESULT: " + json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
