#!/usr/bin/env python3
"""
Compare two parity dumps produced by bench_bls_survey.py --parity.

For each survey and each array key (fast, fast_bjd, batch) reports:
  - max |delta|, rms delta
  - Pearson correlation
  - argmax (peak) equality and peak frequency
  - bit-identical flag

Exit code 1 if any comparison fails the gate:
  correlation > 0.999 AND identical peak location (or bit-identical).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / 'results' / 'bls_survey_speed_jul2026'


def compare(tag_a, tag_b, surveys):
    pdir = RESULTS_DIR / 'raw' / 'parity'
    ok = True
    for name in surveys:
        sname = name.replace('-', '')
        fa = pdir / f'parity_{sname}_{tag_a}.npz'
        fb = pdir / f'parity_{sname}_{tag_b}.npz'
        if not fa.exists() or not fb.exists():
            print(f"[{name}] MISSING: {fa if not fa.exists() else fb}")
            ok = False
            continue
        da, db = np.load(fa), np.load(fb)
        freqs = da['freqs']
        for key in ('fast', 'fast_bjd', 'batch'):
            if key not in da or key not in db:
                continue
            pa, pb = da[key].astype(np.float64), db[key].astype(np.float64)
            bit = bool(np.array_equal(da[key], db[key]))
            corr = float(np.corrcoef(pa, pb)[0, 1])
            maxd = float(np.max(np.abs(pa - pb)))
            rmsd = float(np.sqrt(np.mean((pa - pb) ** 2)))
            ia, ib = int(np.argmax(pa)), int(np.argmax(pb))
            peak_same = ia == ib
            gate = bit or (corr > 0.999 and peak_same)
            ok = ok and gate
            status = 'BITEQ' if bit else ('PASS ' if gate else 'FAIL ')
            print(f"[{name:8s}] {key:8s} {status} corr={corr:.7f} "
                  f"max|d|={maxd:.3e} rms={rmsd:.3e} "
                  f"peak {freqs[ia]:.6f} vs {freqs[ib]:.6f}"
                  f"{'' if peak_same else '  <-- PEAK MOVED'}")
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tag_a')
    ap.add_argument('tag_b')
    ap.add_argument('--surveys', nargs='+',
                    default=['ZTF', 'HAT-Net', 'TESS', 'Kepler'])
    args = ap.parse_args()
    sys.exit(0 if compare(args.tag_a, args.tag_b, args.surveys) else 1)
