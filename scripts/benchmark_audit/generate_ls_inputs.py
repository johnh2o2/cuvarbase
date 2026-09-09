#!/usr/bin/env python3
"""Generate once with the modern environment; all backends load the same bytes."""
from pathlib import Path
import numpy as np
from common import LS_CONFIGS, ls_input


def main():
    out = Path(__file__).resolve().parent / 'inputs'
    out.mkdir(exist_ok=True)
    for cfg, shared in [(x, False) for x in LS_CONFIGS] + [('tess', True)]:
        lcs, freqs, _ = ls_input(cfg, 32, shared)
        path = out / f'ls_{cfg}_{"shared" if shared else "distinct"}.npz'
        np.savez_compressed(path, freqs=freqs,
                            t=np.stack([x[0] for x in lcs]),
                            y=np.stack([x[1] for x in lcs]),
                            dy=np.stack([x[2] for x in lcs]))
        print(path, flush=True)


if __name__ == '__main__':
    main()
