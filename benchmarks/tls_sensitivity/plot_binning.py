#!/usr/bin/env python3
"""Illustrate the actual TLS template and phase-bin approximations; no GPU needed."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from binning import bin_filter, load_tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--template-source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='Output filename stem')
    a = ap.parse_args()
    template, integral, _ = load_tables(a.template_source)
    period, epoch = 5., 2.
    duration = period * np.arcsin((1 / (period * 8.6307))**(2/3)) / np.pi
    x = np.linspace(-2.2, 2.2, 4001)
    t = epoch + x / 24
    point = np.interp((t - epoch) / (.5 * duration), np.linspace(-1, 1, len(template)),
                      template, left=0, right=0)
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    ax.plot(x, 1 - (np.abs(x) <= duration * 12).astype(float), color='#999999',
            lw=1.4, ls='--', label='Box template')
    ax.plot(x, 1 - point, color='#142b3b', lw=2.5,
            label='Transit template on individual observations')
    for bins, color in [(512, '#d47729'), (4096, '#367eaa')]:
        filt = bin_filter(np.r_[0., t], period, epoch, duration, bins, integral)[1:]
        ax.plot(x, 1 - filt, color=color, lw=1.3, alpha=.9,
                label=f'{bins:,} phase bins ({period * 1440 / bins:.3g} min each)')
    ax.set(xlabel='Hours from transit center', ylabel='Normalized brightness',
           ylim=(-.07, 1.08), xlim=(-2.2, 2.2))
    ax.set_yticks([0, .5, 1], ['Transit minimum', '', 'Out of transit'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.14)
    ax.legend(loc='lower left', fontsize=8, frameon=False, bbox_to_anchor=(0, 1.02), ncol=2)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(a.out.with_suffix('.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
