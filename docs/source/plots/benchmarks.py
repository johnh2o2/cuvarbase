from __future__ import print_function

import numpy as np
from time import time
import copy
import matplotlib.pyplot as plt

import cuvarbase.bls as bls
import cuvarbase.ce as ce
import cuvarbase.lombscargle as ls
from astrobase.periodbase.kbls import _bls_runner as astrobase_bls
from astropy.stats.lombscargle import LombScargle as AstropyLombScargle
from tqdm import tqdm


def get_freqs(baseline=5 * 365., fmin=None,
              fmax=(24 * 60.) / 30., samples_per_peak=5):

    df = 1. / baseline / samples_per_peak
    if fmin is None:
        fmin = 2./baseline

    nf = int(np.ceil((fmax - fmin) / df))

    return fmin + df * np.arange(nf)


def data(ndata, baseline=5 * 365.):
    t = baseline * np.sort(np.random.rand(ndata))
    y = np.cos(2 * np.pi * t)
    dy = 0.1 * np.ones_like(t)

    y += dy * np.random.randn(len(t))

    return t, y, dy


def function_timer(func, nreps=3):
    def timed_func(*args, **kwargs):
        dts = []
        for n in range(nreps):
            t0 = time()
            func(*args, **kwargs)
            dt = time() - t0
            dts.append(dt)
        return min(dts)

    return timed_func


eebls_gpu = function_timer(bls.eebls_gpu)
eebls_transit_gpu = function_timer(bls.eebls_transit_gpu)
eebls_gpu_fast = function_timer(bls.eebls_gpu_fast)
astrobase_bls = function_timer(astrobase_bls)

_eebls_defaults = dict(qmin_fac=0.5, qmax_fac=2.0, dlogq=0.25,
                       samples_per_peak=4, noverlap=2)


def time_cuvarbase_ce_run(t, y, dy, freqs, **kwargs):
    proc = ce.ConditionalEntropyAsyncProcess(**kwargs)
    proc.preallocate(len(t), freqs, **kwargs)
    run = function_timer(proc.run)
    return run([(t, y, None)], freqs=freqs, **kwargs)


def time_cuvarbase_bls(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                       memory=None, pre_transfer=False, transit=False,
                       use_fast=True, **kwargs):

    kw = copy.deepcopy(_eebls_defaults)
    kw.update(kwargs)

    if memory is None:
        memory = bls.BLSMemory.fromdata(t, y, dy, freqs=freqs,
                                        transfer=pre_transfer,
                                        qmin=qmin, qmax=qmax)

    if not transit and use_fast:
        return eebls_gpu_fast(t, y, dy, freqs,
                              transfer_to_device=(not pre_transfer),
                              **kw)
    if not transit:
        return eebls_gpu(t, y, dy, freqs,
                         **kw)
    return eebls_transit_gpu(t, y, dy, freqs=freqs, **kw)


def time_astrobase_bls(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                       **kwargs):

    nfreqs = len(freqs)
    minfreq = min(freqs)
    stepsize = freqs[1] - freqs[0]
    nphasebins = int(np.ceil(1./qmin))

    args = (t, y)
    args += (nfreqs, minfreq, stepsize, nphasebins, qmin, qmax)
    return astrobase_bls(*args)


def subset_data(t, y, dy, ndata):
    inds = np.arange(1, len(t) - 1)
    np.random.shuffle(inds)

    subinds = np.concatenate(([0], np.argsort(t[inds[:ndata-2]]),
                             [len(t) - 1]))
    return (arr[subinds] for arr in (t, y, dy))


def time_group(task_dict, group_func, values):
    times = {}
    for name in task_dict.keys():
        print(name)
        dts = []
        for v in tqdm(values):
            dts.append((v, group_func(task_dict[name], v)))
        times[name] = dts
    return times

n0 = 1000
ndatas = np.floor(np.logspace(1, 4, num=8)).astype(np.int)
nblocks = [1, 5, 10, 50, 100, 500, 1000]
freq_batch_sizes = [1, 10, 100, None]

t, y, dy = data(max(ndatas), baseline=365.)
t0, y0, dy0 = subset_data(t, y, dy, n0)

freqs = get_freqs(baseline=(max(t) - min(t)))


tasks = {
    'BLS: cuvarbase (0.2.0)': lambda T, Y, DY, FREQS=freqs,
    force_nblocks=1000, **kwargs:
    time_cuvarbase_bls(T, Y, DY, FREQS, use_fast=True,
                       force_nblocks=force_nblocks, **kwargs),

    'BLS: cuvarbase (0.1.9)': lambda T, Y, DY, FREQS=freqs, **kwargs:
    time_cuvarbase_bls(T, Y, DY, FREQS, use_fast=False, **kwargs),

    'BLS: astrobase': lambda T, Y, DY, FREQS=freqs, **kwargs:
    time_astrobase_bls(T, Y, DY, FREQS, **kwargs),

    'CE: cuvarbase (0.1.9) 10-1-5-0': lambda T, Y, DY, FREQS=freqs,
    use_fast=False, phase_bins=10, phase_overlap=1, mag_bins=5,
    mag_overlap=0, **kwargs:
    time_cuvarbase_ce_run(T, Y, DY, FREQS, use_fast=use_fast, **kwargs),

    'CE: cuvarbase (0.2.0) 10-1-5-0': lambda T, Y, DY, FREQS=freqs,
    use_fast=True, phase_bins=10, phase_overlap=1, mag_bins=5,
    mag_overlap=0, **kwargs:
    time_cuvarbase_ce_run(T, Y, DY, FREQS, use_fast=use_fast, **kwargs)

}

tasks_nblocks = {name: tasks[name] for name in ['BLS: cuvarbase (0.2.0)',
                                                'CE: cuvarbase (0.2.0) '
                                                '10-1-5-0']}


def nblock_group_func(func, nblock):
    return func(t0, y0, dy0, freqs, force_nblocks=nblock)


def ndata_group_func(func, ndata):
    T, Y, DY = subset_data(t, y, dy, ndata)
    return func(T, Y, DY, freqs)


def freq_batch_size_group_func(func, fbs):
    return func(t0, y0, dy0, freqs, freq_batch_size=fbs)


groups = {
    'Ndata': (tasks, ndata_group_func, ndatas),
    'NBLOCKS': (tasks_nblocks, nblock_group_func, nblocks),
    'freq batch size': (tasks_nblocks, freq_batch_size_group_func,
                        freq_batch_sizes)
}

group_times = {group: time_group(*groups[group]) for group in groups.keys()}

for group in group_times:
    times = group_times[group]

    f, ax = plt.subplots()
    for taskname in times.keys():
        values, dts = zip(*times[taskname])
        ax.plot(values, dts, label=taskname)

    ax.set_xlabel(group)
    ax.legend(loc='best')
    ax.set_yscale('log')
    plt.show(block=True)
