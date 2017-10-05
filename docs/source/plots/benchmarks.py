#!/usr/bin/python

from __future__ import print_function

import sys
import numpy as np
from time import time
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda

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


def profile(func):
    def profiled_func(*args, **kwargs):
        cuda.start_profiler()
        func(*args, **kwargs)
        cuda.stop_profiler()
        sys.exit()
    return profiled_func


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


def profile_cuvarbase_ce(t, y, dy, freqs, **kwargs):

    proc = ce.ConditionalEntropyAsyncProcess(**kwargs)
    proc.preallocate(len(t), freqs, **kwargs)
    run = profile(proc.run)

    run([(t, y, None)], freqs=freqs, **kwargs)

    return True


def time_cuvarbase_ce_run(t, y, dy, freqs, **kwargs):
    proc = ce.ConditionalEntropyAsyncProcess(**kwargs)
    proc.preallocate(len(t), freqs, **kwargs)
    run = function_timer(proc.run)

    return run([(t, y, None)], freqs=freqs, **kwargs) / len(freqs)


def time_cuvarbase_gls_run(t, y, dy, freqs, **kwargs):
    proc = ce.LombScargleAsyncProcess(**kwargs)
    proc.preallocate(len(t), freqs=freqs, **kwargs)
    run = function_timer(proc.run)

    return run([(t, y, None)], freqs=freqs, **kwargs) / len(freqs)


def time_cuvarbase_bls(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                       memory=None, pre_transfer=False, transit=False,
                       use_fast=True, **kwargs):

    kw = copy.deepcopy(_eebls_defaults)
    kw.update(kwargs)
    kw['use_fast'] = use_fast

    if memory is None and not transit:
        memory = bls.BLSMemory.fromdata(t, y, dy, freqs=freqs,
                                        transfer=pre_transfer,
                                        qmin=qmin, qmax=qmax)

    if not transit and use_fast:
        return eebls_gpu_fast(t, y, dy, freqs, memory=memory,
                              qmin=qmin, qmax=qmax,
                              transfer_to_device=(not pre_transfer),
                              **kw) / len(freqs)
    if not transit:
        return eebls_gpu(t, y, dy, freqs, qmin=qmin, qmax=qmax,
                         **kw) / len(freqs)

    qvals = kwargs.get('qvals', None)
    if freqs is None:
        freqs, qvals = bls.transit_autofreq(t, **kw)
    elif qvals is None:
        qvals = bls.q_transit(freqs, **kw)

    return eebls_transit_gpu(t, y, dy,
                             freqs=freqs,
                             qvals=qvals,
                             **kw) / len(freqs)


def time_astrobase_bls(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                       **kwargs):

    nfreqs = len(freqs)
    minfreq = min(freqs)
    stepsize = freqs[1] - freqs[0]
    nphasebins = int(np.ceil(1./qmin))

    args = (t, y)
    args += (nfreqs, minfreq, stepsize, nphasebins, qmin, qmax)
    return astrobase_bls(*args)


def time_astropy_gls(t, y, dy, freqs, qmin=1e-2, qmax=0.5,
                     **kwargs):

    ls = AstropyLombScargle(t, y, dy)

    df = freqs[1] - freqs[0]
    T = max(t) - min(t)
    spp = int(round(1./(df * T)))

    kw = dict(minimum_frequency=minimum_frequency,
              maximum_frequency=maximum_frequency,
              samples_per_peak=spp)

    run = function_timer(ls.autopower)
    return run(**kw) / len(freqs)


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

n0 = 500
ndatas = np.floor(np.logspace(1, 4.5, num=8)).astype(np.int)
qmins = np.logspace(-3, -1, num=8)
nblocks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25,
           50, 100, 200, 500, 1000, 2000, 5000]
freq_batch_sizes = [1, 5, 10, 50, 100, 500, 1000,
                    2000, 5000, 10000, 20000, 50000]

t, y, dy = data(max(ndatas), baseline=10. * 365)
freqs_t, qvals_t = bls.transit_autofreq(t, fmin=0.01, **_eebls_defaults)
t0, y0, dy0 = subset_data(t, y, dy, n0)

qmin = min(qvals_t)
qmax = max(qvals_t)
freqs = get_freqs(baseline=(max(t) - min(t)), samples_per_peak=4, fmin=0.01)
freqs_small = freqs[:5000]
print(qmin, qmax, len(freqs_t), len(freqs))
# profile_cuvarbase_ce(t0, y0, dy0, freqs=freqs, use_fast=True, force_nblocks=200) 


algos = {
    'BLS': {
        'BLS: cuvarbase (0.2.0)': lambda T, Y, DY, FREQS=freqs,
        force_nblocks=10000, **kwargs:
        time_cuvarbase_bls(T, Y, DY, FREQS, use_fast=True,
                           force_nblocks=force_nblocks, **kwargs),

        'BLS: cuvarbase (0.2.0) -- transit': lambda T, Y, DY, FREQS=freqs,
        force_nblocks=None, freq_batch_size=20, **kwargs:
        time_cuvarbase_bls(T, Y, DY, None, use_fast=True,
                           freq_batch_size=freq_batch_size,
                           force_nblocks=force_nblocks,
                           transit=True,
                           **kwargs),

        #'BLS: cuvarbase (0.1.9)': lambda T, Y, DY, FREQS=freqs, **kwargs:
        #time_cuvarbase_bls(T, Y, DY, FREQS, use_fast=False, **kwargs),

        'BLS: astrobase': lambda T, Y, DY, FREQS=freqs, **kwargs:
        time_astrobase_bls(T, Y, DY, FREQS, **kwargs),
    },
    'CE': {
        #'CE: cuvarbase (0.1.9) 25-2-10-1': lambda T, Y, DY, FREQS=freqs,
        #use_fast=False, phase_bins=25, phase_overlap=2, mag_bins=10,
        #mag_overlap=1, use_double=False, **kwargs:
        #time_cuvarbase_ce_run(T, Y, DY, FREQS, use_fast=use_fast, **kwargs),

        'CE: cuvarbase (0.2.0) 25-2-10-1': lambda T, Y, DY, FREQS=freqs,
        use_fast=True, phase_bins=25, phase_overlap=2, mag_bins=10,
        mag_overlap=1, use_double=False, **kwargs:
        time_cuvarbase_ce_run(T, Y, DY, FREQS, use_fast=use_fast, **kwargs)
    }
    'GLS': {
        'GLS: cuvarbase': lambda T, Y, DY, FREQS=freqs, **kwargs:
        time_cuvarbase_gls(T, Y, DY, FREQS, **kwargs)
        'GLS: astropy': lambda T, Y, DY, FREQS=freqs, **kwargs:
        time_astropy_gls(T, Y, DY, FREQS, **kwargs)
    }
}


tasks_nblocks = {'all': {name: algos[algo][method]
                         for algo, method in
                         [('BLS', 'BLS: cuvarbase (0.2.0)'),
                          ('CE', 'CE: cuvarbase (0.2.0) '
                                 '25-2-10-1')]}}

tasks_freq_batch_size = deepcopy.copy(tasks_nblocks)
name = 'BLS: cuvarbase (0.2.0) -- transit'
tasks_freq_batch_size['all'][name] = algos['BLS'][name]


tasks_qmin = {'all': {name: algos[algo][method]
                      for algo, method in
                      [('BLS', 'BLS: cuvarbase (0.2.0)'),
                      # ('BLS', BLS: cuvarbase (0.1.9)'),
                       ('BLS', 'BLS: astrobase')]}}


def nblock_group_func(func, nblock):
    return func(t0, y0, dy0, freqs, force_nblocks=nblock)


def ndata_group_func(func, ndata):
    T, Y, DY = subset_data(t, y, dy, ndata)
    return func(T, Y, DY, freqs)


def freq_batch_size_group_func(func, fbs):
    return func(t0, y0, dy0, freqs, force_nblocks=None,
                max_nblocks=max(freq_batch_sizes),
                freq_batch_size=fbs)


def qmin_group_func(func, qmin):
    return func(t0, y0, dy0, freqs_small, qmin=qmin)

groups = {
    'N observations': (algos, ndata_group_func, ndatas),
    'Grid size': (tasks_nblocks, nblock_group_func, nblocks),
    'Frequencies per kernel call': (tasks_nblocks,
                                    freq_batch_size_group_func,
                                    freq_batch_sizes),
    'qmin': (tasks_qmin, qmin_group_func, qmins),
}

all_methods = {}
for algo in algos:
    for method in algos[algo]

dev = pycuda.autoinit.device
attrs = dev.get_attributes()
device_name = dev.name()

print(" using: ", device_name)
#print(len(freqs))
#for attr in attrs.keys():
#    print("{attr}: {value}".format(attr=attr, value=attrs[attr]))

group_times = {}
for var in groups.keys():
    algs, func, x = groups[var]
    group_times[var] = {}

    print("="*len(var))
    print(var)
    print("="*len(var))
    for algo in algs.keys():
        print("  ", "-"*len(algo))
        print("  ", algo)
        print("  ", '-'*len(algo))
        group_times[var][algo] = time_group(algs[algo], func, x)


for var in group_times:
    for algo in group_times[var]:
        times = group_times[var][algo]

        f, ax = plt.subplots()
        for taskname in sorted(list(times.keys())):
            values, dts = zip(*times[taskname])
            ax.plot(values, dts, label=taskname)

        f.suptitle(device_name)
        ax.set_xlabel(group)
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel('Wall time / frequency')
        device_name.replace(' ', '_')
        group.replace(' ', '_')
        fname = '{dev}-{var}-{algo}.png'
        fname = fname.format(dev=device_name.replace(' ', '_'),
                             group=group.replace(' ', '_'),
                             algo=algo)

        f.savefig(fname)

    # plt.show()
