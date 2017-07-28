#!/usr/bin/python

#from cuvarbase.wavelet import Wavelet
import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt
import pycuda.autoinit
from astropy.stats import LombScargle
from cuvarbase.bls import bls_test, bls_test_bst
from cuvarbase.pdm import PDMSpectrogram, PDMAsyncProcess
from cuvarbase.lombscargle import LombScargleSpectrogram, lomb_scargle,\
                                  LombScargleAsyncProcess
from cuvarbase.cunfft import time_shift, NFFTAsyncProcess
from cuvarbase.utils import weights
from numpy.testing import assert_allclose
#from nfft import nfft_adjoint as nfft_adjoint_cpu
#from nfft.utils import nfft_matrix
#from nfft.kernels import KERNELS

import pycuda.driver
seed = 100
rand = np.random.RandomState(seed)

def data(T=10*365, N=3000, y0=10., sigma=0.1, dtau=100., tau=0.5, period=0.77):
	t = T * np.sort(rand.rand(N))
	y = y0 + np.exp(-0.5 * ((t - tau * T) / dtau)**2) * np.cos(2 * np.pi * t / period)
	y += sigma * np.random.randn(N)
	dy = sigma * np.ones_like(y)

	return t, y, dy


def hat_data(fname):
	data = np.loadtxt(fname, skiprows=1)
	t, y, err = [ data[:, i] for i in range(3)]
	w = np.power(err, -2)
	w /= sum(w)
	ybar = np.dot(w, y)
	print(len(data))
	return data[:,0] - min(data[:,0]), data[:,1] - ybar, data[:,2]

def wavelet(dt, sigma, freq):
	a = abs(2 * np.pi * sigma * freq * dt)
	if a < 1:
		return 0
	return 1 - 3 * a**2 + 2 * a**3

def total_weights(dt, y, w, sigma, freq):
	ww = map(lambda DT : wavelet(DT, sigma, freq), dt)
	return np.dot(w, ww)

def test_wavelet():
	from glob import glob
	from time import time
	import sys
	hatids = glob("HAT*txt")
	hatid = hatids[8].replace('_TF3.txt', '')
	print("reading data")
	t0 = time()
	#t, y, dy = hat_data('%s_TF3.txt'%(hatid))
	t, y, dy = data()
	print(time() - t0)

	w = np.power(dy, -2)
	w /= sum(w)

	print("initiating wavelet")
	t0 = time()
	samples_per_peak=5
	resolution_enhancement = 15
	nsplits = 500
	window_length = resolution_enhancement * (max(t) - min(t)) / nsplits
	nbins=10
	wl = Wavelet(t, y, dy, samples_per_peak = samples_per_peak,
				pmin=0.05, pmax=15., sigma=0.05,
				resolution_enhancement=resolution_enhancement,
				precision=5)#.run()
	print(time() - t0)
	t0 = time()
	specgram = wl.run()
	print time() - t0
	taus = wl.taus
	freqs = wl.freqs

	df = 1./(samples_per_peak * (max(t) - min(t)))
	nf = (max(freqs) - min(freqs)) / df

	#i0 = int(min(freqs) / df)
	freqs_lin =  df * (np.arange(nf) + 1)

	#pdmspec = PDMSpectrogram(t, y, w, nbins=nbins, window_length=window_length)
	pdmspec = LombScargleSpectrogram(t, y, w, window_length=window_length)

	lsp = pdmspec.localized_spectrum(freqs=freqs_lin)
	lsp_cpu = LombScargle(t, y, w).power(freqs_lin)

	plt.plot(freqs_lin, lsp)
	plt.plot(freqs_lin, lsp_cpu, color='r', alpha=0.2)
	plt.show()
	sys.exit()

	sgram_times, sgram_freqs, pdm_sgram = pdmspec.spectrogram(nsplits=nsplits, freqs=freqs_lin)

	full_pdm = pdmspec.localized_spectrum(freqs=sgram_freqs)
	all_taus_and_freqs = []
	for tauvals, f in zip(taus, freqs):
		all_taus_and_freqs.extend(zip(tauvals, (f * np.ones_like(tauvals)).tolist()))##

	all_taus, all_freqs = zip(*all_taus_and_freqs)

	#avg_power = np.zeros(len(freqs))
	#tot_wt = np.zeros(len(freqs))

	#sind = 0
	#print("computing weighted avg spectrum")
	#for i, frq in enumerate(wl.freqs):
	#	#print("%d / %d"%(i, len(wl.freqs)))
	#	for j, tau in enumerate(wl.taus[i]):
	#		totw = total_weights(t - tau, y, w, wl.sigma, frq)
	#		avg_power[i] += specgram[sind] * totw
	#		tot_wt[i] += totw
	#		sind += 1

	#tot_wt += 1E-9

	#avg_power /= tot_wt


	taus_plt = np.linspace(min(t), max(t), max([ len(taus_) for taus_ in taus ]))
	freqs_plt = np.linspace(min(freqs), max(freqs), 10 * len(freqs))
	ls = LombScargle(t, y, dy).power(freqs_plt)


	#freqs_plt = np.log10(freqs_plt)




	norm = lambda arr : (np.array(arr) - min(arr)) / (max(arr) - min(arr))

	spec_plt = mlab.griddata(norm(all_taus), norm(all_freqs), specgram,
							 norm(taus_plt), norm(freqs_plt), interp='linear')

	stimes, sfreqs = np.meshgrid(sgram_times, sgram_freqs)

	f = plt.figure(figsize=(15, 8))

	width = 0.6
	left = 0.1

	bottom = 0.1
	height = 0.72
	top = 0.94
	right = 0.93
	cbar_width = 0.02
	wspace = 0.00
	hspace = 0.04
	pdg_width = right - left - width

	ax_sgram = f.add_axes([left + pdg_width,
							bottom ,
							width,
							0.5* height - 0.5 * hspace])
	ax_sgram_cbar  = f.add_axes([left + width + pdg_width,
									bottom ,
									cbar_width,
									0.5 * height - 0.5 * hspace ])
	ax_pdg_pdm  = f.add_axes([left,
							bottom,
							pdg_width - wspace,
							0.5 * height - 0.5 * hspace])

	ax_wvlet = f.add_axes([left + pdg_width,
							bottom+ 0.5 * height + 0.5 * hspace,
							width,
							0.5* height - 0.5 * hspace])

	ax_wvlet_cbar  = f.add_axes([left + width + pdg_width,
							bottom+ 0.5 * height + 0.5 * hspace,
							cbar_width,
							0.5 * height - 0.5 * hspace])

	ax_pdg_ls   = f.add_axes([left,
							bottom + 0.5 * height + 0.5 * hspace,
							pdg_width - wspace,
							0.5 * height -0.5 * hspace])

	ax_lc    = f.add_axes([left + pdg_width,
							bottom + height + 0.5 * hspace,
							width,
							top - height - bottom - 0.5 * hspace])




	cplot = ax_wvlet.pcolormesh(taus_plt, freqs_plt, spec_plt, vmin=0.001, vmax=1)

	cplot_sgram = ax_sgram.pcolormesh(stimes, sfreqs, pdm_sgram.T, vmin=0.001, vmax=1)
	f.colorbar(cplot, cax=ax_wvlet_cbar)
	f.colorbar(cplot_sgram, cax=ax_sgram_cbar)
	f.text(left, top, '%s\n%d observations'%(hatid, len(t)), ha='left', va='top')

	ax_wvlet.axis([min(t), max(t), min(freqs_plt), max(freqs_plt)])
	ax_sgram.axis([min(t), max(t), min(freqs_lin), max(freqs_lin)])


	ax_lc.scatter(t, y, s=1, c='k', alpha=1., marker=',')
	ax_lc.set_xlim(*ax_wvlet.get_xlim())

	#ax_pdg.plot( freqs, avg_power, color='k')

	ax_pdg_ls.plot( ls, freqs_plt, color='r')
	#ax_pdg.set_xlim(0, 1)
	ax_pdg_ls.set_ylim(*ax_wvlet.get_ylim())

	ax_pdg_pdm.plot(full_pdm, sgram_freqs, color='g')
	ax_pdg_pdm.set_ylim(*ax_sgram.get_ylim())

	ax_sgram.set_xlabel('$t - t_0$ (days)')


	pmax = max([ ax.get_xlim()[1] for ax in [ax_pdg_pdm, ax_pdg_ls]])
	for ax in [ax_pdg_ls, ax_pdg_pdm]:


		#ax.set_yticklabels([])
		ax.set_xlim(0, pmax)
		ax.invert_xaxis()
		for s in ['top', 'right']:
			ax.spines[s].set_visible(False)

		ax.set_ylabel('Freq. (1/days)')

	for ax in [ ax_wvlet, ax_sgram ]:
		for s in ['top', 'left', 'right']:
			ax.spines[s].set_visible(False)
		#ax.set_yticklabels([])
		for ylabel in ax.get_yticklabels():
			ylabel.set_visible(False)

		ax.get_xticklabels()[0].set_visible(False)


	ax_pdg_ls.text(0.05, 0.95, 'Lomb-Scargle (floating mean)',
					ha='left', va='top', transform=ax_pdg_ls.transAxes)
	ax_pdg_pdm.text(0.05, 0.95, 'PDM (%d bins, lin. interp)'%(nbins),
					ha='left', va='top', transform=ax_pdg_pdm.transAxes)

	#ax_pdg_ls.yaxis.set_ticks_position('none')
	#ax_pdg_pdm.yaxis.set_ticks_position('none')
	#ax_wvlet.xaxis.set_ticks_position('none')
	#ax_pdg.set_ylim(*ax_lc.get_ylim())

	for xlabel in ax_lc.get_xticklabels():
		xlabel.set_visible(False)
	for xlabel in ax_wvlet.get_xticklabels():
		xlabel.set_visible(False)


	ax_wvlet.set_yscale('log')
	ax_sgram.set_yscale('log')
	ax_pdg_pdm.set_yscale('log')
	ax_pdg_ls.set_yscale('log')
	plt.show()

from nfft import nfft_adjoint

def simple_gpu_nfft(t, y, N, sigma=2, m=8, block_size=128, **kwargs):
    proc = NFFTAsyncProcess()
    results = proc.run([(t, y, N)], sigma=sigma, m=m, block_size=block_size, **kwargs)
    proc.finish()
    return results[0]


def get_cpu_grid(t, y, N, sigma=2, m=8):
    kernel = KERNELS.get('gaussian', 'gaussian')
    mat = nfft_matrix(t, int(N * sigma), m, sigma, kernel, truncated=True)
    return mat.T.dot(y)

def gpu_nfft(t, y, N, sigma=2, m=8):
	proc = NFFTAsyncProcess()
	results = proc.run([(t, y, N)], sigma=sigma, m=m)
	proc.finish()
	return results[0]

def test_fast_gridding(dat, sigma=2, m=8, block_size=160):
    t_, y, N = dat
    t, phi0 = time_shift(t_)

    gpu_grid = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size,
                        just_return_gridded_data=True, fast_grid=True)

    # get CPU grid
    cpu_grid = get_cpu_grid(t, y, N, sigma=sigma, m=m)
    avg_grid = np.median(np.absolute(cpu_grid[np.absolute(cpu_grid) > 0]))

    f, ax = plt.subplots()
    #ax.plot((gpu_grid - cpu_grid) / avg_grid, color='k', alpha=0.4)
    ax.plot(gpu_grid, color='r', alpha=0.5)
    ax.plot(cpu_grid, color='k', alpha=0.5)
    #ax.plot((gpu_grid[inds] - cpu_grid[inds]) / cpu_grid[inds])
    #ax.plot(cpu_grid, color='k', alpha=0.4)
    plt.show()
    #assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)

def test_slow_gridding(dat, sigma=2, m=8, block_size=160):
    t_, y, N = dat
    t, phi0 = time_shift(t_)

    gpu_grid = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size,
                        just_return_gridded_data=True, fast_grid=False)

    # get CPU grid
    cpu_grid = get_cpu_grid(t, y, N, sigma=sigma, m=m)
    avg_grid = np.median(np.absolute(cpu_grid[np.absolute(cpu_grid) > 0]))
    #print(cpu_grid, gpu_grid, )
    f, ax = plt.subplots()

    inds = np.arange(len(cpu_grid))[np.absolute(cpu_grid) > 0]


    #ax.plot(gpu_grid, color='r', alpha=0.5)
    #ax.plot(cpu_grid, color='k', alpha=0.5)
    ax.plot((gpu_grid[inds] - cpu_grid[inds]) / cpu_grid[inds])
    #ax.plot(cpu_grid, color='k', alpha=0.4)
    plt.show()
    #assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)

def pdm_data(seed=100, sigma=0.1, ndata=250, freq=10):

    rand = np.random.RandomState(seed)

    t = np.sort(rand.rand(ndata))
    y = np.cos(2 * np.pi * freq * t)

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err

def test_pdm_plot():
	import matplotlib.pyplot as plt
	kind = 'binned_linterp'
	nbins = 10
	seed = 100
	nfreqs = 10000
	ndata = 500

	t, y, err = pdm_data(seed=seed, ndata=ndata)


	w = weights(err)
	freqs = np.linspace(0,100, nfreqs)
	freqs += 0.5 * (freqs[1] - freqs[0])

	# pow_cpu = pdm2_cpu(t, y, w, freqs, linterp=(kind == 'binned_linterp'), nbins=nbins)

	pdm_proc = PDMAsyncProcess()
	results = pdm_proc.run([(t, y, w, freqs)], kind=kind, nbins=nbins)
	pdm_proc.finish()

	pow_gpu = results[0]
	plt.plot(freqs, pow_gpu)
	plt.show()

def str_fmt_value(num, fmt='%.3f', nmin=3):
	n = int(np.log10(num))
	a = num / (10 ** n)

	if abs(n) > 3:
		st = "%s\\times 10^{%d}"%(fmt, n)
		return st%(a)
	return fmt%(num)

def test_lomb_scargle_recovery():
	import matplotlib.pyplot as plt

	ndata = 100
	snrs0 = np.linspace(2, 7, 30)

	sigmas = (np.sqrt(ndata) / snrs0)[::-1]

	nyquist_freq = 0.5 * ndata

	freq = 3 * nyquist_freq
	ntest = 100
	rrates_cpu = np.zeros_like(sigmas)
	rrates_gpu = np.zeros_like(sigmas)
	err_max = 5E-2

	proc = LombScargleAsyncProcess(use_double=False, sigma=2)

	seeds = np.arange(ntest)

	for i, sigma in enumerate(sigmas):
		datas = []
		errs = []
		for test in range(ntest):
			t, y, err = pdm_data(freq=freq, ndata=ndata, sigma=sigma, seed=seeds[test])
			w = np.power(err, -1)
			w /= sum(w)

			datas.append((t, y, w))
			errs.append(err)

		freqs_gpu, pows_gpu = proc.run(datas, nyquist_factor=10, samples_per_peak=10)
		proc.finish()

		for ((t, y, w), err, fgpu, pgpu) in zip(datas, errs, freqs_gpu, pows_gpu):
			assert(not any(np.isnan(pgpu)))

			pcpu = LombScargle(t, y, err).power(fgpu)

			fmax_gpu = fgpu[np.argmax(pgpu)]
			fmax_cpu = fgpu[np.argmax(pcpu)]

			ferr_gpu = abs(fmax_gpu - freq) / freq

			ferr_cpu = abs(fmax_cpu - freq) / freq

			if ferr_gpu < err_max:
				rrates_gpu[i] += 1./ntest

			if ferr_cpu < err_max:
				rrates_cpu[i] += 1./ntest

	drr = 1./np.sqrt(ntest)

	f, ax = plt.subplots()
	snrs = (np.sqrt(ndata) / sigmas)[::-1]

	rrates_cpu = rrates_cpu[::-1]
	rrates_gpu = rrates_gpu[::-1]

	ax.plot(snrs, rrates_cpu, color='k', lw=2)
	ax.fill_between(snrs, rrates_cpu - drr, rrates_cpu + drr, color='k', alpha=0.4)

	ax.plot(snrs, rrates_gpu, color='r', lw=2)
	ax.fill_between(snrs, rrates_gpu - drr, rrates_gpu + drr, color='r', alpha=0.4)

	ax.set_xlim(min(snrs), max(snrs))
	#ax.set_xscale('log')
	ax.set_xlabel('S/N = $\\sqrt{N_{\\rm obs}} (A / \\sigma)$')
	ax.set_ylabel('Recovery rate ($\\Delta f / f < %s)$'%(str_fmt_value(err_max)))
	ax.set_title('$f = %0.3f f_{\\rm nyq}$; $f_{nyq} = N_{\\rm obs} / 2T$'%(freq / nyquist_freq))
	plt.show()

def test_cunfft():

	t, y, dy = data()
	N = 2 * len(t)
#	t[0] = 0.

	t, phi0 = time_shift(t)
	fcpu = nfft_adjoint_cpu(t, y, N, sigma=2, m=8)
	fgpu = gpu_nfft(t, y, N, sigma=2, m=8)

	plt.plot(np.real(fgpu - fcpu) / np.mean(np.absolute(np.real(fcpu))), color='k', alpha=0.5)
	#plt.plot(, color='b', alpha=0.5)
	#plt.show()
	plt.plot(np.imag(fgpu - fcpu) / np.mean(np.absolute(np.imag(fcpu))), color='r', alpha=0.5)
	#plt.plot(np.imag(fgpu, color='b', alpha=0.5)
	plt.show()

def phase_shift_(phi, phi0):
	dphi = phi - phi0
	if dphi < 0:
		dphi += 1

	return min([ dphi, 1 - dphi])

def phase_shift(phi, phi0):
	if not hasattr(phi, '__iter__'):
		return phase_shift_(phi, phi0)
	else:
		return np.array(map(lambda p : phase_shift_(p, phi0), phi ))


def transit_model(phi0, q, delta, q1=0.):
	def model(t, freq, q=q, phi0=phi0, delta=delta):

		dphi = (t * freq - phi0) % 1.0 - 0.5
		if not hasattr(t, '__iter__'):
			return delta if np.absolute(dphi) < 0.5 * q else 0
		y = np.zeros(len(t))
		y[np.absolute(dphi) < 0.5 * q] -= delta

		return y
	return model


def plot_model(ax, t, y, dy, freq, model):
	phi_plot = np.linspace(0, 1, 600)

	ax.scatter((t * freq)%1.0, y, s=1, c='k')
	ax.plot(phi_plot, model(phi_plot/freq, freq), color='r', lw=2)
	ax.set_xlim(0, 1)
	ax.set_xlabel('phase')
	ax.set_ylabel('mag')


def periodogram(y, yhat, dy):
	w = np.power(dy, -2)
	w /= sum(w)

	ybar = np.dot(w, y)

	chi20 = np.dot(w, np.power(y - ybar, 2))
	chi2m = np.dot(w, np.power(y - yhat, 2))

	return 1. - chi2m / chi20

def plot_bls_sol(ax, t, y, dy, freq, q, phi0):
	w = np.power(dy, -2)
	w /= sum(w)

	phi = (t * freq - phi0) % 1.0 - 0.5

	ybar = np.dot(w, y)

	transit = np.absolute(phi) < 0.5 * q

	ybart = np.dot(y[transit], w[transit]) / sum(w[transit])
	ybar0 = np.dot(y[~transit], w[~transit]) / sum(w[~transit])

	delta = ybar0 - ybart

	model = transit_model(phi0, q, delta)

	yhat = model(t, freq)

	yhat += ybar - np.mean(yhat)

	plot_model(ax, t, y, dy, freq, model)

	return periodogram(y, yhat, dy)

profiling = False
from time import time
import matplotlib.pyplot as plt
def test_bls(seed = 100, freq=20, ndata=10000, delta=0.5,
	         fmin=1, fmax=100, nfreqs=10000, q=.1,
	         phi0=0.23, sigma=0.1, qmin=0.01, qmax=0.5,
	         noverlap=5, alpha=1.1, batch_size=50, nstreams=5):
	rand = np.random.RandomState(seed)
	phi_model = np.linspace(0, 1, 600)

	model = transit_model(phi0, q, delta)

	t = np.sort(rand.rand(ndata))
	phi = (t * freq) % 1.0

	freqs = np.linspace(fmin, fmax, nfreqs)
	dy = sigma * np.ones(ndata)

	w = np.power(dy, -2)
	w /= sum(w)

	noise = rand.randn(ndata)
	y = model(t, freq) + dy * noise

	if profiling:
		pycuda.driver.start_profiler()
	t0 = time()
	bls, sols = bls_test_bst(t, y, dy, freqs, nstreams=nstreams, batch_size=batch_size,
			plot_status=False, noverlap=noverlap, alpha=alpha, qmin=qmin, qmax=qmax)
	dt = time() - t0
	print(dt, dt/nfreqs)

	best_q, best_phi = sols[np.argmax(bls)]
	best_freq = freqs[np.argmax(bls)]

	print(freq, best_freq)
	print(q, best_q)
	print(phi0, best_phi + 0.5 * q)
	if profiling:
		pycuda.driver.stop_profiler()
	if not profiling:
		f, (axbls, axlc, axsol) = plt.subplots(1, 3, figsize=(15, 5))
		axbls.plot(freqs, bls, color='k')
		axbls.axvline(freq, color='k', ls=':')

		plot_model(axlc, t, y, dy, freq, model)


		pdg = plot_bls_sol(axsol, t, y, dy, best_freq, best_q, best_phi)

		axbls.axhline(pdg, color='r', ls='--')

		axbls.axhline(periodogram(y, model(phi/freq, freq), dy), color='b', ls='--')

		print(pdg)
		plt.show()

if __name__ == '__main__':
	#test_pdm_plot()

	test_lomb_scargle_recovery()
	# test_bls()
	#sys.exit()
	"""
	t, y, dy = data()
	N = 2 * len(t)
	t[0] = 0.

	#test_cunfft()
	test_slow_gridding((t, y, N))
	test_fast_gridding((t, y, N))

	sys.exit()
	autopower_kwargs = dict(samples_per_peak=5, nyquist_factor=5)
	freqs_cpu, power_cpu = LombScargle(t, y, dy, center_data=False,
								fit_mean=True).autopower(**autopower_kwargs)
	freqs_g, power_g = lomb_scargle(t, y, dy, use_cpu_nfft=False, **autopower_kwargs)

	plt.plot(freqs_cpu, power_cpu, color='k', alpha=0.5)
	plt.plot(freqs_g, power_g)

	plt.show()
	#test_wavelet()
	"""