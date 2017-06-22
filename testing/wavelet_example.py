
from cuvarbase.wavelet import Wavelet
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pycuda.autoinit
from astropy.stats import LombScargle
from cuvarbase.pdm import PDMSpectrogram
from cuvarbase.lombscargle import LombScargleSpectrogram, lomb_scargle
from cuvarbase.cunfft import time_shift, NFFTAsyncProcess
from numpy.testing import assert_allclose
from nfft import nfft_adjoint as nfft_adjoint_cpu
from nfft.utils import nfft_matrix
from nfft.kernels import KERNELS
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

    f, ax = plt.subplots()
    ax.plot(gpu_grid, color='r', alpha=0.4)
    ax.plot(cpu_grid, color='k', alpha=0.4)
    plt.show()
    #assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)

def test_slow_gridding(dat, sigma=2, m=8, block_size=160): 
    t_, y, N = dat
    t, phi0 = time_shift(t_)

    gpu_grid = simple_gpu_nfft(t, y, N, sigma=sigma, m=m, block_size=block_size,
                        just_return_gridded_data=True, fast_grid=False)

    # get CPU grid
    cpu_grid = get_cpu_grid(t, y, N, sigma=sigma, m=m)
    f, ax = plt.subplots()
    ax.plot(gpu_grid, color='r', alpha=0.4)
    ax.plot(cpu_grid, color='k', alpha=0.4)
    plt.show()
    #assert_allclose(gpu_grid, cpu_grid, rtol=1E-3)
        



def test_cunfft():

	t, y, dy = data()
	N = 2 * len(t)
#	t[0] = 0.

	t, phi0 = time_shift(t)
	fcpu = nfft_adjoint(t, y, N, sigma=2, m=8)
	fgpu = gpu_nfft(t, y, N, sigma=2, m=8)

	plt.plot(np.real(fcpu), color='k', alpha=0.5)
	plt.plot(np.real(fgpu), color='b', alpha=0.5)
	plt.show()
	plt.plot(np.imag(fcpu), color='k', alpha=0.5)
	plt.plot(np.imag(fgpu), color='b', alpha=0.5)
	plt.show()


if __name__ == '__main__':
	t, y, dy = data()
	N = 2 * len(t)
	#test_cunfft()
	t[0] = 0.

	autopower_kwargs = dict(samples_per_peak=5, nyquist_factor=5)
	freqs_cpu, power_cpu = LombScargle(t, y, dy, center_data=False, 
								fit_mean=True).autopower(**autopower_kwargs)
	freqs_g, power_g = lomb_scargle(t, y, dy, **autopower_kwargs)
	
	plt.plot(freqs_cpu, power_cpu, color='k', alpha=0.5)
	plt.plot(freqs_g, power_g)

	plt.show()
	#test_wavelet()