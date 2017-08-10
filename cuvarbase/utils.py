import numpy as np
import pkg_resources

def weights(err):
    """ generate observation weights from uncertainties """
    w = np.power(err, -2)
    return w/sum(w)

def find_kernel(name):
    return pkg_resources.resource_filename('cuvarbase', 'kernels/%s.cu'%(name))

def _module_reader(fname, cpp_defs=None):
    txt = open(fname, 'r').read()

    if cpp_defs is None:
        return txt


    preamble = ['#define {key} {value}'.format(key=key,
                                               value=('' if value is None
                                                      else value))
                for key, value in cpp_defs.iteritems()]
    txt = txt.replace('//{CPP_DEFS}', '\n'.join(preamble))

    return txt


def tophat_window(t, t0, d):
    w_window = np.zeros_like(t)
    w_window[np.absolute(t - t0) < d] += 1.
    return w_window / max(w_window)

def gaussian_window(t, t0, d):
    w_window = np.exp( - 0.5 * np.power(t - t0, 2) / (d * d))
    return w_window / (1. if len(w_window) == 0 else max(w_window))

def autofrequency(t, nyquist_factor=5, samples_per_peak=5,
                      minimum_frequency=None, maximum_frequency = None, **kwargs):
    """
    Determine a suitable frequency grid for data.

    Note that this assumes the peak width is driven by the observational
    baseline, which is generally a good assumption when the baseline is
    much larger than the oscillation period.
    If you are searching for periods longer than the baseline of your
    observations, this may not perform well.

    Even with a large baseline, be aware that the maximum frequency
    returned is based on the concept of "average Nyquist frequency", which
    may not be useful for irregularly-sampled data. The maximum frequency
    can be adjusted via the nyquist_factor argument, or through the
    maximum_frequency argument.

    Parameters
    ----------
    samples_per_peak : float (optional, default=5)
        The approximate number of desired samples across the typical peak
    nyquist_factor : float (optional, default=5)
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float (optional)
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float (optional)
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.

    Returns
    -------
    frequency : ndarray or Quantity
        The heuristically-determined optimal frequency bin
    """
    baseline = max(t) - min(t)
    n_samples = len(t)

    df = 1. / (baseline * samples_per_peak)

    nf0 = 1
    if minimum_frequency is not None:
        nf0 = max([nf0, int(minimum_frequency / df)])

    if maximum_frequency is not None:
        Nf = int(maximum_frequency / df) - nf0
    else:
        Nf = int(0.5 * samples_per_peak * nyquist_factor * n_samples)

    return df * (nf0 + np.arange(Nf))


def dphase(dt, freq):
    dph = dt * freq - np.floor(dt * freq)
    dph_final = dph if dph < 0.5 else 1 - dph
    return dph_final

def get_autofreqs(t, **kwargs):
    autofreqs_kwargs = {var : value for var, value in kwargs.iteritems() \
                        if var in ['minimum_frequency', 'maximum_frequency',
                                   'nyquist_factor', 'samples_per_peak']}
    return autofrequency(t, **autofreqs_kwargs)
