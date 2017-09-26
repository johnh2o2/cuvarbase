import cuvarbase.bls as bls
import numpy as np
import matplotlib.pyplot as plt


def phase(t, freq, phi0=0.):
    phi = (t * freq - phi0)
    phi -= np.floor(phi)

    return phi


def transit_model(t, freq, y0=0.0, delta=1., q=0.01, phi0=0.5):
    phi = phase(t, freq, phi0=phi0)
    transit = phi < q

    y = y0 * np.ones_like(t)
    y[transit] -= delta
    return y


def data(ndata=100, baseline=1, freq=10, sigma=1., **kwargs):
    t = baseline * np.sort(np.random.rand(ndata))
    y = transit_model(t, freq, **kwargs)
    dy = sigma * np.ones_like(t)

    y += dy * np.random.randn(len(t))

    return t, y, dy


def plot_bls_model(ax, y0, delta, q, phi0, **kwargs):
    phi_plot = np.linspace(0, 1, 50./q)
    y_plot = transit_model(phi_plot, 1., y0=y0,
                           delta=delta, q=q, phi0=phi0)

    ax.plot(phi_plot, y_plot, **kwargs)


def plot_bls_sol(ax, t, y, dy, freq, q, phi0, **kwargs):
    w = np.power(dy, -2)
    w /= sum(w)

    phi = phase(t, freq, phi0=phi0)
    transit = phi < q

    def ybar(mask):
        return np.dot(w[mask], y[mask]) / sum(w[mask])

    y0 = ybar(~transit)
    delta = y0 - ybar(transit)

    ax.scatter((phi[~transit] + phi0) % 1.0, y[~transit],
               c='k', s=1, alpha=0.5)
    ax.scatter((phi[transit] + phi0) % 1.0, y[transit],
               c='r', s=1, alpha=0.5)
    plot_bls_model(ax, y0, delta, q, phi0, **kwargs)

    ax.set_xlim(0, 1)
    ax.set_xlabel('$\phi$ ($f = %.3f$)' % (freq))
    ax.set_ylabel('$y$')

# set the transit parameters
transit_kwargs = dict(freq=0.1,
                      q=0.1,
                      y0=10.,
                      sigma=0.002,
                      delta=0.05,
                      phi0=0.5)

# generate data with a transit
t, y, dy = data(ndata=300,
                baseline=365.,
                **transit_kwargs)

# set up search parameters
search_params = dict(qmin=1e-2,
                     qmax=0.5,

                     # The logarithmic spacing of q
                     dlogq=0.1,

                     # Number of overlapping phase bins
                     # to use for finding the best phi0
                     noverlap=3)

# derive baseline from the data for consistency
baseline = max(t) - min(t)

# df ~ qmin / baseline
df = search_params['qmin'] / baseline
fmin = 2. / baseline
fmax = 2.

nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)

bls_power, sols = bls.eebls_gpu(t, y, dy, freqs,
                                **search_params)

# best BLS fit
q_best, phi0_best = sols[np.argmax(bls_power)]
f_best = freqs[np.argmax(bls_power)]

# Plot results
f, (ax_bls, ax_true, ax_best) = plt.subplots(1, 3, figsize=(9, 3))

# Periodogram
ax_bls.plot(freqs, bls_power)
ax_bls.axvline(transit_kwargs['freq'],
               ls=':', color='k', label="$f_0$")
ax_bls.axvline(f_best, ls=':', color='r',
               label='BLS $f_{\\rm best}$')
ax_bls.set_xlabel('freq.')
ax_bls.set_ylabel('BLS power')

# True solution
plot_bls_sol(ax_true, t, y, dy,
             transit_kwargs['freq'],
             transit_kwargs['q'],
             transit_kwargs['phi0'])

# Best-fit solution
plot_bls_sol(ax_best, t, y, dy,
             f_best, q_best, phi0_best)


ax_true.set_title("True parameters")
ax_best.set_title("Best BLS parameters")

f.tight_layout()
plt.show()
