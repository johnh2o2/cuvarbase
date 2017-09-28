import matplotlib.pyplot as plt
import numpy as np
import cuvarbase.bls as bls


def transit_model(phi0, q, delta, q1=0.):
    def model(t, freq, q=q, phi0=phi0, delta=delta):

        phi = t * freq - phi0
        phi -= np.floor(phi)

        if not hasattr(t, '__iter__'):
            return -delta if np.absolute(phi) < q else 0
        y = np.zeros(len(t))
        y[np.absolute(phi) < q] -= delta

        return y
    return model


def plot_bls_sol(t, y, dy, freq, q, phi0):

    w = np.power(dy, -2)
    w /= sum(w)

    phi_plot = np.linspace(0, 1, 50./q)

    phi = (t * freq)
    phi -= np.floor(phi)

    dphi = phi - phi0 - np.floor(phi - phi0)
    mask = dphi < q

    ybt = np.dot(w[mask], y[mask]) / sum(w[mask])
    yb0 = np.dot(w[~mask], y[~mask]) / sum(w[~mask])

    delta = yb0 - ybt

    model = transit_model(phi0, q, delta)

    ym = model(phi_plot, 1.) + yb0

    f, ax = plt.subplots()

    ax.scatter(phi[~mask], y[~mask], c='k', s=1, alpha=0.4)
    ax.scatter(phi[mask], y[mask], c='g', s=1, alpha=0.8)
    ax.plot(phi_plot, ym, color='r')
    ax.axvline(phi0, color='k', ls=':')
    # ax.axvline(phi0 + q, color='k', ls=':')

    ax.axis('off')

    ax.annotate('$\\delta$', xy=(phi0 - 0.03, -0.5 * delta), xytext=(-5, 0),
                textcoords='offset points', ha='right', va='center',
                fontsize=20)

    ax.plot([phi0 - 0.03, phi0 - 0.03], [-delta, -0.03 * delta], ls='--',
            color='k')

    ax.plot([phi0, phi0 + q], [-1.03 * delta, -1.03 * delta], ls='--',
            color='k')
    ax.annotate('$q$', xy=(phi0 + 0.5 * q, -1.03 * delta), xytext=(0, -5),
                textcoords='offset points', ha='center', va='top',
                fontsize=20, transform=ax.transData)

    ax.annotate('$\\phi_0$', xy=(phi0, 0), xytext=(5, 5),
                textcoords='offset points', ha='left', va='bottom',
                fontsize=20, transform=ax.transData)

    ax.annotate('$y_0$', xy=(0.05, 0), xytext=(5, 5),
                textcoords='offset points', ha='left', va='bottom',
                fontsize=20, transform=ax.transData)
    plt.show()

model = transit_model(0.5, 0.1, 0.1)
t = np.sort(np.random.rand(200))
y = model(t, 10.)
dy = 0.01 * np.ones_like(y)

y += dy * np.random.randn(len(t))

plot_bls_sol(t, y, dy, 10., 0.1, 0.5)
