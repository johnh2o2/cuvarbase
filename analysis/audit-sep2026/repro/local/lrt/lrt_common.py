import warnings, numpy as np
warnings.filterwarnings('ignore', message='cuvarbase.nufft_lrt is EXPERIMENTAL')

def make_times(rng, baseline=90.0, n=600):
    nights = np.arange(int(baseline))
    keep = rng.rand(len(nights)) > 0.35
    nights = nights[keep]
    per_night = max(1, int(round(n / max(len(nights), 1))))
    t = (nights[:, None] + 0.25 * rng.rand(len(nights), per_night)).ravel()
    return np.sort(t[:n])

def ou_noise(rng, t, sigma_red, tau):
    x = np.zeros(len(t)); x[0] = sigma_red * rng.randn()
    for i in range(1, len(t)):
        a = np.exp(-(t[i] - t[i-1]) / tau)
        x[i] = x[i-1] * a + sigma_red * np.sqrt(1 - a*a) * rng.randn()
    return x

def box(t, period, epoch, duration, depth):
    phase = np.fmod(t - epoch, period) / period
    phase[phase < 0] += 1; phase[phase > 0.5] -= 1
    y = np.zeros_like(t); y[np.abs(phase) <= duration/(2*period)] = -depth
    return y

def adjoint_dft(t, y, nf):
    t = np.asarray(t, np.float64); y = np.asarray(y, np.float64)
    x = t / (t.max() - t.min()); k = np.arange(nf)
    return np.exp(2j*np.pi*np.outer(k, x)) @ y
