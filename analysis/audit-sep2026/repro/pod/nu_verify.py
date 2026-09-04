import numpy as np, warnings
warnings.filterwarnings('ignore')
import pycuda.autoprimaryctx
from cuvarbase.lombscargle import LombScargleAsyncProcess, get_k0, check_k0
from astropy.timeseries import LombScargle

rng = np.random.RandomState(1)
N, T = 600, 100.0
t = np.sort(rng.uniform(0, T, N)); y = 1 + 0.01*np.sin(2*np.pi*t/0.7) + 0.005*rng.randn(N)
dy = 0.005*np.ones(N)*rng.uniform(0.8, 1.2, N)
ls = LombScargle(t, y, dy)
proc = LombScargleAsyncProcess()

def gpu(f, **kw):
    r = proc.run([(t, y, dy)], freqs=[f], **kw); proc.finish()
    return r[0][0], np.copy(r[0][1])

def met(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return 'maxabs=%.2e corr=%.4f argmax_same=%s' % (np.abs(a-b).max(), np.corrcoef(a, b)[0, 1], np.argmax(a) == np.argmax(b))

def implied(f):
    df = f[1]-f[0]; k0 = get_k0(f); return df*(k0+np.arange(len(f)))

cases = {}
df = 1.0/500; k0 = 50; nf = 300
f = df*(k0+np.arange(nf))
fb = f.copy(); fb[2:] = f[2] + 3*(f[2:]-f[2]); cases['auditor: spacing x3 after 2nd pt'] = fb
# realistic: two concatenated linspace segments (fine low-f, coarse high-f)
cases['concat fine+coarse segments'] = np.concatenate([np.arange(0.1, 1.0, 0.002), np.arange(1.0, 5.0, 0.01)])
# log grid
cases['geomspace 0.1..10, nf=1000'] = np.geomspace(0.1, 10.0, 1000)
# astropy autofrequency but then user drops a chunk (mask) -> gap in middle
fa = f.copy(); cases['uniform grid with 20 freqs deleted mid-way'] = np.delete(fa, np.arange(100, 120))
# reversed (descending) grid
cases['descending uniform grid'] = f[::-1].copy()

for name, fu in cases.items():
    print('\n### %s  (len=%d, freqs[:3]=%s)' % (name, len(fu), np.array2string(fu[:3], precision=5)))
    try:
        check_k0(fu)
        print('check_k0: PASSES (k0=%d)' % get_k0(fu))
    except ValueError as e:
        print('check_k0: raises ValueError -> %s' % str(e)[:90]); continue
    a_user = ls.power(fu)
    a_impl = ls.power(implied(fu))
    for use_fft in (True, False):
        fret, g = gpu(fu, use_fft=use_fft)
        print('use_fft=%-5s returned-freqs-is-user-grid=%s | vs astropy@USER grid: %s | vs astropy@IMPLIED grid: %s'
              % (use_fft, np.array_equal(fret, fu), met(a_user, g), met(a_impl, g)))

print('\n### batched_run_const_nfreq with the concatenated grid')
fu = cases['concat fine+coarse segments']
p = proc.batched_run_const_nfreq([(t, y, dy)], freqs=fu)[0]
print('vs astropy@USER: %s | vs astropy@IMPLIED: %s' % (met(ls.power(fu), p), met(ls.power(implied(fu)), p)))

print('\n### does the concatenated grid trip any existing guard? nf=%d, implied fmax=%.3f vs user fmax=%.3f' % (len(fu), implied(fu)[-1], fu[-1]))
