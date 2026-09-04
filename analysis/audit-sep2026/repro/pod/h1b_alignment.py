"""Root-cause checks for the h1 anomalies."""
import sys, warnings
import numpy as np
sys.path.insert(0, '/workspace/scratch')
from audit_common import make_lc, metrics, run_safely, fmt
import entry_points as E
warnings.simplefilter('ignore')
t, y, dy = make_lc()

print("## BLS/TLS fractional-epoch sensitivity: t+0.5 (no BJD) vs base, vs t+2457000.5")
for name in ['bls_fast', 'bls_std', 'sparse_gpu', 'tls_fast', 'tls_batch']:
    fn = E.ALL[name]
    base = fn(t, y, dy)
    r_half = fn(t + 0.5, y, dy)
    r_bjd = fn(t + 2457000.5, y, dy)
    r_third = fn(t + 0.3333, y, dy)
    print('%-10s t+0.5: %s' % (name, fmt(metrics(base, r_half))))
    print('%-10s t+0.3333: %s' % (name, fmt(metrics(base, r_third))))
    print('%-10s bjd+.5 vs t+0.5: %s' % (name, fmt(metrics(r_half, r_bjd))))
    if name == 'bls_fast':
        for nov in (1, 4, 8):
            b = fn(t, y, dy, noverlap=nov); h = fn(t + 0.5, y, dy, noverlap=nov)
            print('   noverlap=%d t+0.5: %s' % (nov, fmt(metrics(b, h))))

print("\n## NFFT magnitude at BJD: float32 vs use_double, and with epoch pre-subtracted")
from cuvarbase.cunfft import NFFTAsyncProcess
for dbl in (False, True):
    proc = NFFTAsyncProcess(use_double=dbl)
    g0 = np.abs(proc.run([(t, y, 512)])[0]); proc.finish()
    g1 = np.abs(proc.run([(t + 2457000.5, y, 512)])[0]); proc.finish()
    g2 = np.abs(proc.run([(t + 1000.5, y, 512)])[0]); proc.finish()
    print('use_double=%s  |ghat| t vs t+2457000.5: %s' % (dbl, fmt(metrics(g0, g1))))
    print('use_double=%s  |ghat| t vs t+1000.5: %s' % (dbl, fmt(metrics(g0, g2))))
    # reference: exact adjoint DFT magnitudes, absolute-t convention
    T = t.max() - t.min()
    k = np.arange(512)
    ref = np.abs(np.array([np.sum(y * np.exp(2j * np.pi * (kk / T) * t)) for kk in k]))
    print('use_double=%s  |ghat|(t) vs exact DFT: %s' % (dbl, fmt(metrics(ref[:256], g0[:256]))))

print("\n## NUFFT-LRT at BJD: root cause (float32 cast of t) -- pre-subtracted epoch and use_double")
from cuvarbase.nufft_lrt import NUFFTLRTAsyncProcess
P = E.LRT_PERIODS; D = E.LRT_DUR
p32 = NUFFTLRTAsyncProcess()
b32 = p32.run(t, y, P, D).ravel()
bjd32 = p32.run(t + 2457000.5, y, P, D).ravel()
pre32 = p32.run((t + 2457000.5) - 2457000.0, y, P, D).ravel()
print('float32: t vs t+2457000.5      : %s' % fmt(metrics(b32, bjd32)))
print('float32: t vs (t+2457000.5)-2457000 (host pre-subtract): %s' % fmt(metrics(b32, pre32)))
p64 = NUFFTLRTAsyncProcess(use_double=True)
b64 = p64.run(t, y, P, D).ravel()
bjd64 = p64.run(t + 2457000.5, y, P, D).ravel()
print('use_double: t vs t+2457000.5   : %s' % fmt(metrics(b64, bjd64)))
print('float32 vs double at t         : %s' % fmt(metrics(b32, b64)))
# template correctness at BJD in float32
tt = (t + 2457000.5).astype(np.float32)
tmpl32 = p32._generate_template(tt, np.float32(3.3), np.float32(0.0), np.float32(0.165), 1.0)
tmpl64 = p64._generate_template(t + 2457000.5, 3.3, 0.0, 0.165, 1.0)
print('template in-transit count float32=%d float64=%d, agreement=%d/%d' % (
    int((tmpl32 < 0).sum()), int((tmpl64 < 0).sum()), int(((tmpl32 < 0) == (tmpl64 < 0)).sum()), len(t)))
print('float32(t+2457000.5) spacing (days):', float(np.spacing(np.float32(2457000.5))))
