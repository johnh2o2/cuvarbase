import numpy as np
from cuvarbase.ce import ConditionalEntropyAsyncProcess
from cuvarbase.memory.ce_memory import ConditionalEntropyMemory

rng = np.random.RandomState(0)
N = 400
t = np.sort(30 * rng.rand(N)); f0 = 3.1
y = 12 + 0.3*np.cos(2*np.pi*f0*t) + 0.05*rng.randn(N)
# a couple of outliers -> balanced bins should differ strongly from uniform bins
y[:3] += 5.0
err = 0.05*np.ones_like(y)
freqs = np.linspace(2.5, 3.7, 3000)
data = [(t, y, err)]

def run(proc, **kw):
    r = proc.run(data, freqs=[freqs], **kw); proc.finish(); return np.copy(r[0][1])
def large_run(proc, **kw):
    r = proc.large_run(data, freqs=freqs, **kw); proc.finish(); return np.copy(r[0][1])
def batched(proc, **kw):
    r = proc.batched_run_const_nfreq(data, freqs=freqs, **kw); proc.finish(); return np.copy(r[0][1])

plain = run(ConditionalEntropyAsyncProcess())
print("1) ctor(balanced_magbins=True) attribute stored on proc?",
      hasattr(ConditionalEntropyAsyncProcess(balanced_magbins=True), 'balanced_magbins'))
for name, fn in [('run', run), ('large_run', large_run), ('batched_run_const_nfreq', batched)]:
    ctor = fn(ConditionalEntropyAsyncProcess(balanced_magbins=True))
    runkw = fn(ConditionalEntropyAsyncProcess(), balanced_magbins=True)
    print("2) %-24s max|ctor(balanced)-plain|=%.3e   max|runkw(balanced)-plain|=%.3e   max|ctor-runkw|=%.3e"
          % (name, np.abs(ctor-plain).max(), np.abs(runkw-plain).max(), np.abs(ctor-runkw).max()))

# memory objects created by ctor-path
p = ConditionalEntropyAsyncProcess(balanced_magbins=True)
mems = p.allocate(data, freqs=[freqs])
print("3) allocate() memory.balanced_magbins on ctor(balanced=True) proc:", mems[0].balanced_magbins)
p2 = ConditionalEntropyAsyncProcess(balanced_magbins=True)
p2.preallocate(N, freqs, nlcs=1)
print("   preallocate() memory.balanced_magbins:", p2.memory[0].balanced_magbins)

# 4) the test-suite no-op claim: test_inject_and_recover parametrizes ctor with
#    weighted=True AND balanced_magbins=True. If forwarded, ConditionalEntropyMemory raises.
try:
    ConditionalEntropyMemory(weighted=True, balanced_magbins=True)
    print("4) Memory(weighted+balanced): no error (unexpected)")
except ValueError as e:
    print("4) Memory(weighted=True, balanced_magbins=True) raises:", e)
try:
    r = run(ConditionalEntropyAsyncProcess(weighted=True, balanced_magbins=True))
    print("   ctor(weighted=True, balanced_magbins=True).run(): NO error, result finite=%s -> ctor flag not forwarded" % np.all(np.isfinite(r)))
except ValueError as e:
    print("   ctor(weighted=True, balanced_magbins=True).run() raises:", e)
try:
    r = run(ConditionalEntropyAsyncProcess(weighted=True), balanced_magbins=True)
    print("   ctor(weighted=True).run(balanced_magbins=True): NO error")
except ValueError as e:
    print("   ctor(weighted=True).run(balanced_magbins=True) raises:", e)

# 5) widen_mag_range: same story?
w_ctor = run(ConditionalEntropyAsyncProcess(weighted=True, widen_mag_range=True))
w_plain = run(ConditionalEntropyAsyncProcess(weighted=True))
w_runkw = run(ConditionalEntropyAsyncProcess(weighted=True), widen_mag_range=True)
print("5) widen_mag_range: max|ctor-plain|=%.3e  max|runkw-plain|=%.3e" % (np.abs(w_ctor-w_plain).max(), np.abs(w_runkw-w_plain).max()))

# 6) mag_overlap>0 + ctor balanced -> raises (so ctor *appears* to accept it)
try:
    ConditionalEntropyAsyncProcess(mag_overlap=1, balanced_magbins=True)
    print("6) ctor(mag_overlap=1, balanced=True): no error")
except ValueError as e:
    print("6) ctor(mag_overlap=1, balanced_magbins=True) raises:", e)
# ... but run kwarg path with mag_overlap=1 bypasses that check?
try:
    r = run(ConditionalEntropyAsyncProcess(mag_overlap=1), balanced_magbins=True)
    print("   ctor(mag_overlap=1).run(balanced_magbins=True): NO error, finite=%s" % np.all(np.isfinite(r)))
except Exception as e:
    print("   ctor(mag_overlap=1).run(balanced_magbins=True) raises:", type(e).__name__, e)
