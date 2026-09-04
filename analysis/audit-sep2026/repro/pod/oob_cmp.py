import numpy as np, sys
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
fa, pa, qa = a; fb, pb, qb = b
assert np.array_equal(fa, fb)
d = np.abs(pa - pb); nd = (d > 1e-6).sum()
print("nfreq=%d  #powers differing (>1e-6): %d  max|diff|=%.3e  #q differing: %d  first differing idx: %s  corr=%.6f" % (len(fa), nd, d.max(), (qa != qb).sum(), np.nonzero(d > 1e-6)[0][:5], np.corrcoef(pa, pb)[0, 1]))
print("plain  max=%.4f at P=%.4f | padded max=%.4f at P=%.4f" % (pa.max(), 1/fa[np.argmax(pa)], pb.max(), 1/fb[np.argmax(pb)]))
