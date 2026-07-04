"""Float32-faithful CPU simulation of the eebls_gpu (standard) pipeline,
used to reproduce and root-cause the test_standard failures attila reported
with the t += 4.5 fixture shift on PR #65.

Simulates:
  bin_and_phase_fold_bst_multifreq  (float32 fold + hierarchical binning)
  binned_bls_bst                    (bls_value per overlapped bin)
  reduction_max + store_best_sols   (argmax -> (q, phi_sub) solution)
then applies the PR #65 phi round-trip:
  eebls_gpu:  phi_orig = (phi_sub + epoch*freq) % 1.0     (float64)
  single_bls: phi_sub' = (phi_orig - epoch*freq) % 1.0    (float64)
and evaluates single_bls exactly as bls.py (pr65-latest) does.
"""
import numpy as np

f32 = np.float32


# ---------------- test fixture (test_bls.py) ----------------

def transit_model(phi0, q, delta):
    def model(t, freq):
        phi = t * freq - phi0
        phi -= np.floor(phi)
        y = np.zeros(len(t))
        y[np.absolute(phi) < q] -= delta
        return y
    return model


def data(seed=100, sigma=0.1, ybar=12., snr=10, ndata=200, freq=10.,
         q=0.01, phi0=None, baseline=1., tshift=0.0):
    rand = np.random.RandomState(seed)
    if phi0 is None:
        phi0 = rand.rand()
    delta = snr * sigma / np.sqrt(ndata * q * (1 - q))
    model = transit_model(phi0, q, delta)
    t = baseline * np.sort(rand.rand(ndata))
    y = model(t, freq) + sigma * rand.randn(len(t))
    y += ybar - np.mean(y)
    err = sigma * np.ones_like(y)
    t = t + tshift            # attila's fixture change: t = t + 4.5
    return t, y, err


# ---------------- utils.subtract_epoch ----------------

def subtract_epoch(t):
    t = np.asarray(t, dtype=np.float64)
    epoch = np.floor(np.min(t))
    return t - epoch, epoch


# ---------------- single_bls (pr65-latest version) ----------------

def single_bls_pr65(t, y, dy, freq, q, phi0,
                    ignore_negative_delta_sols=False):
    t, epoch = subtract_epoch(t)
    phi0 = (phi0 - (epoch * freq)) % 1.0          # pr65 conversion
    phi = t.astype(np.float32) * np.float32(freq)
    phi -= np.float32(phi0)
    phi -= np.floor(phi)
    mask = phi < np.float32(q)

    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))
    ybar = np.dot(w, np.asarray(y).astype(np.float32))
    YY = np.dot(w, np.power(np.asarray(y).astype(np.float32) - ybar, 2))
    W = np.sum(w[mask])
    YW = np.dot(w[mask], np.asarray(y).astype(np.float32)[mask]) - ybar * W
    if YW > 0 and ignore_negative_delta_sols:
        return 0
    if W < 1e-9 or W > 1 - 1e-4:
        return 0
    return (YW ** 2) / (W * (1 - W)) / YY


def single_bls_old(t, y, dy, freq, q, phi0,
                   ignore_negative_delta_sols=False):
    """v1.0-fixes version: phi0 already in subtracted timescale."""
    t_sub = subtract_epoch(t)[0]
    phi = t_sub.astype(np.float32) * np.float32(freq)
    phi -= np.float32(phi0)
    phi -= np.floor(phi)
    mask = phi < np.float32(q)
    w = np.power(dy, -2)
    w /= np.sum(w.astype(np.float32))
    ybar = np.dot(w, np.asarray(y).astype(np.float32))
    YY = np.dot(w, np.power(np.asarray(y).astype(np.float32) - ybar, 2))
    W = np.sum(w[mask])
    YW = np.dot(w[mask], np.asarray(y).astype(np.float32)[mask]) - ybar * W
    if YW > 0 and ignore_negative_delta_sols:
        return 0
    if W < 1e-9 or W > 1 - 1e-4:
        return 0
    return (YW ** 2) / (W * (1 - W)) / YY


# ---------------- GPU pipeline simulation ----------------

def dnbins(nbins, dlogq):
    if dlogq < 0:
        return 1
    n = int(np.floor(dlogq * nbins))
    return n if n > 0 else 1


def nbins_iter(i, nb0, dlogq):
    nb = nb0
    for _ in range(i):
        nb += dnbins(nb, dlogq)
    return nb


def gpu_sim(t, y, dy, freqs, qmin, qmax, noverlap=2, dlogq=0.5):
    """Returns power (float64 array), sols [(q, phi_sub) float32], and
    per-freq bin membership info for drilling into the worst freq."""
    t_sub, epoch = subtract_epoch(t)
    t32 = t_sub.astype(f32)

    # host-side weights (float64, as in eebls_gpu)
    w = np.power(dy, -2)
    w /= np.sum(w)
    ybar = np.dot(w, y)
    YY = np.dot(w, np.power(np.array(y) - ybar, 2))
    yw = ((np.array(y) - ybar) * np.array(w)).astype(f32)
    w32 = np.asarray(w).astype(f32)

    nbins0 = int(np.floor(1. / qmax))
    nbinsf = int(np.ceil(1. / qmin))

    levels = []
    j = 0
    while nbins_iter(j, nbins0, dlogq) <= nbinsf:
        levels.append(nbins_iter(j, nbins0, dlogq))
        j += 1

    freqs32 = np.asarray(freqs).astype(f32)
    dphi = f32(1.0) / f32(noverlap)

    powers = np.zeros(len(freqs))
    sols = []
    for i, fq in enumerate(freqs32):
        prod = t32 * fq                       # float32 multiply
        phi = prod - np.floor(prod)           # mod1
        best_val = f32(0.0)
        best = None  # (nb, s, jphi)
        for nb in levels:
            for s in range(noverlap):
                arg = f32(nb) * phi - f32(s) * dphi   # float32
                b = np.floor(arg).astype(np.int64) % nb
                yw_bin = np.zeros(nb, dtype=f32)
                w_bin = np.zeros(nb, dtype=f32)
                np.add.at(yw_bin, b, yw)
                np.add.at(w_bin, b, w32)
                # bls_value, float32
                with np.errstate(divide='ignore', invalid='ignore'):
                    val = yw_bin * yw_bin / (w_bin * (f32(1.0) - w_bin))
                ok = (w_bin > f32(1e-10)) & (w_bin < f32(1.0) - f32(1e-4))
                val = np.where(ok, val, f32(0.0)).astype(f32)
                k = int(np.argmax(val))
                if val[k] > best_val:
                    best_val = val[k]
                    best = (nb, s, k)
        nb, s, jphi = best if best is not None else (levels[0], 0, 0)
        q_sol = f32(1.0) / f32(nb)
        # store_best_sols: phi = mod1((float)(q_d*(jphi + s*dphi_d)))
        phi_d = (1.0 / nb) * (jphi + s * (1.0 / noverlap))
        phi_sol = f32(phi_d)
        phi_sol = phi_sol - np.floor(phi_sol)
        powers[i] = float(best_val) / YY
        sols.append((float(q_sol), float(phi_sol), nb, s, jphi))
    return powers, sols, epoch


# ---------------- test_standard replica ----------------

def run_case(q_index, phi_index, tshift, roundtrip=True, verbose=False):
    q_values = np.logspace(-1.5, np.log10(0.1), num=100)
    phi_values = np.linspace(0, 1, int(np.ceil(2. / min(q_values))))
    q = q_values[q_index]
    phi = phi_values[phi_index]
    freq = 1.0

    t, y, dy = data(snr=10, q=q, phi0=phi, freq=freq, baseline=365.,
                    tshift=tshift)

    df = min(q_values) / (10 * (max(t) - min(t)))
    delta_f = 5 * df / freq
    freqs = np.linspace(freq * (1 - delta_f), (1 + delta_f) * freq,
                        int(5. * 2 * delta_f * freq / df))

    power, sols, epoch = gpu_sim(t, y, dy, freqs,
                                 qmin=0.1 * q, qmax=2.0 * q,
                                 noverlap=2, dlogq=0.5)

    bls_c = []
    for fq, (q_sol, phi_sub, nb, s, jphi) in zip(freqs, sols):
        if roundtrip:
            phi_orig = (phi_sub + (epoch * fq)) % 1.0   # eebls_gpu pr65
            p = single_bls_pr65(t, y, dy, fq, q_sol, phi_orig)
        else:
            p = single_bls_old(t, y, dy, fq, q_sol, phi_sub)
        bls_c.append(p)
    bls_c = np.asarray(bls_c)

    diffs = np.absolute(power - bls_c)
    pows = np.asarray(bls_c)
    upper = 1e-3 * pows + 1e-5
    nviol = int(np.sum(diffs > upper))
    mostly_ok = nviol / len(pows) < 1e-2
    not_too_bad = diffs.max() < 1e-1
    iworst = int(np.argmax(diffs))
    res = dict(q_index=q_index, phi_index=phi_index, tshift=tshift,
               epoch=epoch, nfreq=len(freqs), nviol=nviol,
               maxdiff=float(diffs.max()), iworst=iworst,
               fworst=float(freqs[iworst]),
               p_gpu=float(power[iworst]), p_cpu=float(bls_c[iworst]),
               sol=sols[iworst],
               mostly_ok=mostly_ok, not_too_bad=not_too_bad,
               PASS=bool(mostly_ok and not_too_bad))
    if verbose:
        for k, v in res.items():
            print(f"  {k}: {v}")
    return res


if __name__ == '__main__':
    print(f"{'qi':>3} {'pi':>3} {'shift':>6} {'epoch':>6} {'nviol':>5} "
          f"{'maxdiff':>10} {'fworst':>10}  PASS")
    for tshift in (0.0, 4.5):
        for qi in (0, 5, -1):
            for pi in (0, 10, -1):
                r = run_case(qi, pi, tshift)
                print(f"{qi:>3} {pi:>3} {tshift:>6} {r['epoch']:>6.0f} "
                      f"{r['nviol']:>5} {r['maxdiff']:>10.4g} "
                      f"{r['fworst']:>10.6f}  "
                      f"{'PASS' if r['PASS'] else 'FAIL'}"
                      f"  (mostly_ok={r['mostly_ok']}, "
                      f"not_too_bad={r['not_too_bad']})")
