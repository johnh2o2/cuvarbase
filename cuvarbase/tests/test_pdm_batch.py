"""CPU-side tests for the PDM batch API (C1, issue #33).

The actual GPU correctness (batch matching per-LC results, large_run
respecting max_memory) is validated on a pod; here we test the
batch-sizing arithmetic and the chunking/result-shaping logic by mocking
``PDMAsyncProcess.run`` so they run on CPU-only machines.
"""
import numpy as np

from cuvarbase.pdm import PDMAsyncProcess


def _proc():
    # Constructing the process does no GPU work (streams are lazy); on a
    # GPU-less machine ensure_context() just imports the stubbed module.
    return PDMAsyncProcess()


def test_batch_size_from_memory_arithmetic():
    proc = _proc()
    # per_lc = (3*1000 + 2*5000) * 4 = 52000 bytes
    assert proc._bytes_per_lc(1000, 5000) == 52000
    assert proc._batch_size_from_memory(
        1000, 5000, n_lcs=100, max_memory=520000) == 10
    # capped at n_lcs
    assert proc._batch_size_from_memory(
        1000, 5000, n_lcs=3, max_memory=10 ** 9) == 3
    # never below 1, even if a single LC exceeds the budget
    assert proc._batch_size_from_memory(
        1000, 5000, n_lcs=100, max_memory=1) == 1


def test_batched_run_const_nfreq_chunks_and_reuses_freqs(monkeypatch):
    proc = _proc()
    chunk_sizes = []
    freqs_seen = []

    def fake_run(data, freqs=None, **kw):
        chunk_sizes.append(len(data))
        freqs_seen.append(freqs)
        return [(freqs, np.zeros(len(freqs))) for _ in data]

    monkeypatch.setattr(proc, 'run', fake_run)
    monkeypatch.setattr(proc, 'finish', lambda: None)

    data = [(np.linspace(0, 10, 50 + i),
             np.zeros(50 + i), np.ones(50 + i)) for i in range(5)]
    freqs = np.linspace(0.1, 1.0, 20)

    res = proc.batched_run_const_nfreq(data, batch_size=2, freqs=freqs)

    assert chunk_sizes == [2, 2, 1]          # chunked by batch_size
    assert len(res) == 5                      # one result per lightcurve
    assert all(len(f) == 20 for f, p in res)
    # the same const grid is reused for every chunk (no per-LC recompute)
    assert all(fs is freqs_seen[0] for fs in freqs_seen)


def test_batched_run_const_nfreq_empty():
    proc = _proc()
    assert proc.batched_run_const_nfreq([], freqs=np.linspace(0.1, 1, 5)) == []


def test_large_run_uses_memory_capped_batch_size(monkeypatch):
    proc = _proc()
    captured = {}

    def fake_batched(data, batch_size=None, freqs=None, **kw):
        captured['batch_size'] = batch_size
        captured['nfreqs'] = len(freqs)
        return [(freqs, np.zeros(len(freqs))) for _ in data]

    monkeypatch.setattr(proc, 'batched_run_const_nfreq', fake_batched)

    # 6 LCs, max_ndata=1000, nf=5000 -> per_lc=52000; budget fits 4
    data = [(np.linspace(0, 10, 1000), np.zeros(1000), np.ones(1000))
            for _ in range(6)]
    freqs = np.linspace(0.1, 1.0, 5000)
    res = proc.large_run(data, freqs=freqs, max_memory=4 * 52000)

    assert captured['batch_size'] == 4
    assert captured['nfreqs'] == 5000
    assert len(res) == 6
