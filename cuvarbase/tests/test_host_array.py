"""Unit tests for the pinned-host-buffer helper (B3).

These exercise the allocation-strategy logic directly (mocking the pycuda
allocators) so they run on CPU-only machines: pinned by default, graceful
fallback to page-aligned when pinning fails, and pinned=False bypassing
pinning entirely.
"""
import numpy as np

from cuvarbase.memory import _host


def test_host_array_pinned_uses_pagelocked(monkeypatch):
    calls = []
    monkeypatch.setattr(_host.cuda, 'pagelocked_zeros',
                        lambda shape, dtype: (calls.append('pinned')
                                              or np.zeros(shape, dtype)))
    monkeypatch.setattr(_host.cuda, 'aligned_zeros',
                        lambda shape, dtype: (calls.append('aligned')
                                              or np.zeros(shape, dtype)))
    arr = _host.host_array((4,), np.float32, pinned=True)
    assert calls == ['pinned']
    assert arr.shape == (4,) and arr.dtype == np.float32


def test_host_array_falls_back_when_pinning_fails(monkeypatch, recwarn):
    _host._warned_fallback = False  # reset the warn-once latch
    calls = []

    def boom(shape, dtype):
        raise RuntimeError("locked-memory limit exhausted")

    monkeypatch.setattr(_host.cuda, 'pagelocked_zeros', boom)
    monkeypatch.setattr(_host.cuda, 'aligned_zeros',
                        lambda shape, dtype: (calls.append('aligned')
                                              or np.zeros(shape, dtype)))
    arr = _host.host_array((8,), np.float64, pinned=True)
    assert calls == ['aligned']            # fell back to page-aligned
    assert arr.shape == (8,)
    assert any('page-locked' in str(w.message) for w in recwarn.list), \
        "a fallback warning should be emitted"


def test_host_array_does_not_warn_when_fallback_also_fails(monkeypatch):
    # On a GPU-less machine both allocators raise (stubbed); the helper
    # must let that propagate WITHOUT claiming a fallback happened.
    _host._warned_fallback = False

    def boom(shape, dtype):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(_host.cuda, 'pagelocked_zeros', boom)
    monkeypatch.setattr(_host.cuda, 'aligned_zeros', boom)
    try:
        _host.host_array((2,), np.float32, pinned=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the page-aligned failure to propagate")
    assert _host._warned_fallback is False


def test_host_array_pinned_false_skips_pinning(monkeypatch):
    calls = []

    def must_not_call(shape, dtype):
        raise AssertionError("pagelocked_zeros must not be called when "
                             "pinned=False")

    monkeypatch.setattr(_host.cuda, 'pagelocked_zeros', must_not_call)
    monkeypatch.setattr(_host.cuda, 'aligned_zeros',
                        lambda shape, dtype: (calls.append('aligned')
                                              or np.zeros(shape, dtype)))
    arr = _host.host_array((3,), np.float32, pinned=False)
    assert calls == ['aligned']
    assert arr.shape == (3,)
