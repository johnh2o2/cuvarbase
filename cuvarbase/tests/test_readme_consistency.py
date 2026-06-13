"""Guard a few README factual claims that map to real code/release state.

These are the claims that silently rot or contradict the code:
- the removed ``periodograms`` subpackage must not be advertised,
- ``import cuvarbase`` no longer requires a GPU / creates a context (B1),
- the install instructions must not point at the stale PyPI ``0.2.5``,
- ADS links should be https, and the test suite is CPU-runnable.
"""
import os

import pytest


def _readme():
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(root, "README.md")
    if not os.path.exists(path):
        pytest.skip("README.md not found (running outside the source tree)")
    return open(path, encoding="utf-8").read()


def test_readme_does_not_advertise_removed_periodograms_subpackage():
    readme = _readme()
    assert "periodograms/`" not in readme, (
        "README still lists the removed `periodograms/` subpackage")


def test_readme_import_does_not_claim_gpu_required():
    # B1: import creates no CUDA context and needs no GPU.
    readme = _readme().lower()
    assert "creates a cuda context at import" not in readme
    assert "importing cuvarbase still requires a working cuda" not in readme


def test_readme_install_not_pinned_to_stale_pypi():
    # The current PyPI release is 0.2.5; v1.0 installs from source until
    # 1.0.0 is published. A bare ``pip install cuvarbase`` would fetch the
    # stale version, so it must not be the advertised install command.
    readme = _readme()
    assert "pip install cuvarbase\n" not in readme


def test_readme_ads_links_are_https():
    readme = _readme()
    assert "http://adsabs" not in readme
    assert "http://ui.adsabs" not in readme


def test_readme_testing_section_is_cpu_runnable():
    readme = _readme().lower()
    assert "tests require a cuda-capable gpu" not in readme
