"""Guard a few README factual claims that map to real code/release state.

These are the claims that silently rot or contradict the code:
- the removed ``periodograms`` subpackage must not be advertised,
- ``import cuvarbase`` no longer requires a GPU / creates a context (B1),
- the candidate install points to the measured branch and distinguishes the
  published PyPI version; every link is absolute,
- ADS links should be https, and the test suite is CPU-runnable.
"""
import os
import re

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


def test_readme_installs_the_benchmarked_candidate():
    # The measured v1 candidate is not the currently published 0.2.5.
    # Keep an ordinary PyPI install from silently selecting a different
    # implementation from the one advertised by the benchmark figure.
    readme = _readme()
    installation = re.search(
        r"^## Installation\n(.*?)(?=^## |\Z)", readme, re.M | re.S)
    assert installation is not None
    text = installation.group(1)
    assert "0.2.5" in text and "PyPI" in text
    assert ("pip install 'cuvarbase @ git+https://github.com/"
            "johnh2o2/cuvarbase@v1.0-fixes'") in text
    assert "pip install cuvarbase\n" not in text


def test_readme_links_are_absolute():
    # Relative links do not resolve from PyPI; every ``](...)`` target
    # must be an absolute URL or an in-page anchor.
    readme = _readme()
    bad = [m for m in re.findall(r"\]\(([^)]+)\)", readme)
           if not (m.startswith("http") or m.startswith("#"))]
    assert bad == [], bad


def test_readme_ads_links_are_https():
    readme = _readme()
    assert "http://adsabs" not in readme
    assert "http://ui.adsabs" not in readme


def test_readme_testing_section_is_cpu_runnable():
    readme = _readme().lower()
    assert "tests require a cuda-capable gpu" not in readme
