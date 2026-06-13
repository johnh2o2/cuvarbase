"""Keplerian transit-search relations + citation correctness (G1).

The transit-duration/period relation is Seager & Mallen-Ornelas (2003,
ApJ 585, 1038); the frequency-grid spacing and surface-orbit cutoff are
Ofir (2014, A&A 561, A138, "Optimizing the search for transiting planets
in long time series"). These tests pin the derived ``fmax0`` constant to
its physical value and guard the Ofir citation against the
Hippke & Heller TLS-paper title that earlier drafts pasted onto it.
"""
import glob
import os
import re

import numpy as np

import cuvarbase
from cuvarbase.bls import fmax_transit0, q_transit, freq_transit


def test_fmax_transit0_is_derived_surface_orbit_frequency():
    # fmax0 is the documented derived constant, NOT a literature value:
    # the Keplerian orbital frequency at the stellar surface,
    # f = sqrt(G rho_star / 3pi), evaluated at solar mean density.
    assert fmax_transit0(rho=1.0) == 8.6307

    # Validate the derivation from first principles (SI -> cycles/day).
    G = 6.674e-11                      # m^3 kg^-1 s^-2
    M_sun = 1.989e30                   # kg
    R_sun = 6.957e8                    # m
    rho_sun = M_sun / ((4.0 / 3.0) * np.pi * R_sun ** 3)
    f_surface = np.sqrt(G * rho_sun / (3.0 * np.pi))   # per second
    f_cpd = f_surface * 86400.0                        # cycles/day
    # within ~1% -- the small offset is the precision of the adopted
    # constants (the docs quote 8.612, the code 8.6307; same quantity).
    assert abs(f_cpd - 8.6307) / 8.6307 < 0.01

    # scales as sqrt(rho)
    assert np.isclose(fmax_transit0(rho=4.0), 2.0 * 8.6307)


def test_q_freq_transit_are_inverses():
    # q_transit and freq_transit implement SM03 eq. (3) and its inverse;
    # round-tripping must recover the input across the valid q range.
    qs = np.linspace(0.02, 0.45, 25)
    f = freq_transit(qs, rho=1.0)
    q_back = q_transit(f, rho=1.0)
    assert np.allclose(qs, q_back, atol=1e-6)


def test_ofir_2014_cited_with_correct_title():
    # Ofir (2014), A&A 561, A138 must carry its real title. Earlier drafts
    # pasted Hippke & Heller's TLS-paper title ("...periodic transits of
    # small planets") onto the Ofir citation. No cuvarbase source file
    # cites H&H by that formal title (they use "Transit Least Squares"),
    # so the phrase must not appear in package source at all.
    pkg_dir = os.path.dirname(os.path.abspath(cuvarbase.__file__))
    hh_title = "periodic transits of small planets"
    ofir_title = "Optimizing the search for transiting planets"

    offenders = []
    cites_ofir = []
    for path in glob.glob(os.path.join(pkg_dir, "*.py")):
        # collapse whitespace so wrapped titles still match
        flat = re.sub(r"\s+", " ", open(path).read())
        if hh_title in flat:
            offenders.append(os.path.basename(path))
        if "A&A 561" in flat or "561, A138" in flat:
            cites_ofir.append((os.path.basename(path), flat))

    assert not offenders, (
        "Hippke & Heller title pasted onto a citation in %s" % offenders)
    # and every file that cites Ofir 2014 uses the correct title
    for name, flat in cites_ofir:
        assert ofir_title in flat, (
            "%s cites A&A 561, A138 without Ofir's correct title" % name)
