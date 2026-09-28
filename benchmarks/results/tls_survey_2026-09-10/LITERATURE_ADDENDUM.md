# Additional published population comparison

Read-only literature follow-up, 2026-09-11. This note was added during the
held-out injection search; it changes no frozen input, method, threshold,
tolerance, sample size, or interpretation gate. The prospective
[literature audit](../../../docs/TLS_LITERATURE.md) remains unchanged.

The 2025 SPLS preprint reports **60.2% biweight+TLS versus 56.8%
biweight+BLS recovery** over 10,000 injections into Kepler light curves:
a 3.4 percentage-point difference. Figure 13's incorrect-period recoveries,
6.5% and 6.3%, are distinct from null false-positive rates.
[Figure 13](https://arxiv.org/html/2512.02356v1/3_3_2_all.png)

The searches share approximately 39,029 trial periods. Astropy BLS uses
15 logarithmic durations and 15 bins per duration; TLS's minimum depth is
1 ppm. Method-specific thresholds target empirical 10% FPR, without a
described independent calibration/test-null split. Recovery includes
half/double-period aliases. Detrending windows use injected durations;
injections are central, circular transits with periods 10–480 days. The ROC
positive population excludes incorrect-period maxima, whereas the separate
recovery comparison includes all injections. The paper supplies neither
BLS resolution convergence nor a paired uncertainty interval for this
aggregate TLS–BLS difference.
[Methods and results, §§III.1.1–III.1.3](https://arxiv.org/html/2512.02356v1#S3.SS1)

This is additional evidence of a population-specific TLS recovery advantage.
It does not establish the advantage over our independently tuned comparator,
or supply an approximation allowance for the cuvarbase campaign.

The reviewer inspected three additional primary papers; this was the most
relevant population comparison. The parent independently checked the linked
methods and visually verified Figure 13. The downloaded figure remains in
the external work directory; its SHA256 is
`eb63ff47e4ec6e17e406d802ea46238f0656b994b1b712077b3b3ac9090aecd2`.
