# TLS accuracy and computational efficiency

This 9 September 2026 audit evaluates the earlier phase-binned TLS engine,
retained as `method='binned'`. “Defaults” below refers to that frozen engine.
See the [current transit report](../../../docs/TRANSIT_BENCHMARKS.md) for the
observation-level default. These measurements remain historical evidence.

This audit separates the sensitivity cost of cuvarbase's fast TLS approximation
from the effect of optimizing its implementation. Binning is inexpensive in
many ordinary transit examples, but a universal 1–2% SNR-loss bound is false.
The duration prior and search grids can matter more than binning alone.

| Evidence | Question answered |
|---|---|
| [Expected-SNR diagnostic](accuracy/README.md) | What do the fixed template, phase bins and coarse grids lose at the true period across physical transit regimes? |
| [Kernel validation](kernel/README.md) | Does skipping empty bins accelerate the same search while preserving its numerical results? |
| [High-impact recovery pilot](high-impact/README.md) | On new noisy TESS inputs, can GTLS recover narrow transits that the then-default binned search misses, and how do alternative cuvarbase settings behave? |

The CPU diagnostic uses 19 physical regimes and three search configurations,
plus observed TESS/ZTF cadences. The focused GPU pilot independently calibrates
each method before testing new injections and nulls. The engineering timings
compare the original and optimized CUDA kernels with identical inputs and
settings; they are not a new competitor sensitivity experiment.

At identical fine-resolution settings, the optimized ZTF search uses 23% less
time (1.30× faster); fine-resolution TESS timings are effectively unchanged.
All 169 TLS tests pass, and comparisons on 384 lightcurves preserve primary
periods and reported SNR within the recorded numerical checks. Separately,
the targeted high-impact pilot recovers 61/256 injections with the defaults,
112/256 with GTLS and 116/256 with a wider cuvarbase duration search. The pilot
does not establish equivalence between the latter two configurations.

The pre-optimization cuvarbase reference is
`11317fb0ff1b68af05ae3f67de5f298c9a90e46b`; public GTLS is pinned to
`74e449c325792a763dde4fbffab98039c5e8c111`. Each subdirectory records the exact
sources, input hashes, configuration and validation applicable to its claims.
Large generated lightcurve arrays are kept outside the release repository and
can be regenerated using the frozen protocol.

[The TLS numerical guide](../../../docs/TLS_NUMERICS.md) distinguishes the
current default from the retained binned engine. The separate
[calibrated survey benchmark](../tls_sensitivity_2026-09-09/README.md) supplied
the historical binned-TLS speed figure; current release claims are in the
[transit report](../../../docs/TRANSIT_BENCHMARKS.md).
