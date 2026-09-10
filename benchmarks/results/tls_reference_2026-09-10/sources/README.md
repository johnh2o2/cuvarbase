# Source and environment identities

`scientific_sources.tar.gz` preserves the exact source bytes used to generate
the main and supplementary populations, instrument and compare the searches,
prove when the native correction is a no-op, execute the cohorts, and produce
their compact summaries. Its 17 members include the physical-signal helper
and the two stronger-control generator. `manifest.json` gives each member's
SHA256 and connects the archive to both unchanged study seals.

`production_sources.tar.gz` is the complete cuvarbase source/test snapshot
checked before confirmation. `gpu_tests.json` records 265 passing TLS tests
with no failures, errors or skips and identifies that exact snapshot. The
independent per-case output receipts separately verify the production source
identity used by the scientific runs.

The original scientific source archive is an audit trail. The maintained
[reproduction tools](../../../tls_reference/README.md) provide the portable
command-line workflow and record their own executing hashes. Their packaging
changes do not rewrite the original seals or create another independent
population. Development-only scripts, prototype engines, and internal draft
notes are not included; their historical identities remain visible where the
original seals recorded them.

`generation_environment.json` distinguishes distribution versions from module
version strings. The original CPU environment was Python 3.9.6 on macOS 26
arm64, NumPy 1.26.4, SciPy 1.12.0 and **batman-package 2.5.3**. That distribution
reports `batman.__version__ == "2.5.1"`; installing a package inferred from the
module string would not reproduce the recorded environment. Numerical input
regeneration was checked on that original environment. Cross-platform
floating-point equality is not assumed; the exact input bank supplies a
separate route that does not regenerate the physical signals.

`input_restoration_proof.json` records an independent byte, dtype, shape and
metadata check for all 209 restored cases and their 1,463 numerical arrays.
The exact restorer and verifier sources are in
`input_restoration_sources.tar.gz`. The verifier retains its original local
paths as an audit artifact; use the maintained `inputs.py` command for a
portable replay. All numerical arrays match. The two stronger controls have
different NPZ container encodings after restoration, which does not change
any stored numerical value; their original and restored container hashes are
both recorded.

The earlier 21-case development input manifest names a historical generator
version whose source bytes were not retained. Its numerical vectors and
original generator identity remain available, and its actual search/comparison
harness is included here. The maintained generator separately reproduced all
21 input arrays and NPZ container hashes on the original CPU environment.
This check does not recover the missing historical source version or make
those development cases independent evidence. Both long-period parents and
the two stronger controls have their exact generation source in this archive.

`result_postprocessing_sources.tar.gz` preserves the selected-grid metric
packager, outcome-table generator and original collection auditors. These
checks extract saved results and authenticate recovered records; they do not
create new search measurements or reconstruct a scientific acceptance gate.
The main collection receipt distinguishes the nine uncollected output
containers from the 405 archives removed under the original retention rule.

The supplemental collection audit authenticates its original 24-case acceptance,
all 72 records and 48 comparisons. Its nine uncollected retained NPZ containers
remain separate from 63 predeclared prunes. The stress packager explicitly
marks the three original solar-control output containers as uncollected and
leaves their unavailable truth-grid ranks blank; it preserves the original
failed comparison and does not substitute later diagnostic runs.
For the invalid-row fixture, it stores numerical and JSON identities for two
oversized transit-time lists instead of duplicating their values. Original
record hashes, all other output identities and comparisons remain unchanged.

The later [focused repeatability diagnostic](../stress/diagnostic/README.md)
has its own exact source archive and compact raw vectors. Its repeated
executions investigate the retained solar-control failure; they do not add
independent scientific samples or modify either completed population gate.
