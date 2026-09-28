# Final collected data and setup transfer

Prepared locally from the verified final collection. This is a separate supplement to `../release-validation-transfer-prep/source-bundle.tar.gz`; it does not replace that immutable source bundle. No installation, GPU tests, provider operations or scientific searches occurred during preparation.

The 15,111,320-byte archive contains the original **80-case** development manifest unchanged, exactly the **11 predeclared original NPZ files** (14,619,794 bytes), original collected freeze/install/GPU/CPU/quota evidence, the final collector receipt, collection provenance, and setup derived by the unchanged pinned integration runner. All selected originals match final archive inventory `6cd942436ee8e1bfcf5acce8f74747f136031b817abd30b5b012e9aae4b78fe9`. The original complete manifest must not be rewritten to eleven rows. No earlier development or regenerated input is used.

| Artifact | SHA256 |
|---|---|
| data-setup-bundle.tar.gz | 634239ec084d487c932cdfecc8e5153bc53f0163e4cf00809752fddf3bca2183 |
| data-transfer-inventory.json | ad8a4dcdb08631e87c8279215e503e0ae96ac84ec4ed5511ab76332c3ed217aa |
| verify_data_transfer.py | 332a830a51e2f64e964e5a26ff5b74d6267ae333777b8121b23cde14e98899ac |
| setup-derived/setup.json | 65cb0e2a2e6bd7cddcca6b402d54051d97e75a423766c0b460af0648ff3fc5c9 |
| setup-derived/requirements.txt | e629af0ef28323c2c570c932010d02f5022f662434ccb490e593dbc8c5e014e1 |
| original-collector-final.json | 3ad52f46237a5b0cd662cfdc43eab367d5272e8c8cacb902da60956798695d68 |

Transfer the archive, external inventory and verifier to a fresh incoming directory on the separately authorized integration host. First extract and verify the earlier source bundle into `/workspace/release-integration` using its own runbook. Then check the verifier SHA and use this exact command from the incoming directory:

```sh
python3 verify_data_transfer.py   --archive data-setup-bundle.tar.gz   --archive-sha256 634239ec084d487c932cdfecc8e5153bc53f0163e4cf00809752fddf3bca2183   --inventory data-transfer-inventory.json   --inventory-sha256 ad8a4dcdb08631e87c8279215e503e0ae96ac84ec4ed5511ab76332c3ed217aa   --destination /workspace/release-integration
```

The destination must already exist. Verification checks all members and every destination collision before publishing missing files. It refuses overwrite; any interrupted partial extraction remains for review instead of being resumed or silently replaced. Paths are disjoint from the earlier source bundle.

The CPU-only `integrate.py setup` derivation has completed, producing 64 exact dependency versions from the original collected freeze/install and the separately identified observed build-tool metadata. Use those files with the separately reviewed installer; do not substitute local host/build freezes or upgrade versions. Once actual Linux Python/package pins match, bind on that host under the unchanged runner's runbook:

```sh
cd /workspace/release-integration
venv/bin/python ops/integrate.py bind   --baseline-root baseline --precursor-root precursor --release-root release   --manifest dev-final/manifest.json --inputs dev-final   --setup setup-derived/setup.json   --setup-sha256 65cb0e2a2e6bd7cddcca6b402d54051d97e75a423766c0b460af0648ff3fc5c9   --termination original-collector-final.json   --termination-sha256 3ad52f46237a5b0cd662cfdc43eab367d5272e8c8cacb902da60956798695d68   --output binding.json
```

This does not authorize launching the 900-second validation run. Its fixed development plan, controls, preserved failures and budget remain unchanged. The historical 5,111/5,120 exact comparisons and failed zero-mismatch qualification do not qualify the new release wiring.
