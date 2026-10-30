# Dataset-v2C — acquisition and prospective QC

Acquisition complete: **1,359 new structurally validated FITS + 60 unchanged pilot FITS = 1,419** for **473 frozen catalogue candidates**. Structural validity does not establish usable signal: all pixel/visual QC problems are retained below. No object was admitted or rejected; no split, model, checkpoint or v1 scientific record was accessed or modified.

Dataset-v2A and Dataset-v2B are byte-identical across all 345 preserved files, including inventories and receipts. The sixty pilot FITS remain referenced at their original v2B paths and were never requested again. All 1,359 new requests are the predeclared exact CDS HiPS2FITS URLs; only DSS2 blue/red/NIR, fixed 512×512 TAN ICRS geometry, north-up/east-left, rotation zero, with the unchanged catalogue-only FOV rule. Primaries and reserves are both included. No candidate-specific adaptation, retries, replacement, extra metadata service or remote preview requests.

## Prospective QC results

| Class | Objects | Technical pass | Review needed | Technical failure | Valid FITS |
|---|---:|---:|---:|---:|---:|
| galaxies | 120 | 77 | 43 | 0 | 360 |
| globularclusters | 113 | 71 | 42 | 0 | 339 |
| nebulae | 120 | 19 | 101 | 0 | 360 |
| opencluster | 120 | 38 | 82 | 0 | 360 |

**Signal/coverage failures remain visible:** 7 products have constant pixels, including 7 entirely zero products. These pass structural FITS parsing but do not provide useful signal in that band; their objects retain review-needed dispositions under the frozen QC policy. See `constant_pixel_products.csv`. They are not counted as scientifically usable benchmark admissions.

- SIMBAD:NGC  3094, nir: constant value 0.0; SHA-256 `a305458276ff19b4cc9a4a56dbd26512246ce26fde1f9d58fe9f5b53651730aa`.
- HEASARC-GLOBCLUST:NGC 6304, blue: constant value 0.0; SHA-256 `2933bedd21068863a01e04145087f29d77ec2f88ded6a467b35701f367f54811`.
- HEASARC-GLOBCLUST:Pal 6, blue: constant value 0.0; SHA-256 `198cc940b58bac0147179d75bd12bb904663f7e50a80973269fd1abc28f7d6d1`.
- HEASARC-GLOBCLUST:Pal 8, blue: constant value 0.0; SHA-256 `79a7487fd4e0a99488dfaa7e1cab188a0fd697a8d0803f6e363f9cf4e8f17d25`.
- SIMBAD:NGC  2420, nir: constant value 0.0; SHA-256 `d65ec267fd0787ffc8c435afc44f47a7dd7561be181dd8d79a680a5619c1f2a4`.
- SIMBAD:NGC  6134, blue: constant value 0.0; SHA-256 `a03ffcf338e9c7c8620bbfda83e61ecb4162ae2d459f865683ff6814ba28cd16`.
- SIMBAD:Cl Trumpler   27, blue: constant value 0.0; SHA-256 `9c2f58b15ad8b79fc9a80488319f913289df275beb8462f0e0c2d83fcf442c9a`.

QC labels are prospective screening dispositions only. `technical_pass` is not benchmark admission or identity certification. All flags remain in the per-object table; counts are non-exclusive. Complete class/subtype issue counts are in `counts_by_class_subtype.csv` and `issue_counts_by_class_subtype.csv`.

The unchanged pilot checks cover HTTP200; full primary FITS blocks and 512×512 dimensions; header preservation; supported integer/float encoding with BSCALE/BZERO/BLANK; finite and zero pixels; p1/p99 dynamic range; empirical repeated-max/clipping proxy; row/column median artifact proxy; returned band history; catalogue/WCS centering; finite invertible TAN WCS and round-trip corners; identical three-band grids. Additional verification checks returned pixel scale against the exact frozen request FOV. A NaN from an entirely invalid row/column in the original median-outlier helper is recorded as an unavailable screen and review flag, never silently accepted.

Frozen review thresholds: invalid/declared BLANK >1%; zero >95% (zeros are not presumed blank); max plateau >0.5%; row/column median MAD score >8; catalogue extent/FOV outside [0.02,0.8]; extent below16 output pixels; missing, capped or reversed catalogue axes. Full header cards and original byte hashes are retained. FITS HISTORY confirms the requested DSS2 band. Resampled products may lack original plate IDs/exposure times and photometric calibration; their absence is documented. White cores in fixed-stretch previews do not prove photographic saturation.

Visual screening is documented for 473 objects; 0 remain explicitly pending. The original20 pilot findings are preserved, including every prior review-needed case. Per-object records state actual inspection resolution and notes. Automated proxies and contact-sheet inspection can miss subtle plate defects; no artifact-free or visual physical-identity certification is claimed.

## Field-overlap dependence graph

All **111,628 unordered pairs** across473 candidates were tested. The graph contains **21 overlap edges**, **17 nontrivial connected components**, 454 total components including isolated nodes, and a largest component of **3 objects**. All fields use actual retained FITS WCS: **True**.

Footprints use the closed spherical quadrilateral at pixel outer edges0.5/512.5 and the union over three bands. An enclosing spherical-cap rejection precedes vertex containment and great-circle edge intersection; RA wrap, poles, containment, edge crossings without contained vertices, boundary touches and symmetry have explicit tests. Touching within1e-10 radians is conservatively connected. There is no centre-only approximation. Overlap does not establish shared physical identity.

**Future constraint:** every connected component must remain in one split, or some observations must be prospectively excluded before a split is made. No split or exclusion is constructed here. See `footprint_overlap_graph.json`, `footprint_overlap_edges.csv`, and `footprint_components.csv`.

## Storage and provenance

New FITS: 1,359 files / 720,161,280 bytes. Reused FITS:60 / 31,795,200 bytes, with zero duplicate storage. All retained FITS: **751,956,480 bytes** (0.700314GiB). The pilot projected751,956,480bytes for473 objects; the observed total is now measured rather than projected.

Derived previews: 1,892 PNGs / 509,979,586 bytes; 95 class contact sheets / 301,753,327 bytes. Full phase storage including JSON, headers, overview sheets, code and inventories is in the final inventory receipt. Raw masters are immutable; derived PNGs are inspection aids only.

Every request preserves URL, exact survey ID and parameters, local/UTC timestamps, HTTP status/headers, elapsed time, raw byte size/SHA-256 and full FITS cards. `provenance_ledger.json` and `.csv` bind reused and new products. The original v2B policy/cache is reused, without further metadata requests. Retain DSS/STScI/NASA/CDS and underlying survey credits; a HiPS metadata licence is not a blanket assertion of unrestricted redistribution of every underlying image.

## Review index and preservation

Open `review_index.html` locally for all473 frozen identities, class/subtype and role, FOV, class contact sheets, full512-pixel previews, FITS masters and QC/provenance records. Pseudo-RGB maps NIR/red/blue to R/G/B with the unchanged independent p1/p99 asinh stretch, is deterministic, and is not calibrated true colour. All source files remain unchanged.

The completed-record offline resume check verified all1,359 cached new records with **zero network requests and zero inference**. All original hashes and all retained FITS hashes were independently checked again. Restricted initial/final Git status outside v2C is identical. No installation, split, training, inference, checkpoint loading, benchmark admission, relabeling, image rewriting/deletion, manuscript editing, staging, commit or push occurred.

A first finalization attempt was stopped by the write audit before Python environment discovery could open /dev/null outside this phase directory. Its exact source and all partial outputs are preserved under `failed_finalization_attempt_01/`, with hashes and a failure record. The correction only replaces `platform.platform()` (which launches a processor query subprocess) with the read-only `os.uname()` system call. No acquisition request, FITS, QC threshold, visual record, footprint algorithm or disposition changed; the repeated validation and resume checks performed zero network access.

Validation passed: **True**. This validates the acquisition/QC evidence, while retaining all scientific and visual issues. Stop after Dataset-v2C; author review and a separately authorized prospective admission/grouping protocol are required for later benchmark construction.

The inventory SHA-256 is stored externally in `inventory_receipt.json` and returned to the author; `REPORT.md` is itself covered by `file_inventory.json`. The inventory and its receipt exclude themselves to avoid a circular hash.
