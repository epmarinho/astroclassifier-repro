# Dataset-v2D — prospective curation and shortcut-control gate

**Completed as a conditional proposal, not an activated benchmark.** All473 frozen candidates remain in the ledger. Dataset-v2A/v2B/v2C are byte-identical across 8440 files. No source FITS/preview/manifest/report/inventory was renamed or altered; all new outputs are under this phase directory. No network, acquisition, package installation, model fitting/inference, checkpoint access, final split, manuscript or Git mutation occurred.

**Representation remains open.** There are330 quality-ready proposed candidates and70 genuine author-quality decisions, with up to100 conditional slots per class. This is not400 admitted usable objects. Quality readiness is not class/identity certification, representation adequacy or proof of shortcut control. No image-based physical identity or class inference was performed.

## Preservation and prospective policy

`qc_issue_policy.json` was frozen before raw-pixel triage and before proposing admissions. File SHA-256: `4efec5d9bc393ffcd6ec07bee6ee2d50ccc70378753b298e916f65eb8c3ed4aa`; canonical digest: `6d18fb78b4000e2ee88b0bcfbdd9e6d5242daaa74bac3e7cbe5017efc8dd31f0`. All174 distinct prior QC issue names are mapped explicitly, with rationale. Unknown new issue types cannot silently become admissions.

a: hard signal/coverage failure, established from actual required-band facts; b: serious localized artifact or unresolved usable-signal/association question; c: non-blocking extent/FOV/axis metadata uncertainty; d: compulsory footprint grouping; e: ordinary astrophysical context, weak nonzero contrast, halos/nebulosity or a scalar screening proxy alone. Historical umbrella flags are expanded to underlying evidence rather than becoming standalone rejections.

Hard blocks: missing/corrupt/wrong-grid required band; no finite signal or constant finite pixels in any required band (including all-zero); declared BLANK/nonfinite >10% globally or >1% in central64×64 source pixels. Zeros are not presumed BLANK. Visual blank-looking/partial-coverage and percentile-clipping proxies without those hard facts remain bounded quality questions. A white fixed-stretch core does not establish photographic saturation. Metadata, overlap, halos, surrounding nebulosity or weak contrast alone do not reject.

## Why blanket rejection is invalid

| Class | C pass / review | Review fraction | If only C pass retained |
|---|---|---|---|
| galaxies | 77 / 43 | 35.8% | 77/120 |
| globularclusters | 71 / 42 | 37.2% | 71/113 |
| nebulae | 19 / 101 | 84.2% | 19/120 |
| opencluster | 38 / 82 | 68.3% | 38/120 |

Rejecting all review records would retain77/120 galaxies,71/113 globular clusters,19/120 nebulae and38/120 open clusters. That would select heavily by astrophysical appearance, scale, catalogue metadata and survey context. This proposal does not do so. `selection_bias_diagnostics.json` retains subtype/FOV summaries for the full population, the inappropriate pass-only counterfactual, conditional slots and quality-ready proposal. This triage reduces an identifiable bias mechanism; it does not establish that all residual selection bias has vanished.

## Per-class proposal accounting

The following four categories are mutually exclusive and sum to473. “Deterministically usable” means no remaining deterministic quality impediment in a proposed slot under this phase policy, subject to the representation/control gate. Unused reserves retain their separate quality status, including serious issues deferred until they are actually needed.

| Class | Quality ready | Blocked | Author pending | Reserve only | Conditional slots |
|---|---|---|---|---|---|
| galaxies | 93 | 1 | 7 | 19 | 100 |
| globularclusters | 89 | 3 | 11 | 10 | 100 |
| nebulae | 74 | 0 | 26 | 20 | 100 |
| opencluster | 74 | 3 | 26 | 17 | 100 |

Across the full473 candidates,372 have no hard or serious-quality block,94 have serious issues and7 are blocked. The unused reserves comprise42 quality-ready and24 serious-review cases; these66 are not admitted and the24 serious reserve cases are not requested for author review now. Among400 conditional proposed slots,330 are quality ready and70 remain pending. Pending/deferred cases are never replaced just to obtain cleaner-looking or more easily adjudicated images.

## Zero bands and reserve substitutions

Seven all-zero products affect five primaries and two unused reserves. All seven are forbidden as three-channel inputs. No filling, redownload, repair or author override is offered. Substitutions use frozen reserve ordering, same class/subtype where possible; quality-pending reserves are not skipped for cleaner ones.

| Blocked candidate | Role / band | Proposed substitute | Condition |
|---|---|---|---|
| SIMBAD:NGC  3094 | primary / nir | SIMBAD:IC  381 | subtype shift; quality_usable |
| HEASARC-GLOBCLUST:NGC 6304 | primary / blue | HEASARC-GLOBCLUST:ESO-SC06 | quality_usable |
| HEASARC-GLOBCLUST:Pal 6 | primary / blue | HEASARC-GLOBCLUST:Terzan 6 | pending_genuine_author_review |
| HEASARC-GLOBCLUST:Pal 8 | primary / blue | HEASARC-GLOBCLUST:Terzan 10 | pending_genuine_author_review |
| SIMBAD:NGC  2420 | primary / nir | SIMBAD:NGC  1579 | quality_usable |
| SIMBAD:NGC  6134 | reserve / blue | None: unused reserve | remain blocked |
| SIMBAD:Cl Trumpler   27 | reserve / blue | None: unused reserve | remain blocked |

The galaxy replacement is explicitly barred_spiral→unbarred_spiral because no same-subtype frozen reserve exists. The frozen primary roles themselves contain24 each of the first four galaxy/nebula subtypes and only4 unbarred spirals/SNR; their20 reserves are entirely those latter subtypes. This inherited subtype imbalance is disclosed, not silently rebalanced or described as uniform. See the exact source indices, paths and SHA-256 hashes in `zero_band_resolution.csv` and subtype counts in `selection_bias_diagnostics.json`.

## Scale and shortcut risk

Full per-class FOV distributions, all473 sorted values, empirical CDFs, unique-value counts and quantiles are preserved in `FOV_distributions.csv` and `shortcut_risk_analysis.json`. These include every frozen candidate, including blocked records and reserves; no outcome-dependent pruning precedes the analysis.

| Class | FOV min / Q1 / median / Q3 / max (arcmin) | Median sampling (arcsec/pixel) | Extent unknown |
|---|---|---|---|
| galaxies | 5 / 5 / 5 / 5 / 23 | 0.585938 | 0 |
| globularclusters | 5 / 5 / 5 / 6.36 / 15.64 | 0.585938 | 4 |
| nebulae | 5 / 5 / 15 / 18 / 120 | 1.75782 | 47 |
| opencluster | 5 / 14.9 / 28 / 74.15 / 120 | 3.28127 | 9 |

Class explains 0.5164 of in-collection log(FOV) variance (eta-squared). Predeclared FOV bins have mutual information 0.6471bits and Cramér V 0.5183 with class. These are descriptive associations, **not predictive accuracy**, fitted baselines, significance tests or proof of causal leakage. Pixel scale and FOV are almost deterministic functions here and their information must not be added as independent evidence.

| Potential shortcut proxy | Mutual information (bits) | Cramér V |
|---|---|---|
| Extent missingness | 0.1535 | 0.4708 |
| FOV cap | 0.0733 | 0.2988 |
| Recorded explicit artifact | 0.0457 | 0.2469 |
| Raw maximum plateau >0.5% | 0.0034 | 0.0677 |
| Outer-field tile structure, pooled quartiles | 0.1572 | 0.2484 |

All1,419 products were reread offline with byte hashes checked. The outer-field mask excludes central128×128 pixels. Fixed32×32 tile medians, normalized tile spread/IQR, row/column median steps, centre/background differences and raw clipping/zero fractions quantify field structure. Their complete per-band facts and per-class distributions are retained. Outer fields can contain real diffuse target light and neighbours, and photographic DN is not a common flux calibration. Manual artifact flags can also reflect observer sensitivity and prior screening; these are risk indicators rather than ground-truth artifact incidence. Empirical maximum plateaux and display clipping do not certify calibrated physical saturation.

## Representation comparison — no silent choice

Option A retains catalogue-adaptive512×512 fields and far more useful structure/context, but its strong scale and background association remains. Option B can geometrically derive common4arcmin fields at15arcsec/pixel, or nested1/2/4arcmin views, without new data. It has only16×16 effective samples at the largest scale. A fine512×512 resized output would not restore missing resolution; common pixels also do not guarantee common PSF. Larger common fields are unavailable in many5arcmin products. The4arcmin margin avoids the half-pixel centring asymmetry of an exact5arcmin field.

All473 candidates have geometric4arcmin support in all bands, but this includes blocked products and is not a signal certificate. Recorded extent exceeds4arcmin for5 galaxies,15 globular clusters,36 nebulae and98 open clusters;60 candidates have unknown extent. At15arcsec/pixel,15 galaxies,2 globular clusters and13 nebulae have recorded extent below2 pixels. These proxy counts understate outer-extent loss for Harris half-light diameters. Option B therefore changes the scientific task and discards substantial information. Neither representation is selected or rendered here. See `benchmark_representation_options.md`; a final benchmark cannot claim morphology-only generalization without addressing these losses/confounds.

## Blinded author review and false-pass audit

Only70 genuine proposed-slot cases are in `blinded_review/index.html` and the blank `author_decisions.csv`. Classes, subtypes, catalogue names, frozen roles, prior dispositions and class-bearing issue text are hidden; opaque randomized IDs and ordering are independent of class. All three512-pixel individual bands, pseudo-RGB, WCS coordinate marker and unresized128-pixel crops in all four views are shown with neutral raw QC facts. “Native” refers to the downloaded FITS output grid, not original photographic-plate resolution. FOV and appearance still provide clues: this is procedural label blinding, not perceptual or cryptographic blinding. Keep the unblinding key and class-bearing ledgers closed during quality decisions.

Allowed decisions are exactly accept_quality, reject_signal, reject_wrong_or_uncentered_field, defer_ambiguous and request_alternative_product. Author decisions remain blank; none was inferred. Wrong-field decisions require independent target/centering evidence; the brightest peak need not coincide with the coordinate, so uncertainty can be deferred. The author is not asked to classify an object by appearance. Requests for alternative products authorize no acquisition here. `author_decision_template.csv` preserves the immutable blank template; the identical `author_decisions.csv` is intentionally writable for author completion and must be separately hashed at a future authorized ingestion.

A frozen deterministic FOV-rank-stratified audit sampled10 technical_pass and8 review_needed per class:40+32=72 cases, mixed under the same blinded panel format. All12 mosaics were inspected with native central crops; two suspected new line defects were confirmed at full512pixels. No new definite defect was found among the40 technical_pass sentinels at this inspection scale. Two additional line/annotation findings were in already review-needed, already serious-review records; no C disposition or author decision changed. Zero observed false passes in this small sentinel sample does not prove zero population false passes or establish calibrated screening sensitivity. See `audit_plan.json`, `audit_results.json`, and the retained detail images.

## Atomic overlap constraints and future controls

The immutable source graph has21 edges,17 nontrivial components involving36 candidates, and454 total components including isolated nodes. Every original component is retained verbatim as an atomic future assignment unit, including blocked nodes, unused reserves, cross-class members and bridges. No final split is created, and no component is dissolved by excluding or deferring a bridge. All bands/renderings of each frozen physical-object candidate remain one observation. Neither this grouping nor quality review newly certifies physical identity from appearance.

`future_leakage_control_plan.md` requires an unfitted-now FOV/metadata-only baseline, outer-field/target-masked controls with explicit residual-emission caveats, scale/extent-stratified performance, and future object-level plus original-overlap-component-aware splitting. No catalogue type, name or class-bearing metadata can be used as baseline predictors. All controls, representation, preprocessing, quality outcomes and memberships must be frozen before future held-out evaluation; no test outcome can trigger curation or tuning.

## Validation and stopping point

Validated: all8,440 original file sizes/hashes unchanged; all473 rows uniquely accounted for with frozen labels/roles; seven zero bands blocked; five deterministic reserve queues independently reproduced; only70 genuinely pending proposed slots in the blind package;560 label-free image assets and280 native crops checked against unchanged source pixels outside marker segments;72 audit records with source-image hashes; complete FOV distributions; all17 original nontrivial components preserved. No author decisions, representation, admissions or split were activated.

Inventory SHA-256 is in `inventory_receipt.json`; the report itself is covered by `file_inventory.json`. The inventory and receipt exclude themselves to avoid circular hashes. Initial/final restricted Git status is retained and outside this new directory is unchanged. Stop after Dataset-v2D. Author-quality decisions and a separately approved representation/control protocol are necessary before any benchmark activation or training.
