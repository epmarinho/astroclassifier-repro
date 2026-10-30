# Dataset-v2F — representation, object grouping, split and shortcut-control design

**Ready for split materialization: false.** Complete proposed image recipes, grouping constraints, exact split algorithm and shortcut-control specifications are frozen. No real split assignment, tensor, model, prediction or benchmark metric was produced.

Only Dataset-v2E supplies quality eligibility. The primary pool contains exactly370 quality-ready candidates:330 carried-forward deterministic cases and40 explicit author acceptances. The13 rejected products,17 deferred cases,66 unused reserves and7 earlier blocked candidates are outside this pool. No replacement, reserve promotion, class reassignment or new acquisition occurred.

## Counts established before proposing the split algorithm

| Frozen class | Three-band observations | FITS band images | Distinct catalogue-supported objects | Assignment units touching class |
|---|---:|---:|---:|---:|
| galaxies | 98 | 294 | 98 | 93 |
| globularclusters | 92 | 276 | 92 | 92 |
| nebulae | 85 | 255 | 85 | 82 |
| opencluster | 95 | 285 | 95 | 92 |
| **Total** | **370** | **1110** | **370** | **358 unique units** |

The class-touching unit counts sum to359 because one unit spans two classes; the global count is358. Three FITS bands jointly form one observation. The object is the future statistical unit; neither band files, previews nor augmentations inflate n. Counts and full identity/lineage records were written to object_group_inventory.json before split_protocol.json.

## Reconstructed physical-object and field grouping

Reconstructed from frozen catalogue main identifiers, aliases and E product ownership, using case/whitespace/source-prefix normalization while preserving catalogue signs, decimals, brackets and component suffixes. No coordinate-nearest, perceptual-similarity, model-agreement or visual identity inference was used. There are no cross-candidate main-ID/alias collisions, unresolved shared-alias relations or identity/class contradictions in these frozen records. These are catalogue-supported physical-object groups, not new online or substructure adjudication.

All473 original candidate nodes enter the constraint union before eligibility projection. Each same-object image/band/known derivative stays with its object. All17 original nontrivial footprint components (21 edges,36 original candidates) remain indivisible, including excluded bridges and cross-class relations. Of these components,11 retain multiple eligible objects (10pairs and1triple) and6 retain one eligible object. This yields370 objects in358 active assignment/dependence units. Rejecting or deferring a bridge does not dissolve an original component.

All2534 known standalone derivatives are explicitly parent-owned across the full473-source universe:1972 B/C previews,560 D blinded individual panels/crops and2 D full-resolution audit details. Of these,1872 have E-eligible parents (1552 B/C previews and320 D panels/crops);662 have noneligible parents. Every one inherits the physical-object and full-universe split-unit constraint and none is a new observation or authorized model input. Contact sheets and audit mosaics combine objects and are categorically excluded from model input, rather than being treated as astronomical observations that would force unrelated objects together. Only1110 exact E-master FITS references may later generate tensors.

## Recovered declarations and explicit author decisions

No adopted v2 numerical split proportions, class floors or split seed were found in A–E. v2A explicitly deferred them; D specified group-aware construction without numbers; E preserved that open gate. The earlier context contains different scoped v1 declarations:

| Historical scope | Fractions train/val/test | Object floor/class | Split seed |
|---|---|---|---|
| v1 Phase4F | 485/801,158/801,158/801 | 20/5/10 | 20261022 |
| v1 Phase4F.1 scenario | 60/20/20 | 20/5/10 | not separately declared |
| v1 Phase4F.2 proposal, confirmed floor in4F.3 | 60/20/20 | 21/6/10 | 3908274 |

The later v1 floor explicitly superseded the older v1 minimum;21/6/10 also satisfies20/5/10, so the floors are not mathematically incompatible. The differing fractions and seeds cannot both determine one split. Neither policy is silently carried into v2. The earlier v1 preference for100 total test objects was a planning preference, not a v2 hard floor;20% of370 is about74. No software minimum of one group per class is substituted for a scientific floor.

The D blinding and sentinel seeds2026102301/2026102302 are not split seeds. New namespaced prospective seeds2026102303 for augmentation and2026102305 for bootstrap have different roles and never select the split. Declaration snapshots, hashes, scopes and JSON pointers are retained in frozen_declarations/, declaration_recovery.json and historical_preference_addendum.json. No v1 images, quality memberships, checkpoints, test results or prediction records were imported.

Active v2 split policy: **None**. Primary representation: **None**. The two explicit author decisions were requested; absent a supplied resolution, they remain pending. The frozen profiles are complete alternatives, not two simultaneously active split policies. open_decisions.md records the exact choices.

## Complete prospective FITS-to-tensor recipes

Both recipes read the exact unmodified FITS masters in **blue/red/NIR** channel order, not the NIR/red/blue pseudo-RGB used for QC. Parse FITS scaling in float64; verify hashes,512×512 primary dimensions, ICRS TAN WCS, common band grids, catalogue centre, corner round trips and source FOV. Failures stop materialization; they do not authorize repair, alignment fallback, exclusion or changed E quality decisions. No source PNG, EXIF/ICC/RGB conversion or astrophysical class cue is used as input.

Per-band percentiles are fixed linear p1/p99 on the defined raw view. Clip(DN−p1)/(p99−p1) to[0,1], then asinh(10z)/asinh(10). Keep finite raw zeros; reject entirely constant bands or degenerate ranges. Needed declared invalid pixels stop the materialization gate, with no masks, inpainting or candidate-specific padding. Independent band stretches sacrifice absolute/calibrated colour ratios; photographic DN does not establish physical colour calibration. Fixed normalization(u−0.5)/0.5 produces contiguous float32 CHW[3,224,224] in[-1,1]. There is no train/test-fitted image normalization.

**A, adaptive full-field:** retain each frozen5–120arcmin512×512 field, with no new crop or angular adaptation. After stretch and optional train-only discrete augmentation, resize512→224 by exact separable pixel-area overlap averaging, not an unspecified library default. No letterbox or padding. It retains much more morphology/context but is not FOV-neutral.

**B, common angular support/sampling:** fixed4arcmin TAN field centred at the frozen coordinates, fixed15arcsec/pixel16×16 grid. Apply a fixed added angular Gaussian FWHM15arcsec on source DN with radiusceil(4sigma), no boundary padding. Use fixed8×8 midpoint area quadrature and bilinear WCS sampling per output pixel. Quantiles are then estimated only on the common16×16 view, never the adaptive source field. After stretch and optional train-only D4 operation, repeat each cell14×14 to224×224. This has only16×16 effective samples and restores no resolution. Fixed added smoothing is not certification of identical measured original PSFs.

Recipe B crop, Gaussian support and bilinear margin fit in all370 frozen source fields; the minimum checked source-pixel margin is5.700033pixels. All1110 cached band headers agree by object, and cached invalid/dynamic-range conditions satisfy the proposed input contract. Actual FITS must be independently revalidated by any separately authorized materialization implementation; no pixels were decoded into a representation here.

Training-only augmentation is a SHA256-keyed, class-independent D4 transform (horizontal reflection and0/90/180/270degree rotation), applied identically to all channels/control views on the native recipe lattice. It uses no interpolation or padded corners. Validation/test use identity augmentation and the identical deterministic preprocessing. No random crops, brightness/colour jitter, noise, test-time augmentation or accuracy-driven preprocessing choices are allowed.

## FOV association and mandatory shortcut controls

FOV/scale association remains after E curation. The following are descriptive E-pool metadata, not fitted classifier or benchmark results:

| Class | FOV min / Q1 / median / Q3 / max (arcmin) | Known extent >4arcmin | Extent unknown |
|---|---|---:|---:|
| galaxies | 5 / 5 / 5 / 5 / 23 | 1 | 0 |
| globularclusters | 5 / 5 / 5 / 6.14 / 15.64 | 13 | 2 |
| nebulae | 5 / 5 / 15 / 15 / 120 | 14 | 39 |
| opencluster | 5 / 15 / 27.2 / 71.5 / 120 | 77 | 9 |

Class accounts for 0.5827 of E-pool log(FOV) variance descriptively. This is not predictive accuracy or causal proof. The primary representation choice cannot be made from future accuracy: A preserves context with strong confounding risk; B controls angular field/sampling but truncates large objects and blurs small ones. Upsampling B adds no information, and source PSF/grain can still retain clues. Neither option is silently chosen here.

shortcut_control_protocol.json freezes: (1) a train-only-imputed/scaled FOV/metadata logistic baseline, plus a fixed depth3 tree diagnostic; (2) mandatory common-field B images on the same objects/components/split, or A as a context comparator if B is primary; (3) a fixed central target-mask/outer-field control with intensity percentiles calculated from unmasked pixels only, so the removed target cannot leak through stretch statistics; (4) every declared FOV bin, including original source FOV even for B, plus recorded angular extent and pixel-extent strata. No catalogue type/name/alias/path/index or class-bearing metadata enters predictors. Background controls are not certified pure sky: diffuse emission and neighbours can remain. Every control is reported regardless of outcome, with no post-test curation or choice of a preferred view.

## Deterministic object-stratified split algorithm — not executed on real objects

The exact algorithm respects all full-universe groups, including the cross-class unit. Enumerate3^11=177147 assignments of the11 coupled active units in a seeded SHA256 order. For each candidate assignment, allocate singleton counts per class using exact rational convex squared-deviation minimization subject to the adopted object floors. Choose the globally minimal sum of squared deviations of role fractions across classes; deterministic tie order is frozen. Then assign each class’s single-object units in a seeded SHA256 order to the optimal counts. No FOV, quality rank, appearance or model outcome enters that objective. A budget cap, infeasible floors, identity conflicts or hash changes stop materialization without a heuristic fallback, new split seed, pool expansion or weakened floor.

Only synthetic correctness tests were run:30 count-apportionment cases against brute force, cross-class atomicity, input-order independence and stopping on infeasible floors. No real candidate membership was assigned. Nominal integer class targets for each unadopted profile are count arithmetic only and do not constitute a proposed or activated split.

## Future statistical unit and preservation

The physical astronomical object is the scoring unit. The three bands and all derivatives remain one observation. If multiple observations are ever separately authorized, primary scoring uses one deterministic representative per object, never the best-performing image; all other views remain same-role auxiliary evidence. Current data have one three-band bundle per object. For paired sample uncertainty, resample whole dependence components and retain object weights, not independent FITS rows or equal-weight component averages. The frozen bootstrap design is conditional on future frozen checkpoints and does not fully account for training-seed uncertainty. No statistics or model metrics were evaluated now.

All9,080 Dataset-v2A–v2E files, including completed D author decisions, immutable blank templates, reports and inventories, remain byte-identical. Initial/final restricted Git status is retained and outside F is unchanged. No source image/FITS/record, manuscript or historical result was modified. No package installation, training, inference, checkpoint loading, network, download, staging, commit or push occurred.

The first F design snapshot is preserved under implementation_attempts/attempt01_partial_derivative_lineage/. Its explicit derivative ledger covered eligible B/C previews; final provenance validation extended it to all source B/C previews and D individual blinded-review/detail panels. correction_record.json retains all original file hashes, including that snapshot inventory. No E eligibility, physical-object grouping, overlap component, representation formula or pending choice changed. No real split/tensor/model was generated by either attempt.

**Ready for split materialization: false.** Explicit author choices are still required for the v2 numeric split policy and primary representation. The design and alternatives are frozen; activation is not approved or inferred.
Stop after Dataset-v2F and await author review. file_inventory.json covers REPORT.md and all new evidence; the final inventory digest is in inventory_receipt.json and the final response. The inventory and receipt exclude themselves to avoid circular hashes.
