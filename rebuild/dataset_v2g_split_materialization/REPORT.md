# Dataset-v2G — one frozen object-grouped split

**Exactly one split was materialized and validated.** The author explicitly adopted the complete phase4f2 numerical profile and representationA as primary. Dataset-v2A–v2F remain byte-for-byte unchanged. No tensor, model, inference, prediction or benchmark result was generated or inspected.

## Author resolution and frozen method

The exact author message is preserved in author_resolution.json. Adopted train/validation/test target fractions are60/20/20, distinct-object class floors21/6/10, split seed3908274. No parts were imported from phase4f or other profiles. A remains the primary adaptive full-field scientific view; B remains the mandatory common4arcmin/16×16-effective-sample shortcut control. Metadata/FOV-only, outer-field and FOV-stratified controls remain mandatory regardless of outcome. No future accuracy can change the primary representation.

All numerical FITS-to-tensor and control recipes are the unchanged frozen F recipes; no tensor is materialized now. F’s historical pending-state documents remain intact; their two gates are resolved only in this separately recorded G phase. The source F inventory SHA-256 is `66eafccf1a30fa738a56359617ac4dcb5b4cfc73ec6684166c4ee7752d4b1b39`.

The original split_algorithm_reference.py was imported and optimize was invoked exactly once on358 active assignment units. It enumerated177147 coupled-unit assignments using the unchanged seed/tie rules and rational objective. No implementation acceleration, seed search, manual adjustment, alternate split or profile mixing occurred. Runtime and source/result hashes are retained in optimizer_invocation.json and optimizer_completion.json.

## Exact primary split counts

| Split | Physical objects / multiband observations | FITS band images | Assignment units | Actual object fraction |
|---|---:|---:|---:|---:|
| train | 222 | 666 | 222 | 60.000000% |
| validation | 75 | 225 | 75 | 20.270270% |
| test | 73 | 219 | 61 | 19.729730% |
| **Total** | **370** | **1110** | **358** | **100%** |

| Frozen class | Train objects / FITS / units | Validation objects / FITS / units | Test objects / FITS / units | Floors |
|---|---|---|---|---|
| galaxies | 59 / 177 / 59 | 20 / 60 / 20 | 19 / 57 / 14 | 21/6/10 satisfied |
| globularclusters | 55 / 165 / 55 | 19 / 57 / 19 | 18 / 54 / 18 | 21/6/10 satisfied |
| nebulae | 51 / 153 / 51 | 17 / 51 / 17 | 17 / 51 / 14 | 21/6/10 satisfied |
| opencluster | 57 / 171 / 57 | 19 / 57 / 19 | 19 / 57 / 16 | 21/6/10 satisfied |

The statistical unit is the physical astronomical object, with three jointly used bands. Class-touching assignment-unit counts are not additive where one original unit spans two classes; the overall358 units are unique. All370 E quality-ready objects are represented exactly once, preserving330 deterministic and40 explicit author-quality evidence levels.13 rejected products and17 deferred cases remain excluded, with no new reserve promotion or replacement.

## Fractions, integer allocation and optimality

The60/20/20 fractions are the frozen per-class object-count optimization targets, not a requirement to override exact integer/tie/group constraints. Global ideal counts are222/74/74. Actual counts and integer deviations are recorded in split_statistics.json. No post-hoc rebalance was performed to force global fractions: that would change the frozen algorithm. Every class satisfies21 train,6 validation and10 test distinct-object floors.

The independently recomputed exact squared-deviation objective is `4517/36289400`. Brute-force integer **count-only** lower bounds were calculated separately for each class; their sum equals the saved optimizer objective. The result therefore reaches the global lower bound even with the indivisible group constraints. This validation creates no second membership assignment and does not call optimize again. Frozen source/input hashes and the one invocation record establish the seeded tie/order procedure.

## Complete memberships and grouping

- object_membership.json/.csv:370 primary objects, exact class/subtype/evidence, aliases, source master hashes and role.
- image_membership.json/.csv:1110 unchanged FITS masters with repository-relative path, byte size, SHA-256, band, survey and owning object/unit/role.
- splits/train.json, validation.json and test.json plus CSV companions are the three roles of this single split. They reference masters; no images are moved or copied into tensor directories.
- assignment_unit_membership.json/.csv: all454 full-universe units, including358 active and96 inactive units.
- full_candidate_membership.json/.csv: all473 source candidates. Noneligible members may inherit a constraint role from an eligible neighbour without becoming primary observations.
- derivative_membership.json/.csv:2534 standalone derivative lineages, including1872 with eligible parents. Every derivative inherits its owning object/unit constraint. QC previews, native review crops and panels remain prohibited model inputs. Multi-object review montages are not independent astronomical observations.

All17 original nontrivial overlap components remain indivisible, including rejected/deferred/blocked/reserve bridges and the cross-class unit. Original physical identities, normalized aliases, exact eligible FITS hashes and known derivatives have zero cross-split violations. The original full473-universe closure is preserved before eligibility projection. No grouped relation was dissolved to improve class balance.

## Frozen split digest

`1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf`

split_manifest.json defines the canonical binding and SHA-256. It binds the explicit author resolution, complete adopted profile, immutable F algorithm/protocol/grouping, E quality authority, all object/image/unit/full-candidate/derivative memberships, role manifests, grouping/optimality evidence and mandatory representation/control roles. Every membership file has a byte size and SHA-256 in that binding. split_digest.txt and split_freeze_receipt.json repeat the digest. The broader file_inventory.json additionally covers this report, execution and preservation evidence.

## Preservation, validation and stop

All9,145 A–F files were hashed at start and verified unchanged at completion. All1110 referenced master image hashes/sizes were independently verified without decoding pixels. Initial/final restricted Git status is recorded and outside this G directory is unchanged. The known expected preexisting completed-author-copy change in D was already bound by E; no old inventory/report/template is rewritten.

Result-only validation reuses the one immutable optimizer result and creates no alternative split. All memberships are unique, exhaustive and group-isolated; all adopted floors and exact rational targets/objective are verified. No tensors, FITS-to-tensor processing, new data, model training/inference, checkpoint access, model outcome inspection, network, manuscript edit, staging, commit or push occurred.

The first manifest-export attempt stopped because D review-asset derivative rows contain additional review_id/ownership provenance columns. Its exact code and partial outputs are preserved under implementation_attempts/export01_heterogeneous_derivative_columns/ and bound by export_correction_record.json. The correction forms a deterministic union of CSV columns. Both export attempts use the identical saved optimizer result; optimize was never invoked again and no role, scientific policy, source artifact or membership value was changed. Only one canonical split manifest was created and frozen.

Stop after this split freeze and author review. Tensor materialization, training and evaluation require separate authorization. The final inventory SHA-256 is in inventory_receipt.json and the final response; the inventory and its receipt exclude themselves to avoid circular hashes.
