# Dataset-v2E — author-quality decision ingestion and freeze

**Completed and validated; no benchmark, representation or split activated.** The authoritative completed CSV was copied byte-for-byte, hashed and validated before interpretation/unblinding. All70 review IDs are present exactly once; every decision is permitted; all13 rejections and17 deferrals have nonempty justifications. Author decisions and notes remain exact, with no trimming, translation, replacement or inferred completion.

## Frozen authority and preservation

- Source: `rebuild/dataset_v2d_curation_shortcut_gate/author_decisions.csv`.
- Exact frozen copy: `frozen_author_decisions.csv`, 12,835bytes; SHA-256 `36e095e5c5a4ecba7d8f25b3b9218de8c9bba8b8cdd6913c574136d080590ad0`.
- Immutable blank template: `frozen_inputs/author_decision_template.csv`; SHA-256 `800dbab083cd2608b941f09530e021072c043ef09b5a7a670dd563b5803860a4`.
- All9,050 current files of Dataset-v2A/v2B/v2C/v2D were inventoried before interpretation and verified again unchanged after ingestion.
- The only difference from the historical D inventory was the expected author completion of its intentionally editable decision copy, already present before E. D’s blank template, historical report and inventory remain unchanged. E records that transition without rewriting prior evidence.
- `decision_copy_receipt.json`, `decision_validation.json`, `decision_freeze.json` and `unblinding_receipt.json` record the ordered gate. The existing D key alone performs the join; it was not reconstructed from names, appearances, class hints or the random seed.

## Author decisions and class inventory effect

| Frozen class | Author accept | Reject signal product | Reject wrong field | Deferred | Alternative requested | D quality ready | E quality ready |
|---|---:|---:|---:|---:|---:|---:|---:|
| galaxies | 5 | 0 | 0 | 2 | 0 | 93 | 98 |
| globularclusters | 3 | 6 | 0 | 2 | 0 | 89 | 92 |
| nebulae | 11 | 2 | 0 | 13 | 0 | 74 | 85 |
| opencluster | 21 | 5 | 0 | 0 | 0 | 74 | 95 |
| **Total** | **40** | **13** | **0** | **17** | **0** | **330** | **370** |

The40 explicit author quality acceptances and330 carried-forward deterministic D cases are permanently distinct evidence levels in `accepted_inventory.json`. The latter are not retrospectively called author approvals. The combined370 are current-product quality-ready candidates in the earlier proposal, **not370 final benchmark admissions or certified object identities**. Representation and shortcut-control gates remain open.

The400 previously proposed slots now contain370 quality-ready products,13 rejected current products and17 deferred cases. No new reserve is promoted, no replacement is selected, no pending case is silently dropped, and no100-object quota is forced. The66 unused reserves and7 earlier hard-blocked products remain exactly as D documented, including reserve quality conditions. All473 frozen physical-object candidates remain accounted for in `prospective_inventory_effect.json` and `.csv`.

## Product rejection, deferral and alternative queues

`rejected_product_queue.json` contains13 reject_signal records with exact author notes and the hashes/paths of their unchanged three-band bundles. These are rejections of current products, **not permanent rejection of their astronomical objects**. The13 objects remain eligible for a separately authorized replacement-product search. No search, new band/FOV choice, download or repair is authorized or performed. Band descriptions stay in the author’s free text; no replacement specification was inferred.

`deferred_review_queue.json` preserves17 defer_ambiguous cases without converting them into acceptances, rejections or alternative-product requests. `alternative_product_queue.json` has zero explicit author requests. It references the distinct future-search eligibility of rejected signal products, rather than relabeling those decisions. No author submitted reject_wrong_or_uncentered_field; none was invented.

## Previously compromised blindness

Exactly one supplied note explicitly reports class disclosure before the decision:

- Review ID: `Q-0E8F2AB236AFD50A`; candidate `97c952b9dced93e9`; SIMBAD:MCG-06-10-003; frozen class `galaxies`, subtype `lenticular`.
- Preserved author decision: `defer_ambiguous`.
- Author note, verbatim:

> Caso visualmente difícil cuja classe foi revelada inadvertidamente antes da decisão; requer adjudicação independente e cega.

This case remains deferred for independent blind adjudication. No corrective quality outcome was inferred. The other69 notes contain no explicit report of prior compromise; that is not evidence certifying blindness. This records the author-reported exposure and does not independently reconstruct its timing or cause. The current E unblinding occurs only after the completed decision freeze and is distinct from the reported previous compromise. Full evidence is in `blinding_compromise_record.json` and the decision ledger.

## Object and footprint grouping

All17 original nontrivial footprint components (21 edges,36 participating candidates) remain indivisible, along with the454 total components including singletons. `frozen_inputs/overlap_grouping_constraints.json` is a byte-identical snapshot of D. Candidate/component links, all members including rejected/deferred/reserve/blocked cases, cross-class groups and bridge constraints are preserved. A rejected product does not dissolve its object group or disconnect an overlap component. No split was created or optimized.

## Validation, remaining gates and stop

Validated the70-ID population against the immutable blank template and review-package manifest, exact allowed decisions, required justifications, copy/freeze hashes, unique D-key joins, immutable class/subtype/role membership, exact decision/note round trips, disjoint exhaustive queues,370 unique quality-ready candidates, all473 inventory states, unchanged original overlap constraints and all previous-phase byte hashes. The blank template is preserved both in D and as a frozen E snapshot.

No source image/FITS was modified or visually re-evaluated; no model prediction or checkpoint was inspected; no benchmark metric was calculated. No representation was selected. No network, acquisition, training, inference, package installation, manuscript/historical-result modification, staging, commit or push occurred. Initial/final restricted Git status is retained and outside E is unchanged.

This phase freezes supplied author-quality decisions only. Representation, preprocessing, future leakage controls, deferred/alternative-product research and any object/component-disjoint split require their own author review and authorization. No next phase is started. The inventory SHA-256 is in `inventory_receipt.json` and the final response. `REPORT.md` is itself inventoried; the inventory and its receipt exclude themselves to avoid circular hashes.
