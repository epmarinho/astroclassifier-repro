# AstroClassifier: object-independent evaluation

This repository accompanies the manuscript **Object-independent evaluation of
small-data astronomical image classifiers: architecture, representation, and
acquisition shortcuts**.

AstroClassifier compares a custom CNN--Transformer with pretrained ResNet-50 and
ViT-B/16 models on four astronomical classes. The evaluation groups derivatives
of the same physical object and reserves previously unseen objects for the final
test. It also tests common-field and central-region-masked representations, plus
metadata-only controls, to measure acquisition and context shortcuts.

## Main result

Balanced accuracy on 73 held-out objects, mean ± sample SD over three seeds:

| Model | A | B | A outer | B outer |
|---|---:|---:|---:|---:|
| Custom CNN--Transformer | 0.773 ± 0.014 | 0.704 ± 0.050 | 0.731 ± 0.019 | 0.625 ± 0.016 |
| ResNet-50 | 0.874 ± 0.054 | 0.737 ± 0.030 | 0.737 ± 0.036 | 0.608 ± 0.014 |
| ViT-B/16 | 0.866 ± 0.023 | 0.793 ± 0.015 | 0.800 ± 0.041 | 0.667 ± 0.017 |

The metadata-only decision tree reached 0.735 balanced accuracy. The strong
masked-view and metadata results show that acquisition and surrounding context
carry substantial class information. They do not establish that morphology is
unused. No pre-specified architecture contrast was significant after Holm
correction.

## Repository map

- [`paper/`](paper/) — manuscript, bibliography, figures, and figure script.
- [`rebuild/`](rebuild/) — model and analysis implementation.
- `rebuild/dataset_v2a_*` through `dataset_v2e_*` — selection, acquisition,
  provenance, and curation records.
- `rebuild/dataset_v2f_*` through `dataset_v2i_*` — representation, split, and
  training protocols.
- `rebuild/dataset_v2j1_*` and `dataset_v2j2_*` — validation search and controls.
- `rebuild/dataset_v2k_*` — frozen test plan, predictions, and aggregate scoring.
- [`RELEASE_INVENTORY.sha256`](RELEASE_INVENTORY.sha256) — checksums for every
  distributed file.

## Data and large artifacts

The Git history does not store FITS products, materialized tensors, trained
checkpoints, or the legacy internet-image corpus. DSS2 products can also be
reconstructed from the retrieval and provenance manifests, subject to the source
archive's terms. Frozen records may contain paths to omitted local artifacts;
those paths document the original execution and are not expected to resolve in a
fresh clone.

The versioned FITS and tensor datasets are available from the
[Dataset-v2 release](https://github.com/epmarinho/astroclassifier-repro/releases/tag/dataset-v2.0.0).
They are release assets rather than Git objects. Read
[`datasets/DATA_USE.md`](datasets/DATA_USE.md) for DSS attribution and
non-commercial-use conditions, and verify downloads with
[`datasets/RELEASE_ASSETS.sha256`](datasets/RELEASE_ASSETS.sha256).

## Manuscript

The compiled manuscript is available at
[`paper/cnn_transformer_article_ascom.pdf`](paper/cnn_transformer_article_ascom.pdf).
To rebuild it locally:

```bash
make -C paper
```

The scientific pipeline requires the Python packages listed in
[`requirements.txt`](requirements.txt), access to the survey products described
above, and substantial storage and GPU resources. The archived prediction and
aggregate scoring records allow verification of the reported final results
without rerunning training. The result plots can be regenerated directly; the
four-panel representation example additionally requires the omitted materialized
tensors, which can be reconstructed with the archived materialization protocol.

## Licence and citation

See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) before redistributing manuscript,
catalogue, or survey-derived material. Citation metadata are in
[`CITATION.cff`](CITATION.cff).
