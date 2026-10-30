# AstroClassifier Dataset-v2

The [public Dataset-v2 release](https://github.com/epmarinho/astroclassifier-repro/releases/tag/dataset-v2.0.0)
is distributed separately from the Git repository so that the source history
remains small. It contains two versioned assets:

- `astroclassifier-dataset-v2-fits.tar.zst`: 1,419 three-band DSS2 FITS products
  for 473 catalogue candidates, including the 60 pilot products.
- `astroclassifier-dataset-v2-tensors.tar.zst`: the 1,480 materialized NumPy
  tensors used by the frozen object-independent experiment.

The repository contains the object membership, labels, split, retrieval URLs,
provenance ledger, representation protocols, and SHA-256 indices needed to audit
both assets. Trained checkpoints are model artifacts and are not part of the
dataset release.

After downloading an asset from the GitHub release, verify it against
`RELEASE_ASSETS.sha256` and unpack it with:

```bash
tar --zstd -xf astroclassifier-dataset-v2-fits.tar.zst
tar --zstd -xf astroclassifier-dataset-v2-tensors.tar.zst
```

Read `DATA_USE.md` before redistributing or using the images. The DSS material is
copyrighted and is not covered by the Apache licence used for AstroClassifier
source code.
