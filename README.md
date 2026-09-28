# Amazon ML Challenge 2026 — hanuman-Dev

Business entity resolution pipeline, configurations, and submission documentation.
See [the pipeline README](code/business_entity_resolution/README.md) for setup,
training, inference, and provenance.

## Files stored separately

This GitHub repository contains the source code. The following files from the
submission package are excluded from Git and kept locally:

- `code/business_entity_resolution/artifacts/hanuman-Dev_models.tar.gz`
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`

To run inference with the submitted models, obtain the model archive separately
and place it at the path above before running `scripts/prepare_models.py`.
Organizer datasets must also be supplied separately under `dataset/`.

`MANIFEST.json` and `verify_package.py` describe and verify the original complete
submission package, including the excluded files. Package verification requires
those files; a GitHub clone alone is not the complete submission package.
