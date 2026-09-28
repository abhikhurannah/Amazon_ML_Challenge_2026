# hanuman-Dev — Business Entity Resolution

Team: Abhay Kumar, Shaurya Bansal, Raghav Mahajan, Piyush Lokhande.

The packaged outputs are the exact files supplied in `results/`, corresponding to the user-reported public macro F0.5 **0.984918**. The architecture is reused from the Mveen3/amazon-ml-challange pipeline; see the provenance note below. Renaming the package does not change code or model provenance.

## Environment and data

Use Python 3.12, Linux and CUDA-capable GPUs. The supplied logs describe 2 GPUs, roughly 30 GB host RAM and substantial scratch storage (budget at least 100 GB). Preserve the pinned requirements supplied with the pipeline. Installation and the full GPU run have not been rerun during packaging.

Place the organizer-provided files at the archive root:
`dataset/train/train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`, and `dataset/test/test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`.

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python scripts/prepare_models.py
python -m ber.pipeline.run --config configs/hanuman_dev_inference.yaml --stage all
python ../../utils/validate_submission.py --matching ../../reproduced_output/matching_results.tsv --candidate ../../reproduced_output/candidate_pairs.tsv --test-dir ../../dataset/test
```

This uses the bundled tables, four-fold boosted models, two cross-encoders, calibrators and thresholds. No private checkpoint credentials are required. Outputs go to `reproduced_output/` so the submitted files remain unchanged. Package verification is available at the archive root with `python verify_package.py`.

## Configuration and exact-reproduction limits

The supplied execution log records **Track FA**, using Track A models with France feature adaptation. Round-2 and gate density normalization are false in `track_fa.yaml`; describing these submitted models as the separate Track G/final training run would be inaccurate. `hanuman_dev_inference.yaml` inherits Track FA, disabling private checkpoint restoration and using local paths.

The archive contains trained models but no original pair-score, feature, vector or cross-encoder-score caches. The logged run reused existing cross-encoder scores. Recomputing them may change which pairs have scores, so byte-for-byte equality after fresh inference is **not established**. The included output checksums identify the exact submission regardless. The evaluator can run the supplied inference path; a full exact replay may require the original work-directory caches. We do not claim that changing a threshold or replaying `final.yaml` reproduces this particular output exactly.

## Training from raw data

The reusable pipeline supports full training with:

```bash
python -m ber.pipeline.run --config configs/final.yaml --set checkpoint.enabled=false --set paths.work_dir=work_retrain --set paths.output_dir=../../retrained_output --stage all
```

This retrains the final architecture and does not claim to reproduce the historical Track FA score. It may download the public `intfloat/multilingual-e5-small` initialization. The relevant original settings are preserved in `configs/` for audit. The supplied pins and upstream licenses have not been independently reverified during packaging.

## Earlier experiments

Earlier experiments explored direct name/address evidence, premise and floor roles, multilingual normalization, France-specific handling, and residual recovery. Public results reported by the team were v2 0.970441, refined review 0.973765, country specialist 0.973944, and France recovery 0.973945. Broad residual recovery regressed to 0.973207 and was rejected. These approaches are historical experiments, not components silently blended into the selected 0.984918 output.

Later review screened 10,000 suspected rows and rejected broad deletion rules. A strict digit-1-to-letter-L recovery rule proposed 17 additions, with 9,621 true training matches and no observed false ones. This was a diagnostic, not model-conditional validation. Repeated-letter recovery had 351 false proposals out of 1,391 and was rejected. None of these unscored postprocessing variants is included in the submitted TSV. Historical code, duplicate documents and logs are omitted from this streamlined archive.

## Provenance

The supplied pipeline and trained artifacts reuse https://github.com/Mveen3/amazon-ml-challange/. The submission team is hanuman-Dev; this metadata does not imply original authorship of reused code. The pretrained model is `intfloat/multilingual-e5-small`; upstream documentation reports MIT licensing. Existing source notices and dependency identities are preserved. No upstream repository license file was present in the supplied tree; no license has been invented or independently certified here.
