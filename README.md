# Amazon ML Challenge 2026 — hanuman-Dev

A business entity resolution pipeline that links noisy Source 2 and Source 3
business records to Source 1 reference entities using names, addresses, candidate
retrieval, boosted models, and multilingual neural scoring.

**Team:** Abhay Kumar, Shaurya Bansal, Raghav Mahajan, and Piyush Lokhande.

**Selected public macro F0.5:** **0.984918**, as reported by the team for the
supplied Track FA submission. This score was not independently rerun for this
repository.

## Contents

- [Approach](#approach)
- [Repository structure](#repository-structure)
- [Requirements and setup](#requirements-and-setup)
- [Input data](#input-data)
- [Inference with the submitted models](#inference-with-the-submitted-models)
- [Training from raw data](#training-from-raw-data)
- [Configuration and stage control](#configuration-and-stage-control)
- [Outputs and validation](#outputs-and-validation)
- [Artifacts and reproducibility](#artifacts-and-reproducibility)
- [Troubleshooting](#troubleshooting)
- [Provenance and documentation](#provenance-and-documentation)

## Approach

The objective is to select a set of matching records for each reference entity.
Macro F0.5 averages the score across reference entities and gives precision more
weight than recall. Empty match sets are supported.

The pipeline proceeds through these steps:

1. **Ingest and normalize:** read the source TSVs, create internal record IDs,
   and normalize business names and addresses. Name views include legal forms,
   alternate names, domains, and Indic transliteration; address views retain
   numeric, postcode, locality, and administrative evidence.
2. **Generate candidates:** combine country-specific character-trigram retrieval
   with exact-key blocking, then use a boosted pre-ranker to reduce the candidate
   set. Two-hop expansion can add candidates when its measured gain is sufficient.
3. **Score pairs:** compute name, address, and candidate-context features; apply
   a first round of boosted models and a multilingual cross-encoder.
4. **Refine matches:** combine pair scores with cluster consensus and competing
   ownership in a second boosted round. An entity-level gate estimates whether
   any match exists.
5. **Select and export:** apply calibrated decision rules, enforce ownership
   constraints, and write matching and candidate TSVs.

The supplied Track FA run uses Track A models with France feature adaptation.
Its round-2 and gate density normalization settings differ from `final.yaml`.
Training with `final.yaml` is a separate workflow and does not establish
reproduction of the reported Track FA score.

## Repository structure

```text
.
├── README.md                         # Repository setup and workflows
├── Documentation_template.md         # Submission methodology and reported results
├── MANIFEST.json                     # Original submission package checksums
├── verify_package.py                 # Original package integrity checker
├── utils/
│   └── validate_submission.py         # Output format and consistency validator
└── code/business_entity_resolution/
    ├── README.md                     # Additional submission notes and provenance
    ├── requirements.txt              # Supplied dependency pins
    ├── configs/                      # Inherited YAML configurations
    ├── scripts/prepare_models.py     # Extract the separately supplied model archive
    └── src/ber/
        ├── blocking/                 # Retrieval, pre-ranking, and expansion
        ├── normalize/                # Name and address normalization
        ├── mining/                   # Training-derived normalization tables
        ├── features/                 # Pair, context, and consensus features
        ├── models/                   # Boosted models, cross-encoder, and gate
        ├── decision/                 # Calibration-dependent match selection
        ├── eval/                     # Metrics and diagnostics
        ├── pipeline/                 # CLI orchestration and output writing
        └── checkpoint.py             # Optional Hugging Face checkpoint sync
```

Datasets, trained artifacts, generated outputs, work directories, virtual
environments, and caches are excluded through `.gitignore`.

## Requirements and setup

The documented full-scale environment is **Python 3.12 on Linux with CUDA-capable
GPUs**. The supplied run notes describe two GPUs and approximately 30 GB of host
RAM. Budget at least 100 GB of scratch storage for intermediate files. These are
reference resources from the supplied workflow, not measured minimum requirements.

From a local clone, start at the repository root:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

The supplied requirements include Polars, NumPy, PyArrow, scikit-learn, LightGBM,
XGBoost, PyTorch, and Transformers. Dependency installation and a full GPU run
have not been revalidated for this documentation update.

All pipeline commands below run from `code/business_entity_resolution/` with the
virtual environment active and `PYTHONPATH` set as above. Configuration paths
such as `../../dataset` resolve relative to that project directory.

## Input data

Obtain the organizer-provided data separately and keep its filenames and headers:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Source files use the columns `entity_id`, `business_name`, `business_address`,
and `country`. Full training needs the training sources and ground truth as well
as the test sources. The local inference configuration processes the test split
using previously trained models and normalization tables.

## Inference with the submitted models

The model archive is **not included in GitHub**. Obtain it separately and place
it at this path relative to the repository root:

```text
code/business_entity_resolution/artifacts/hanuman-Dev_models.tar.gz
```

Then extract the models and run the local Track FA configuration:

```bash
python scripts/prepare_models.py
python -m ber.pipeline.run \
  --config configs/hanuman_dev_inference.yaml \
  --stage all
```

The extraction script writes to `work_hanuman_dev/`. The configuration uses those
local artifacts, disables remote checkpoint restoration, skips training-only
stages, and writes results to the repository's `reproduced_output/` directory.
No private checkpoint credentials are required for this workflow.

Fresh inference recomputes intermediate data that was not included in the model
archive. Byte-for-byte agreement with the historical submission is not established;
see [Artifacts and reproducibility](#artifacts-and-reproducibility).

## Training from raw data

With the organizer data in place, run:

```bash
python -m ber.pipeline.run \
  --config configs/final.yaml \
  --set checkpoint.enabled=false \
  --set paths.work_dir=work_retrain \
  --set paths.output_dir=../../retrained_output \
  --stage all
```

This workflow trains the final architecture, stores intermediate files and models
in `work_retrain/`, and writes outputs to `retrained_output/`. Neural training may
download the public `intfloat/multilingual-e5-small` initialization. Full training
requires substantial compute and disk space; no runtime estimate has been verified
for a new machine.

## Configuration and stage control

Configurations inherit and override settings in this order:

```text
default.yaml → kaggle.yaml → final.yaml → track_fa.yaml → hanuman_dev_inference.yaml
```

| Configuration | Purpose |
|---|---|
| `default.yaml` | Base pipeline, retrieval, model, and decision settings |
| `kaggle.yaml` | Resource settings for the documented two-T4 environment |
| `final.yaml` | Full training architecture and standalone paths |
| `track_fa.yaml` | Historical France adaptation with remote base-checkpoint references |
| `hanuman_dev_inference.yaml` | Local inference using the separately supplied model archive |

Use `hanuman_dev_inference.yaml` for local submitted-model inference. Running
`track_fa.yaml` directly expects access to its configured remote checkpoints.

The CLI accepts individual stages, comma-separated stage lists, and `--from` to
continue through the remaining stages. For example, after the preceding inference
stages have completed:

```bash
# Regenerate predictions and output files from existing intermediate results.
python -m ber.pipeline.run \
  --config configs/hanuman_dev_inference.yaml \
  --stage predict,outputs

# Resume from second-round scoring using existing prerequisite artifacts.
python -m ber.pipeline.run \
  --config configs/hanuman_dev_inference.yaml \
  --from r2
```

Completion markers under the work directory let most completed stages be skipped
on subsequent runs. `--force` reruns selected stages and can replace intermediate
results. Use a separate work directory for independent experiments; changing a
configuration does not automatically invalidate every completion marker.

Dotted settings can be overridden with repeated `--set key=value` arguments.
For example, `--set run.max_gpus=1` caps GPU use. Optional Hugging Face checkpoint
sync is implemented in `checkpoint.py`; the commands above keep it disabled.

## Outputs and validation

The output writer produces two tab-separated files:

| File | Columns |
|---|---|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` |

Each reference entity has one row. The candidate file records the candidate set;
selected matches must be contained in that set. The pipeline invokes the supplied
validator during output generation and logs its result. To check the result
explicitly, run:

```bash
python ../../utils/validate_submission.py \
  --matching ../../reproduced_output/matching_results.tsv \
  --candidate ../../reproduced_output/candidate_pairs.tsv \
  --test-dir ../../dataset/test
```

For a fresh training run, replace `reproduced_output` with `retrained_output`.
Submission validation checks output consistency; it does not establish leaderboard
performance.

## Artifacts and reproducibility

These original submission files are stored separately and remain ignored by Git:

| File relative to repository root | Approximate size |
|---|---:|
| `code/business_entity_resolution/artifacts/hanuman-Dev_models.tar.gz` | 746 MiB |
| `output/candidate_pairs.tsv` | 297 MiB |
| `output/matching_results.tsv` | 93 MiB |

`MANIFEST.json` records sizes and SHA-256 hashes for the original complete submission
package. With all original files present, verify it from the repository root:

```bash
python verify_package.py
```

A GitHub clone alone cannot pass this check because the large artifacts are
excluded. The manifest is a record of the original package; later changes to a
listed source or documentation file will also cause its checksum to differ.

The supplied model archive contains trained tables and models but omits the
original feature, vector, pair-score, and cross-encoder-score caches. The historical
run reused cross-encoder scores. Recomputing them can change which pairs receive
scores, so the reported public score and exact output bytes are not guaranteed by
a fresh run. The reported score, package integrity, format validation, and fresh
model evaluation are distinct checks.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named ber` | Run from `code/business_entity_resolution/` and export `PYTHONPATH="$PWD/src"`. |
| Missing model archive | Supply the archive at the documented `artifacts/` path before running `prepare_models.py`. |
| Missing model or table during inference | Extract the archive and use `hanuman_dev_inference.yaml` with `work_hanuman_dev/`. |
| Missing source TSV or column | Check the dataset layout, filenames, and organizer headers. |
| Remote checkpoint access error | Use the local inference config; `track_fa.yaml` directly references remote base runs. |
| Completed stage is skipped | Inspect the work directory's completion markers; use `--force` only when rerunning is intended. |
| GPU or host memory exhaustion | Review batch sizes and chunk/shard sizes in the config, and check available resources before restarting. |
| Package verifier fails | Restore missing original artifacts or inspect which listed file differs from the original manifest. |

## Provenance and documentation

The supplied pipeline and trained artifacts reuse the
[Mveen3/amazon-ml-challange implementation](https://github.com/Mveen3/amazon-ml-challange/).
The team name identifies the submission and does not imply original authorship of
all reused architecture or code. No upstream repository license file was present
in the supplied tree; this repository does not add a license claim for that code.

Further details:

- [Submission methodology and reported results](Documentation_template.md)
- [Pipeline notes, prior experiments, and reproduction limits](code/business_entity_resolution/README.md)
- [Local inference configuration](code/business_entity_resolution/configs/hanuman_dev_inference.yaml)
- [Original submission manifest](MANIFEST.json)
