# Amazon ML Challenge 2026 — Business Entity Resolution

**Team Name:** hanuman-Dev  
**Team Members:** Abhay Kumar; Shaurya Bansal; Raghav Mahajan; Piyush Lokhande  
**Submission preparation date:** 28 September 2026  
**Selected public leaderboard macro F0.5:** 0.984918 (user-provided screenshot)  
**Selected output:** `output/matching_results.tsv`

## 1. Executive Summary

The selected submission links noisy Source 2 and Source 3 business records to Source 1 reference entities. It uses country-partitioned candidate generation, boosted pre-ranking, two rounds of boosted classification, a multilingual cross-encoder, ownership competition and an entity-level match gate. The final decision targets macro F0.5, which places greater weight on precision than recall.

The best confirmed output is preserved exactly. Its pipeline and trained artifacts were supplied in the `results/` folder and reuse the Mveen3/amazon-ml-challange implementation, with provenance recorded in `code/business_entity_resolution/README.md`. Earlier local experiments and later unsubmitted edits are distinguished below; we do not attribute all upstream architecture development to this team.

## 2. Methodology

### 2.1 Problem Analysis

The test set has 1,732,544 reference entities and includes US, India and France. Labeled training data contains US and India, while France is unseen in training. Each required reference has one output row; a record can be assigned to at most one reference. Empty match lists are permitted. The metric is mean per-reference F0.5, rather than ordinary classification accuracy.

Important sources of ambiguity are misspelled names, legal suffix variations, abbreviations, multilingual scripts, missing addresses, address component reordering, house/unit numbering, and businesses sharing names or premises. Plausible spelling differences can also be deliberate nonmatches, so string similarity alone is insufficient.

### 2.2 Solution Strategy

Normalize names and addresses into multiple views, generate a broad candidate union within each country, pre-rank the candidates, score them with contextual models, enforce one-owner competition and select an entity-level match set. Name normalization includes legal forms, word order, domains, alternate names and Indic transliteration. Address views retain components, numbers, administrative tokens and locality evidence. Generic normalization tables are mined from labeled training pairs.

## 3. Candidate Generation / Blocking Strategy

The supplied configuration combines bidirectional character-trigram TF-IDF retrieval with 192-dimensional random projections and exact-key channels. Address retrieval uses top-k 30 forward / 3 reverse; name-plus-address retrieval uses 20 / 3. Exact channels include name skeleton, sorted and concatenated names, name/locality, number/rare address token, postcode/number and exact address keys. Country partitions avoid cross-country candidates. Exact blocks are capped at 50 reference entities and 5,000 pairs.

A boosted pre-ranker retains candidates above the reported 0.0005 floor. The supplied training report gives a candidate ceiling of 0.9967069673, compared with union ceiling 0.9968652658. Two-hop expansion is gated by measured candidate-ceiling gain rather than enabled unconditionally.

The selected test output contains **22,465,264 candidate pairs** and **5,835,143 final matches**, as recorded in the original execution log. `candidate_pairs.tsv` is the actual selected run's candidate output, not a synthetic list rebuilt from final matches.

## 4. Model Architecture and Feature Engineering

The selected Kaggle configuration uses XGBoost, four entity-grouped folds and out-of-fold scores. Pair features include character/edit/token similarities over several name views, IDF-weighted coverage, legal-form agreement, domain/alternate-name evidence, address component agreement, numeric and postcode features, and edit-type signatures. Context features measure candidate ranks, margins and competing ownership.

Two cross-fitted halves of a fine-tuned `intfloat/multilingual-e5-small` cross-encoder provide neural pair evidence. Round 2 combines earlier scores with consensus from confident cluster members. An entity gate estimates whether any match exists; calibrated ownership scores and gate probabilities drive the first-member and subsequent-member rules. Source caps are five Source 2 and six Source 3 records.

**Actual run configuration:** the supplied log records `track_fa`, reusing Track A models and adapting France features. Round-2 and gate density normalization are disabled to match those models. The separate `final.yaml` / Track G architecture is present in the upstream source but is not asserted to be the generator of these selected outputs.

Reported decision parameters are kappa 0.8, minimum threshold 0, delta 0 and the heuristic selection rule for US and India. France inherits the US fallback. The presence of an expected-F0.5 implementation does not mean it was selected: the supplied thresholds specify the heuristic rule.

## 5. Results, Previous Optimizations and Error Analysis

### Confirmed and reported results

| Artifact / approach | Public macro F0.5 | Evidence / disposition |
|---|---:|---|
| Earlier v2 direct-evidence matcher | 0.970441 | User-reported result; summarized in README |
| Refined review | 0.973765 | User screenshot; superseded |
| Aggressive country specialist | 0.973944 | User screenshot; superseded |
| France reference recovery | 0.973945 | User screenshot; superseded |
| Learned residual recovery | 0.973207 | Regressed; rejected |
| Supplied Track FA pipeline | **0.984918** | Highest confirmed screenshot; exact outputs selected |

The supplied out-of-fold report gives training out-of-fold macro F0.5 **0.9895175582**, precision **0.9985087184**, recall **0.9704622652**, US macro F0.5 0.98982 and India 0.98907. These are supplied-run metrics, not a new evaluation executed during packaging. France has no labeled validation in that report. The training value is not substituted for the public score, and no 0.99 leaderboard result is claimed.

### Earlier optimization work

The earlier local matcher explored direct evidence, premise/unit/floor roles, name and address normalization, multilingual and France-specific handling, candidate ambiguity and conservative country-specific combinations. A role-feature model reported local validation gains, but these did not eliminate the local-to-leaderboard gap. Broad address-based residual additions lowered the reported score and were rejected. Earlier work is summarized in README; these experiments are not silently blended into the selected submission.

### Post-submission review, excluded from final outputs

A whole-file assignment audit found no duplicate target ownership, nonexistent matched IDs or cross-country assignments in the selected baseline. A systematic review screened 10,000 suspected reference rows; it was not 10,000 independent manual adjudications. Broad name-change deletion rules were rejected because many apparently suspicious patterns were correct labeled matches.

A code review found that digit 1 is normalized to i in names even when the intended character is l. A strict alternative interpretation with full-name, unique-address and number-multiset checks recovered 9,621 true labeled training pairs and no observed false ones, and proposed 17 test additions. This diagnostic includes easy training cases and is not conditional on this model's rejected predictions. A repeated-letter rule produced 351 false matches among 1,391 training proposals and was rejected. Neither the 17-addition output nor earlier 46-edit output has a confirmed higher leaderboard score, so neither is included in `output/matching_results.tsv`.

## 6. Reproducibility and Relevant Information

The ZIP includes source, pinned dependency files, original trained tables/models, actual output TSVs, a local inference configuration and a checksum verifier. The required filenames remain unchanged because the organizer specifies them. The archive and team-facing metadata use hanuman-Dev.

See `code/business_entity_resolution/README.md` for inference and retraining commands. The inference entry point uses the included models with Track FA settings and writes to a separate output directory. The supplied logs reused cached cross-encoder scores, but these intermediate score/feature caches are absent from the provided bundle. Full exact GPU replay was not performed during packaging, and byte-for-byte fresh-inference reproduction is not established. This limitation is disclosed rather than replaced by an unsupported reproduction claim.

Only artifact integrity, Python syntax, configuration loading and submission consistency are checked during packaging. The selected outputs are identical to the supplied files; the archive's manifest records their hashes. The organizer-provided raw dataset is not duplicated in the archive.

## Appendix: Included Files

- `output/`: unchanged best-confirmed matching and candidate TSVs.
- `code/business_entity_resolution/src/`: complete inference and training source.
- `code/business_entity_resolution/configs/`: only the required configuration inheritance chain.
- `code/business_entity_resolution/README.md`: reproduction instructions, prior approaches, provenance and limitations.
- `code/business_entity_resolution/requirements.txt`: supplied pinned environment.
- `code/business_entity_resolution/artifacts/hanuman-Dev_models.tar.gz`: trained models and tables, unchanged.
- `code/business_entity_resolution/scripts/prepare_models.py`: safe model extraction.
- `utils/validate_submission.py`: organizer-format validator.
- `MANIFEST.json`, `verify_package.py`: file-integrity verification.

## Packaging Validation

The organizer validator passed with both matching and candidate files supplied. Source syntax and the Track FA local configuration were checked. Full GPU inference was not rerun. Required filenames and the confirmed leaderboard output remain unchanged.
