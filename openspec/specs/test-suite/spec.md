# test-suite Specification

## Purpose
TBD - created by archiving change timegenrec-credibility-closed-loop. Update Purpose after archive.
## Requirements
### Requirement: SHALL Test suite must cover dataset loading
The system SHALL Tests in `tests/test_dataset.py` must verify SeqRecDataset and EvalDataset correctness.

#### Scenario: SeqRecDataset returns correct shapes and padding
- **When** SeqRecDataset is instantiated with max_len=10
- **Then** all samples have hist.shape=(10,), pos.shape=(10,), target is scalar, and padding zeros precede the actual history

#### Scenario: EvalDataset separates last item as target
- **When** EvalDataset is given a sequence [1,2,3,4,5]
- **Then** history is [1,2,3,4] (plus padding) and target is 5

#### Scenario: EvalDataset with timestamps returns extended tuple
- **When** timestamps are provided
- **Then** __getitem__ returns a 5-element tuple including time_deltas matrix

### Requirement: SHALL Test suite must cover evaluation metrics
The system SHALL Tests in `tests/test_metrics.py` must verify recall_at_k, ndcg_at_k, and mrr_at_k edge cases.

#### Scenario: Empty ground truth returns zero
- **When** any metric is called with empty ground truth
- **Then** the result is 0.0

#### Scenario: Perfect prediction returns 1.0
- **When** the ranked list contains all ground truth items in the first positions
- **Then** recall@k and ndcg@k are 1.0

### Requirement: SHALL Test suite must cover SASRec forward pass
The system SHALL Tests in `tests/test_sasrec_forward.py` must verify output shape, numerical stability, and determinism.

#### Scenario: Output has correct shape
- **When** SASRec is called with batch_size=3, num_items=50
- **Then** scores.shape is (3, 51) including the padding item slot

#### Scenario: Output contains no NaN or Inf
- **When** SASRec forward pass completes
- **Then** all values in the scores tensor are finite

### Requirement: SHALL Test suite must cover TiSASRec time bias effect
The system SHALL Tests in `tests/test_tisasrec_time_bias.py` must verify that time_deltas affect model output.

#### Scenario: Non-zero time deltas produce different scores than zero
- **When** TiSASRec is called with non-zero vs zero time_deltas
- **Then** the output scores are different (time bias has an effect)

#### Scenario: Time discretization produces correct bucket indices
- **When** discretize_time_delta is called with known time values
- **Then** the bucket indices match the expected bucket boundaries

### Requirement: SHALL Test suite must cover pipeline end-to-end
The system SHALL Tests in `tests/test_pipeline_smoke.py` must verify that all pipeline stages run without errors.

#### Scenario: Full pipeline completes successfully
- **When** PipelineRunner with recall + pre-rank + rank + re-rank stages is run
- **Then** all stages complete without exceptions and every user has candidates

#### Scenario: Multi-recall merge has no duplicates
- **When** two recall stages are added to the pipeline
- **Then** merged candidates contain no duplicate (user, item) pairs

