# pipeline-observability Specification

## Purpose
TBD - created by archiving change timegenrec-credibility-closed-loop. Update Purpose after archive.
## Requirements
### Requirement: SHALL PipelineStats dataclass for per-stage metrics
The system SHALL A new `PipelineStats` dataclass must track stage name, input/output candidate counts, wall time, and optional hit rate.

#### Scenario: PipelineStats captures stage performance
- **When** a pipeline stage completes its predict method
- **Then** PipelineStats records input_candidates, output_candidates, and wall_time_ms for that stage

#### Scenario: Hit rate is computed when ground truth is available
- **When** predict_with_stats is called with a ground_truth dict
- **Then** the PipelineStats.hit_rate field contains the fraction of users whose ground truth item appears in the output candidates

### Requirement: SHALL PipelineStage.predict_with_stats method
The system SHALL Each PipelineStage must expose a `predict_with_stats` method that wraps `predict` with timing and counting.

#### Scenario: predict_with_stats times the stage execution
- **When** predict_with_stats is called
- **Then** it returns both the CandidateList result and a PipelineStats object with accurate timing

### Requirement: SHALL PipelineRunner collect_stats mode
The system SHALL PipelineRunner must accept a `collect_stats` flag that enables per-stage statistics collection.

#### Scenario: Stats collection is opt-in
- **When** PipelineRunner is created with collect_stats=False (default)
- **Then** run() returns an empty stats list, preserving backward compatibility

#### Scenario: Stats are collected when enabled
- **When** PipelineRunner is created with collect_stats=True
- **Then** run() returns a non-empty stats list with one PipelineStats per stage

### Requirement: SHALL CandidateList.total_candidates property
The system SHALL CandidateList must expose a `total_candidates` property for quick counting.

#### Scenario: total_candidates sums across all users
- **When** CandidateList has multiple users with varying candidate counts
- **Then** total_candidates equals the sum of all per-user candidate lists

### Requirement: SHALL format_stats_table for readable output
The system SHALL A `format_stats_table` function must format a list of PipelineStats into a human-readable table string.

#### Scenario: Stats table includes all columns
- **When** format_stats_table is called with a list of PipelineStats
- **Then** the output table has columns: stage name, input count, output count, wall time (ms), hit rate

