# data-prep Specification

## Purpose
TBD - created by archiving change timegenrec-credibility-closed-loop. Update Purpose after archive.
## Requirements
### Requirement: SHALL prepare_data.py must support leave-one-out split mode
The system SHALL The data preparation script must support a leave-one-out split where each user's last interaction is the test target and the second-to-last is the validation target.

#### Scenario: Leave-one-out split produces correct splits
- **When** `prepare_data.py --dataset beauty --split-mode leave_one_out` is run
- **Then** train sequences have len-2 items removed from the end, val sequences end one item later, and test sequences contain the full user history

#### Scenario: Random split is preserved for backward compatibility
- **When** `prepare_data.py --dataset beauty --split-mode random` is run
- **Then** users are randomly split into train/val/test sets as before

### Requirement: SHALL Timestamps must be preserved and output
The system SHALL The script must extract `unixReviewTime` from raw review data and output `timestamps.pt`.

#### Scenario: Timestamps align with sequences
- **When** `timestamps.pt` is loaded
- **Then** each user's timestamp list has the same length as their item sequence, and values are monotonically increasing

### Requirement: SHALL Item categories must be extracted from metadata
The system SHALL The script must download and parse meta JSON files to extract leaf categories for each item.

#### Scenario: item_categories.pt covers all items
- **When** `item_categories.pt` is loaded
- **Then** all items that appear in sequences have a category mapping; unmatched items are logged

### Requirement: SHALL All random operations use a unified seed
The system SHALL The `--seed` parameter must control all randomization for reproducibility.

#### Scenario: Same seed produces identical splits
- **When** the script is run twice with `--seed 42`
- **Then** the resulting train/val/test splits are identical

### Requirement: SHALL Statistics must be logged and saved
The system SHALL Key dataset statistics must be printed to the console and saved to `stats.json`.

#### Scenario: Stats include key metrics
- **When** preprocessing completes
- **Then** the output includes: total users, items, categories, interactions, sparsity, avg/median/min/max sequence length

