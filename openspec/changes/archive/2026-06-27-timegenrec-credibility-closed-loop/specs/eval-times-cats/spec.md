## ADDED Requirements

### Requirement: SHALL EvalDataset must support timestamps and item categories
The system SHALL EvalDataset must optionally accept `timestamps` and `item_categories` to enable time-aware evaluation of TiSASRec and TiSASRec-Cat models.

#### Scenario: EvalDataset with timestamps returns time_deltas
- **When** EvalDataset is initialized with `timestamps` dict and `item_categories` dict
- **Then** `__getitem__` returns a 6-element tuple including time_deltas matrix and same_cat_mask

#### Scenario: EvalDataset without extra data is backward-compatible
- **When** EvalDataset is initialized without timestamps or item_categories
- **Then** it emits a warning and `__getitem__` returns the original 4-element tuple

### Requirement: SHALL model_eval must compute real time_deltas and same_cat_mask
The system SHALL The model_eval function must use real timestamps and categories when available, instead of zero placeholders.

#### Scenario: TiSASRec-Cat receives real time data in evaluation
- **When** model_eval is called with `item_categories` and `item_timestamps` parameters
- **Then** TiSASRec-Cat receives non-zero time_deltas and valid same_cat_mask during forward pass

#### Scenario: Warning when real data is missing
- **When** model_eval is called without timestamps for a TiSASRec model
- **Then** a `logging.warning` is emitted and zero placeholders are used as fallback

### Requirement: SHALL Helper functions for building time matrices
The system SHALL New helper functions `_build_time_deltas_from_hist` and `_build_same_cat_mask_from_hist` must construct the matrices from item-level data.

#### Scenario: Time deltas constructed from item timestamps
- **When** `_build_time_deltas_from_hist` is called with hist tensor and item_timestamps dict
- **Then** it returns a (B, L, L) tensor where each [i,j] entry is the absolute time difference in seconds
