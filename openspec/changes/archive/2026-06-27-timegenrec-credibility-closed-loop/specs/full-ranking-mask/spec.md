## ADDED Requirements

### Requirement: SHALL Full-ranking evaluation must exclude specified items
The system SHALL The evaluate_full_sort function must support an `exclude_items` parameter that sets scores of specified items to -inf before top-k ranking.

#### Scenario: Padding item 0 is excluded by default
- **When** evaluate_full_sort is called without explicit exclude_items
- **Then** item 0's scores are set to -inf, preventing it from appearing in top-K

#### Scenario: Training-interacted items are excluded
- **When** evaluate_full_sort is called with exclude_items containing training-interacted item IDs
- **Then** those items never appear in the ranked top-K, preventing inflated metrics

#### Scenario: Exclusion is post-scoring
- **When** exclude_items is provided
- **Then** the model computes scores for all items first, then excluded items are masked to -inf before top-k
- **So that** the model architecture remains unchanged

### Requirement: SHALL model_eval must pass exclude_items through to evaluate_full_sort
The system SHALL The model_eval function must accept and forward the `exclude_items` parameter.

#### Scenario: model_eval excludes training items by default
- **When** model_eval is called without explicit exclude_items
- **Then** it defaults to `{0}` (padding item) and passes this to evaluate_full_sort

#### Scenario: train_model.py collects training items for exclusion
- **When** train_model.py runs evaluation
- **Then** it collects all items that appear in the training set and passes them as exclude_items
