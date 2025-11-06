# KeyError Root Cause Analysis - StatisticalOutlierDetector

## Executive Summary

The KeyError occurs in `detect_zscore_outliers()`, `detect_iqr_outliers()`, and `detect_percentile_outliers()` methods when attempting to access DataFrame indices that don't exist in the original DataFrame after filtering operations.

## Root Cause

### Problem Location

**File**: `src/anomaly_detector.py`  
**Methods**: 
- `detect_zscore_outliers()` (line 262-347)
- `detect_iqr_outliers()` (line 349-447) 
- `detect_percentile_outliers()` (line 449-547)

### The Issue

All three methods follow this problematic pattern:

```python
# Step 1: Extract values and drop NaN (creates new Series with filtered index)
values = df[indicator].dropna()

# Step 2: Calculate z-scores on the filtered values
z_scores = np.abs(stats.zscore(values))

# Step 3: Find outliers (returns boolean mask)
outlier_mask = z_scores > threshold
outlier_indices = values[outlier_mask].index  # These are indices from 'values', not 'df'

# Step 4: PROBLEM - Try to access df using indices from filtered 'values'
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]  # KeyError if idx not in df.index!
    z_score = z_scores[values.index.get_loc(idx)]  # Also problematic
```

### Why It Fails

1. **Index Mismatch After dropna()**: When `values = df[indicator].dropna()` is called, the resulting Series has a subset of the original DataFrame's index (only non-NaN rows).

2. **Filtered Data Processing**: The `outlier_indices` come from the `values` Series, which may have a different index than the original `df`.

3. **Direct Index Access**: The code assumes that indices from `values` exist in `df`, but this isn't guaranteed if:
   - The DataFrame has been filtered before being passed to the detector
   - There are duplicate territory_ids (mentioned in requirements)
   - The index has been reset or modified during data loading/merging

### Specific Problem Lines

#### In `detect_zscore_outliers()` (lines 310-313):
```python
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]  # KeyError here
    z_score = z_scores[values.index.get_loc(idx)]  # And here
    deviation = actual_value - mean_val
```

#### In `detect_iqr_outliers()` (lines 399-401):
```python
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]  # KeyError here
    median_val = values.median()
```

#### In `detect_percentile_outliers()` (lines 497-499):
```python
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]  # KeyError here
    percentile = stats.percentileofscore(values, actual_value)
```

## Example Scenario

Consider this scenario:

```python
# Original DataFrame
df = pd.DataFrame({
    'territory_id': [1, 2, 3, 4, 5],
    'indicator': [10, np.nan, 30, 40, 1000]
}, index=[0, 1, 2, 3, 4])

# After dropna()
values = df['indicator'].dropna()
# values.index = [0, 2, 3, 4]  (index 1 is missing)

# Calculate z-scores
z_scores = np.abs(stats.zscore(values))
# z_scores is array([1.2, 0.5, 0.3, 2.8])

# Find outliers (assuming threshold=2.0)
outlier_mask = z_scores > 2.0
outlier_indices = values[outlier_mask].index
# outlier_indices = [4]  (the value 1000)

# Now if df was filtered earlier and doesn't have index 4:
df_filtered = df.loc[[0, 1, 2]]  # Only first 3 rows

# This will cause KeyError:
for idx in outlier_indices:  # idx = 4
    actual_value = df_filtered.loc[idx, 'indicator']  # KeyError: 4
```

## Impact

This bug causes:
1. **Complete detector failure**: StatisticalOutlierDetector crashes and returns 0 anomalies
2. **Pipeline degradation**: Main pipeline continues but loses all statistical outlier detection
3. **Silent failures**: Error is logged but not immediately visible to users
4. **Data quality issues**: Missing critical anomalies that should be detected

## Evidence from Logs

From the validation report, we can see:
```
StatisticalOutlierDetector failed with error: KeyError(2103)
```

This indicates that the detector tried to access index `2103` which didn't exist in the DataFrame.

## Solution Requirements

The fix must:

1. **Use safe index access**: Check if index exists before accessing
2. **Maintain original indices**: Preserve index alignment between filtered and original data
3. **Use .loc[] consistently**: Replace direct indexing with pandas .loc[] accessor
4. **Handle edge cases**: Deal with missing indices, filtered data, and duplicate IDs
5. **Apply to all methods**: Fix z-score, IQR, and percentile methods consistently

## Recommended Fix Pattern

```python
# BEFORE (causes KeyError)
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]
    z_score = z_scores[values.index.get_loc(idx)]

# AFTER (safe access)
for idx in outlier_indices:
    # Check if index exists in original DataFrame
    if idx not in df.index:
        continue
    
    actual_value = df.loc[idx, indicator]
    
    # Safe z-score access using boolean indexing
    z_score_idx = values.index.get_loc(idx)
    z_score = z_scores[z_score_idx]
```

## Next Steps

1. Implement safe index access in all three methods
2. Add index validation before accessing DataFrame values
3. Use boolean indexing instead of direct index access where possible
4. Add regression tests with filtered DataFrames and missing indices
5. Test with duplicate territory_ids scenario

## Related Requirements

- **Requirement 1.1**: Fix StatisticalOutlierDetector KeyError
- **Requirement 1.2**: Handle duplicate territory_id values
- **Requirement 15.2**: Add regression tests for StatisticalOutlierDetector

## Files to Modify

- `src/anomaly_detector.py`: Fix all three detection methods
- `tests/test_detectors.py`: Add regression tests for KeyError scenarios
