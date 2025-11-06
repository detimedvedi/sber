# Indicator Filtering Usage Guide

## Overview

Task 11.2 implements indicator filtering functionality that automatically skips indicators with high missing values (>50% by default) from anomaly detection analysis. This helps improve detection quality by excluding unreliable indicators.

## Implementation

The `filter_indicators_by_missingness()` method has been added to the `DataPreprocessor` class in `src/data_preprocessor.py`.

## Usage

### Basic Usage

```python
from src.data_preprocessor import DataPreprocessor

# Initialize preprocessor
config = {}
preprocessor = DataPreprocessor(config)

# Filter indicators with >50% missing values (default threshold)
valid_indicators, skipped_indicators = preprocessor.filter_indicators_by_missingness(df)

print(f"Valid indicators: {len(valid_indicators)}")
print(f"Skipped indicators: {len(skipped_indicators)}")
```

### Custom Threshold

```python
# Use custom threshold (e.g., 30%)
valid_indicators, skipped_indicators = preprocessor.filter_indicators_by_missingness(
    df,
    threshold=30.0
)
```

### Specify Indicators

```python
# Filter specific indicators only
indicators_to_check = ['consumption_total', 'salary_average', 'population_total']
valid_indicators, skipped_indicators = preprocessor.filter_indicators_by_missingness(
    df,
    indicators=indicators_to_check,
    threshold=50.0
)
```

### Auto-Detection

```python
# Automatically detect numeric indicators (excludes ID columns like territory_id, oktmo)
valid_indicators, skipped_indicators = preprocessor.filter_indicators_by_missingness(df)
```

## Logging

The method logs detailed warnings for each skipped indicator:

```
WARNING - Skipping indicator 'salary_Финансы' due to high missing values: 1850/3101 (59.7% missing)
WARNING - Skipped 5 indicators with >50% missing values. These indicators will not be included in anomaly detection.
```

Log entries include structured extra data for monitoring:
- `indicator`: Name of the skipped indicator
- `missing_count`: Number of missing values
- `total_count`: Total number of rows
- `missing_percentage`: Percentage of missing values
- `threshold`: Threshold used for filtering
- `data_quality_issue`: Set to 'high_missing_indicator'

## Return Values

The method returns a tuple:
1. **valid_indicators** (List[str]): List of indicator names that passed the filter
2. **skipped_indicators** (List[str]): List of indicator names that were skipped

## Integration with Detectors

While detectors already have built-in logic to skip indicators with >50% missing values, this centralized filtering provides:

1. **Consistent filtering**: All detectors use the same filtered list
2. **Better logging**: Warnings are logged once at the preprocessing stage
3. **Performance**: Avoids redundant missing value checks in each detector
4. **Flexibility**: Easy to adjust threshold globally

## Configuration

The filtering threshold can be configured in `config.yaml`:

```yaml
missing_value_handling:
  indicator_threshold: 50.0  # Skip indicators with >50% missing values
  municipality_threshold: 70.0  # Flag municipalities with >70% missing indicators
```

## Requirements Satisfied

This implementation satisfies **Requirement 11.2**:
- ✅ Skip indicators with >50% missing values
- ✅ Log warnings for skipped indicators
- ✅ Configurable threshold
- ✅ Comprehensive test coverage

## Testing

Run tests with:

```bash
pytest tests/test_missingness_analysis.py::TestIndicatorFiltering -v
```

Test coverage includes:
- No missing values scenario
- High missing values scenario
- Custom threshold
- Auto-detection of indicators
- Empty DataFrame handling
- Edge case: exactly at threshold

## Example Output

```
INFO - Filtering indicators with >50% missing values...
WARNING - Skipping indicator 'salary_Финансы и страхование' due to high missing values: 1850/3101 (59.7% missing)
WARNING - Skipping indicator 'salary_Образование' due to high missing values: 1620/3101 (52.2% missing)
INFO - Indicator filtering complete: 34 valid, 2 skipped
WARNING - Skipped 2 indicators with >50% missing values. These indicators will not be included in anomaly detection.
```

## Next Steps

To integrate this filtering into the main pipeline:

1. Call `filter_indicators_by_missingness()` after loading data
2. Pass the `valid_indicators` list to detectors
3. Update detector methods to use the filtered list instead of auto-detecting

This ensures consistent, high-quality anomaly detection across all detectors.
