# Municipality Flagging Usage Guide

## Overview

The municipality flagging feature automatically identifies municipalities with severe data quality issues by detecting those with >70% missing indicators. These municipalities are flagged as logical consistency anomalies to alert analysts about incomplete data that may affect analysis results.

This feature was implemented as part of Phase 2 (Quality Improvements) - Task 11.3.

## Features

### Automatic Detection
- Identifies municipalities where more than 70% of indicators are missing (configurable threshold)
- Adds flagged municipalities to logical consistency anomalies
- Provides detailed information about missing indicators

### Severity Scoring
- 90%+ missing: Severity score 95 (critical)
- 80-90% missing: Severity score 85 (high)
- 70-80% missing: Severity score 75 (medium-high)

### Detailed Reporting
- Lists all missing indicators (up to 10 shown in description)
- Shows available indicators when count is low
- Provides actionable recommendations

## Usage

### Basic Usage

The municipality flagging is automatically integrated into the `LogicalConsistencyChecker` detector:

```python
from src.anomaly_detector import LogicalConsistencyChecker
import pandas as pd

# Configuration with municipality threshold
config = {
    'thresholds': {'logical': {}},
    'missing_value_handling': {
        'municipality_threshold': 70.0  # Flag municipalities with >70% missing
    }
}

# Initialize detector
detector = LogicalConsistencyChecker(config)

# Run detection (includes municipality flagging)
anomalies_df = detector.detect(df)

# Filter for high missing municipality anomalies
high_missing = anomalies_df[
    anomalies_df['detection_method'] == 'high_missing_municipality'
]

print(f"Found {len(high_missing)} municipalities with high missing indicators")
```

### Custom Threshold

You can override the configured threshold when calling the method directly:

```python
# Use custom threshold of 80%
anomalies = detector.flag_high_missing_municipalities(df, threshold=80.0)
```

### Integration with Main Pipeline

The municipality flagging is automatically included when running the full anomaly detection pipeline:

```python
# In main.py, after loading data:

from src.detector_manager import DetectorManager

# Initialize detector manager
detector_manager = DetectorManager(config)

# Run all detectors (includes LogicalConsistencyChecker with municipality flagging)
all_anomalies = detector_manager.run_all_detectors(unified_df)

# The results will include flagged municipalities
```

## Configuration

Add or update the following section in `config.yaml`:

```yaml
# Missing value handling configuration
missing_value_handling:
  # Threshold for flagging indicators with high missing values (percentage)
  indicator_threshold: 50.0
  
  # Threshold for flagging municipalities with high missing indicators (percentage)
  municipality_threshold: 70.0
```

## Anomaly Record Structure

Flagged municipalities generate anomaly records with the following structure:

```python
{
    'anomaly_id': 'uuid',
    'territory_id': 1234,
    'municipal_name': 'Municipality Name',
    'region_name': 'Region Name',
    'indicator': 'high_missing_indicators',
    'anomaly_type': 'logical_inconsistency',
    'actual_value': 85.0,  # Percentage of missing indicators
    'expected_value': 70.0,  # Threshold
    'deviation': 15.0,  # How much threshold is exceeded
    'deviation_pct': 21.4,  # Percentage deviation from threshold
    'severity_score': 85.0,  # Based on missing percentage
    'z_score': None,
    'data_source': 'metadata',
    'detection_method': 'high_missing_municipality',
    'description': 'Municipality flagged for data quality: 34 of 40 indicators missing (85.0%)',
    'potential_explanation': 'Severe data quality issue - 85.0% of indicators are missing. Only 6 of 40 indicators have values. This municipality should be excluded from analysis or investigated for data collection issues. Missing indicators include: consumption_total, population_total, salary_avg, market_access, connection_rate, migration_total... and 28 more',
    'detected_at': datetime
}
```

## Example Output

```
2025-10-31 10:30:15 - anomaly_detector - INFO - Starting logical consistency checking
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 0 negative value anomalies
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 0 impossible ratio anomalies
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 0 contradictory indicator anomalies
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 5 unusual missing data pattern anomalies
2025-10-31 10:30:15 - anomaly_detector - INFO - Found 12 municipalities with >70% missing indicators
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 12 municipalities with high missing indicators
2025-10-31 10:30:15 - anomaly_detector - INFO - Detected 0 duplicate identifier anomalies
2025-10-31 10:30:15 - anomaly_detector - INFO - Total unique logical inconsistencies detected: 17
```

## Use Cases

### 1. Data Quality Monitoring
Identify municipalities that need data collection improvements:

```python
high_missing = anomalies_df[
    anomalies_df['detection_method'] == 'high_missing_municipality'
]

# Group by region to identify problematic areas
by_region = high_missing.groupby('region_name').size()
print("Regions with most incomplete municipalities:")
print(by_region.sort_values(ascending=False).head(10))
```

### 2. Filtering for Analysis
Exclude municipalities with insufficient data:

```python
# Get list of flagged territories
flagged_territories = high_missing['territory_id'].unique()

# Filter main dataset
clean_df = df[~df['territory_id'].isin(flagged_territories)]

print(f"Excluded {len(flagged_territories)} municipalities with insufficient data")
print(f"Remaining municipalities: {len(clean_df)}")
```

### 3. Reporting to Stakeholders
Generate reports on data completeness issues:

```python
# Create summary report
report = []
for _, anomaly in high_missing.iterrows():
    report.append({
        'Territory ID': anomaly['territory_id'],
        'Municipality': anomaly['municipal_name'],
        'Region': anomaly['region_name'],
        'Missing %': f"{anomaly['actual_value']:.1f}%",
        'Severity': 'Critical' if anomaly['severity_score'] >= 90 else 'High'
    })

report_df = pd.DataFrame(report)
report_df.to_excel('output/data_quality_issues.xlsx', index=False)
```

## Relationship with Other Features

### Task 11.1: Missingness Analysis
The missingness analysis (Task 11.1) provides the foundation by calculating missing percentages. Municipality flagging uses these calculations to identify problematic municipalities.

### Task 11.2: Indicator Filtering
Indicator filtering (Task 11.2) removes indicators with >50% missing values before analysis. Municipality flagging identifies municipalities with too many missing indicators after filtering.

### Task 11.4: Statistics Calculation
Municipalities flagged by this feature should be excluded from statistics calculations to avoid bias from incomplete data.

## Requirements Satisfied

This implementation satisfies the following requirements from the design document:

- **Requirement 11.3**: Flag municipalities with >70% missing indicators ✓
- **Requirement 11.5**: Add flagged municipalities to logical consistency anomalies ✓

## Testing

Run the tests with:

```bash
# Run all LogicalConsistencyChecker tests
pytest tests/test_detectors.py::TestLogicalConsistencyChecker -v

# Run only municipality flagging tests
pytest tests/test_detectors.py::TestLogicalConsistencyChecker::test_flag_high_missing_municipalities -v
pytest tests/test_detectors.py::TestLogicalConsistencyChecker::test_flag_high_missing_municipalities_custom_threshold -v
pytest tests/test_detectors.py::TestLogicalConsistencyChecker::test_flag_high_missing_municipalities_no_missing -v
```

All tests should pass:
- test_flag_high_missing_municipalities
- test_flag_high_missing_municipalities_custom_threshold
- test_flag_high_missing_municipalities_no_missing

## Best Practices

1. **Review Flagged Municipalities**: Always review the list of flagged municipalities to understand data collection issues
2. **Adjust Threshold**: If too many or too few municipalities are flagged, adjust the `municipality_threshold` in config
3. **Investigate Root Causes**: Use the detailed explanations to identify patterns in missing data
4. **Exclude from Analysis**: Consider excluding flagged municipalities from statistical analysis to avoid bias
5. **Report to Data Providers**: Share the list of flagged municipalities with data collection teams for improvement

## Troubleshooting

### Too Many Municipalities Flagged
If a large percentage of municipalities are flagged:
- Increase the `municipality_threshold` (e.g., from 70% to 80%)
- Check if there's a systematic data collection issue
- Review indicator filtering settings (Task 11.2)

### No Municipalities Flagged
If no municipalities are flagged but you expect some:
- Decrease the `municipality_threshold` (e.g., from 70% to 60%)
- Check if indicator filtering removed too many indicators
- Verify the data loading process

### False Positives
If municipalities are incorrectly flagged:
- Review the list of indicators being counted
- Ensure ID columns (territory_id, oktmo) are properly excluded
- Check if temporal data is being handled correctly
