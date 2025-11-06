# Missingness Analysis Usage Guide

## Overview

The missingness analysis feature provides comprehensive analysis of missing data patterns in the СберИндекс anomaly detection system. It helps identify:

- Indicators with high percentages of missing values
- Municipalities with incomplete data
- Overall data quality metrics

This feature was implemented as part of Phase 2 (Quality Improvements) - Task 11.1.

## Features

### 1. Missing Percentage Per Indicator
Calculates the percentage of missing values for each indicator across all municipalities.

### 2. Missing Percentage Per Municipality
Calculates the percentage of missing indicators for each municipality.

### 3. High Missing Detection
Automatically identifies:
- Indicators with >50% missing values (configurable threshold)
- Municipalities with >70% missing indicators (configurable threshold)

### 4. Overall Completeness Score
Provides an overall data completeness metric (0-1 scale).

## Usage

### Basic Usage

```python
from src.data_preprocessor import DataPreprocessor, MissingnessAnalyzer
import pandas as pd

# Option 1: Using DataPreprocessor (recommended)
config = {
    'missing_value_handling': {
        'indicator_threshold': 50.0,      # Flag indicators with >50% missing
        'municipality_threshold': 70.0     # Flag municipalities with >70% missing
    }
}

preprocessor = DataPreprocessor(config)

# Analyze missingness
report = preprocessor.analyze_missingness(df)

# Access results
print(f"Overall completeness: {report.overall_completeness:.2%}")
print(f"Indicators with high missing: {len(report.indicators_with_high_missing)}")
print(f"Municipalities with high missing: {len(report.municipalities_with_high_missing)}")

# Option 2: Using MissingnessAnalyzer directly
analyzer = MissingnessAnalyzer(
    high_missing_indicator_threshold=50.0,
    high_missing_municipality_threshold=70.0
)

report = analyzer.analyze(df)
```

### Accessing Report Data

```python
# Get missing percentage for specific indicator
missing_pct = report.missing_pct_per_indicator['consumption_total']
print(f"consumption_total: {missing_pct}% missing")

# Get missing percentage for specific municipality
territory_id = 1234
missing_pct = report.missing_pct_per_municipality[territory_id]
print(f"Territory {territory_id}: {missing_pct}% missing indicators")

# List all indicators with high missing values
for indicator in report.indicators_with_high_missing:
    pct = report.missing_pct_per_indicator[indicator]
    print(f"  {indicator}: {pct}% missing")

# List all municipalities with high missing values
for territory_id in report.municipalities_with_high_missing:
    pct = report.missing_pct_per_municipality[territory_id]
    print(f"  Territory {territory_id}: {pct}% missing")
```

### Integration with Main Pipeline

The missingness analysis can be integrated into the main pipeline after data loading:

```python
# In main.py, after loading and merging data:

from src.data_preprocessor import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(config)

# Analyze missingness
missingness_report = preprocessor.analyze_missingness(unified_df)

# Log results
logger.info(f"Data completeness: {missingness_report.overall_completeness:.2%}")

if missingness_report.indicators_with_high_missing:
    logger.warning(
        f"Found {len(missingness_report.indicators_with_high_missing)} indicators with high missing values",
        extra={
            'high_missing_indicators': missingness_report.indicators_with_high_missing[:5],
            'data_quality_issue': 'high_missing_indicators'
        }
    )

if missingness_report.municipalities_with_high_missing:
    logger.warning(
        f"Found {len(missingness_report.municipalities_with_high_missing)} municipalities with high missing indicators",
        extra={
            'high_missing_municipalities_count': len(missingness_report.municipalities_with_high_missing),
            'data_quality_issue': 'high_missing_municipalities'
        }
    )
```

## Configuration

Add the following section to `config.yaml`:

```yaml
# Missing value handling configuration
missing_value_handling:
  # Threshold for flagging indicators with high missing values (percentage)
  indicator_threshold: 50.0
  
  # Threshold for flagging municipalities with high missing indicators (percentage)
  municipality_threshold: 70.0
```

## MissingnessReport Structure

The `MissingnessReport` dataclass contains:

```python
@dataclass
class MissingnessReport:
    # Dictionary mapping indicator names to missing percentage (0-100)
    missing_pct_per_indicator: Dict[str, float]
    
    # Dictionary mapping territory_id to missing percentage (0-100)
    missing_pct_per_municipality: Dict[int, float]
    
    # List of indicators with >threshold% missing values
    indicators_with_high_missing: List[str]
    
    # List of territory_ids with >threshold% missing indicators
    municipalities_with_high_missing: List[int]
    
    # Total number of indicators analyzed
    total_indicators: int
    
    # Total number of municipalities analyzed
    total_municipalities: int
    
    # Overall data completeness score (0-1)
    overall_completeness: float
```

## Example Output

```
2025-10-31 10:15:23 - data_preprocessor - INFO - Starting missingness analysis...
2025-10-31 10:15:23 - data_preprocessor - INFO - Analyzing missingness for 36 indicators
2025-10-31 10:15:23 - data_preprocessor - INFO - Missingness analysis complete:
2025-10-31 10:15:23 - data_preprocessor - INFO -   Total indicators analyzed: 36
2025-10-31 10:15:23 - data_preprocessor - INFO -   Total municipalities analyzed: 3101
2025-10-31 10:15:23 - data_preprocessor - INFO -   Overall data completeness: 84.12%
2025-10-31 10:15:23 - data_preprocessor - WARNING -   Found 3 indicators with >50% missing values
2025-10-31 10:15:23 - data_preprocessor - WARNING -   Top indicators with missing values:
2025-10-31 10:15:23 - data_preprocessor - WARNING -     salary_Финансы и страхование: 65.3% missing
2025-10-31 10:15:23 - data_preprocessor - WARNING -     consumption_luxury: 58.7% missing
2025-10-31 10:15:23 - data_preprocessor - WARNING -     market_access_premium: 52.1% missing
2025-10-31 10:15:23 - data_preprocessor - WARNING -   Found 12 municipalities with >70% missing indicators
```

## Next Steps

After implementing missingness analysis (Task 11.1), the following tasks will build upon this:

- **Task 11.2**: Implement indicator filtering (skip indicators with >50% missing)
- **Task 11.3**: Implement municipality flagging (flag municipalities with >70% missing)
- **Task 11.4**: Update statistics calculation to handle missing values properly
- **Task 11.5**: Add tests for missing value handling

## Requirements Satisfied

This implementation satisfies the following requirements from the design document:

- **Requirement 11.1**: Calculate missing percentage per indicator ✓
- **Requirement 11.3**: Calculate missing percentage per municipality ✓

## Testing

Run the tests with:

```bash
pytest tests/test_missingness_analysis.py -v
```

All 8 tests should pass:
- test_analyze_no_missing_values
- test_analyze_with_missing_values
- test_analyze_high_missing_indicators
- test_analyze_high_missing_municipalities
- test_analyze_empty_dataframe
- test_analyze_auto_detect_indicators
- test_analyze_missingness_method
- test_custom_thresholds
