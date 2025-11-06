# Missing Value Handling Methodology

## Overview

This document describes the comprehensive methodology for handling missing values in the СберИндекс Anomaly Detection System. The system implements a multi-layered approach to ensure robust statistical analysis while maintaining data quality and transparency.

## Core Principles

### 1. Pairwise Deletion Strategy

The system uses **pairwise deletion** (also known as "available case analysis") for all statistical calculations. This means:

- **Definition**: For each statistical calculation, only non-missing values are used
- **Implementation**: Missing values are excluded on a per-calculation basis using `.dropna()`
- **Advantage**: Maximizes the use of available data without discarding entire records
- **Trade-off**: Different calculations may use different subsets of data

### 2. Transparency and Logging

All missing value handling decisions are logged with appropriate severity levels:

- **INFO**: Normal operations (e.g., "Calculated statistics using 2,845 non-null values")
- **WARNING**: Data quality concerns (e.g., "Indicator has 65% missing values")
- **ERROR**: Critical issues that prevent analysis

## Implementation Details

### Phase 1: Data Loading and Initial Assessment

#### Missingness Analysis

Before any statistical analysis, the system performs comprehensive missingness analysis:

```python
# Implemented in: src/data_preprocessor.py - MissingnessAnalyzer class

1. Calculate missing percentage per indicator
   - Formula: (missing_count / total_rows) * 100
   - Logged for all indicators

2. Calculate missing percentage per municipality
   - Formula: (missing_indicators / total_indicators) * 100
   - Logged for all municipalities

3. Calculate overall data completeness
   - Formula: 1 - (total_missing_cells / total_cells)
   - Provides dataset-level quality metric
```

#### Indicator Filtering

Indicators with excessive missing values are automatically filtered:

```python
# Implemented in: src/data_preprocessor.py - DataPreprocessor.filter_indicators_by_missingness()

Threshold: 50% missing values (configurable)

Process:
1. Calculate missing percentage for each indicator
2. If missing_pct > threshold:
   - Skip indicator from all anomaly detection
   - Log warning with details
   - Add to skipped_indicators list
3. Return only valid indicators for analysis

Rationale:
- Indicators with >50% missing data provide unreliable statistics
- Including them would generate false positives
- Better to exclude than to make unreliable inferences
```

#### Municipality Flagging

Municipalities with excessive missing indicators are flagged as data quality issues:

```python
# Implemented in: src/data_preprocessor.py - MissingnessAnalyzer.analyze()

Threshold: 70% missing indicators (configurable)

Process:
1. Calculate missing percentage for each municipality
2. If missing_pct > threshold:
   - Flag municipality as data quality issue
   - Add to logical consistency anomalies
   - Include in reports with explanation
3. Municipality data still used where available

Rationale:
- Municipalities with >70% missing data likely have systematic collection issues
- Flagging alerts users to data quality problems
- Partial data still valuable for available indicators
```

### Phase 2: Statistical Calculations

#### Pairwise Deletion in Practice

All statistical calculations follow this pattern:

```python
# Example from StatisticalOutlierDetector.detect_zscore_outliers()

# Step 1: Remove missing values for this specific indicator
values = df[indicator].dropna()

# Step 2: Check if sufficient data remains
if len(values) < 3:  # Minimum threshold
    continue  # Skip this indicator

# Step 3: Calculate statistics using only non-null values
mean_val = values.mean()
std_val = values.std()

# Step 4: Calculate z-scores maintaining index alignment
z_scores_array = np.abs(stats.zscore(values))
z_scores_series = pd.Series(z_scores_array, index=values.index)
```

**Key Points:**
- Each indicator is analyzed independently
- Missing values in one indicator don't affect analysis of other indicators
- Original DataFrame indices are preserved for accurate record matching
- Minimum data requirements prevent unreliable statistics

#### Robust Statistics with Missing Values

The system uses robust statistical methods that are less sensitive to missing data patterns:

```python
# Implemented in: src/data_preprocessor.py - RobustStatsCalculator

For each indicator:
1. Remove missing values: clean_values = values.dropna()

2. Check minimum data requirement:
   if len(clean_values) < 3:
       return None  # Insufficient data

3. Calculate robust statistics:
   - Median (instead of mean)
   - MAD - Median Absolute Deviation (instead of std)
   - IQR - Interquartile Range
   - Percentiles (1st, 5th, 25th, 75th, 95th, 99th)
   - Skewness

4. Return RobustStats object with:
   - All calculated statistics
   - Count of non-null values used
```

**Advantages of Robust Statistics:**
- Median is less affected by missing data patterns than mean
- MAD is more stable than standard deviation with missing values
- Percentiles provide distribution information without assumptions
- Count field documents how many values were actually used

### Phase 3: Anomaly Detection

#### Missing Value Handling by Detector Type

**1. Statistical Outlier Detector**

```python
# Implemented in: src/anomaly_detector.py - StatisticalOutlierDetector

For each indicator:
1. Skip if >50% missing values
2. Use pairwise deletion: values = df[indicator].dropna()
3. Require minimum 3 values for z-score
4. Require minimum 3 values for IQR
5. Require minimum 10 values for percentile analysis

Methods:
- Z-score: Uses mean and std of non-null values
- IQR: Uses quartiles of non-null values
- Percentile: Uses percentile ranks of non-null values
```

**2. Geographic Anomaly Detector**

```python
# Implemented in: src/anomaly_detector.py - GeographicAnomalyDetector

For each region and indicator:
1. Group by region and municipality_type
2. For each group:
   - Remove missing values: group_values = group[indicator].dropna()
   - Calculate robust statistics (median, MAD)
   - Identify outliers using robust z-scores
3. Require minimum group size for reliable comparison

Missing value impact:
- Municipalities with missing values excluded from that indicator's analysis
- Regional statistics calculated from available municipalities only
- Type-specific thresholds account for different data availability patterns
```

**3. Cross-Source Comparator**

```python
# Implemented in: src/anomaly_detector.py - CrossSourceComparator

For each indicator pair:
1. Identify valid observations:
   valid_mask = df[sber_indicator].notna() & df[rosstat_indicator].notna()
   
2. Use only complete pairs:
   valid_data = df[valid_mask]
   
3. Calculate correlation on complete pairs only

4. Detect discrepancies only where both values present

Rationale:
- Cross-source comparison requires both values
- Listwise deletion appropriate for paired comparisons
- Missing in one source doesn't invalidate the other
```

**4. Temporal Anomaly Detector**

```python
# Implemented in: src/anomaly_detector.py - TemporalAnomalyDetector

For each municipality and indicator:
1. Extract time series: ts = group[indicator].dropna()
2. Require minimum 3 time points
3. Calculate temporal statistics on available points
4. Detect spikes/drops using consecutive non-null values

Missing value handling:
- Gaps in time series are preserved (not interpolated)
- Trend calculations skip missing periods
- Volatility calculated from available observations
- Minimum data requirements prevent false positives
```

**5. Logical Consistency Checker**

```python
# Implemented in: src/anomaly_detector.py - LogicalConsistencyChecker

For each logical rule:
1. Check if required fields are present
2. Skip check if any required field is missing
3. Flag missing data patterns as separate anomaly type

Special handling:
- Missing values in logical checks treated as "unknown"
- Impossible values (negative population) flagged regardless
- Missing data patterns flagged when systematic (>70% missing)
```

### Phase 4: Results Aggregation and Reporting

#### Missing Value Documentation in Results

All anomaly records include metadata about data availability:

```python
# Anomaly record structure includes:

{
    'indicator': 'consumption_total',
    'actual_value': 1234.56,
    'expected_value': 890.12,
    'description': 'Value deviates 3.2 std from mean (calculated from 2,845 non-null values)',
    # ... other fields
}
```

#### Summary Statistics

Summary reports include data quality metrics:

```python
# Implemented in: src/results_aggregator.py - ExecutiveSummaryGenerator

Summary includes:
1. Overall data completeness score (0-1)
2. Number of indicators skipped due to missing values
3. Number of municipalities flagged for data quality
4. Missing value patterns by region
5. Recommendations for data collection improvements
```

## Configuration Options

### Missing Value Thresholds

All thresholds are configurable in `config.yaml`:

```yaml
missing_value_handling:
  # Indicator filtering threshold (percentage)
  indicator_threshold: 50.0
  
  # Municipality flagging threshold (percentage)
  municipality_threshold: 70.0
  
  # Minimum values required for statistical calculations
  min_values_for_stats: 3
  min_values_for_percentile: 10
  
  # Minimum group size for geographic comparisons
  min_group_size: 5
```

### Aggregation Methods for Temporal Data

When temporal data is aggregated, missing values are handled according to the chosen method:

```yaml
temporal:
  aggregation_method: "latest"  # Options: latest, mean, median
  
  # How each method handles missing values:
  # - latest: Uses most recent non-null value
  # - mean: Calculates mean of non-null values
  # - median: Calculates median of non-null values
```

## Best Practices

### 1. Understand Your Data

Before running analysis:
- Review missingness analysis report
- Identify systematic missing patterns
- Understand why data is missing (MCAR, MAR, MNAR)

### 2. Adjust Thresholds Appropriately

Default thresholds (50% for indicators, 70% for municipalities) are conservative:
- **Increase** if you have high-quality data and want stricter filtering
- **Decrease** if data is sparse but valuable
- Document any threshold changes in your analysis

### 3. Interpret Results with Context

When reviewing anomalies:
- Check how many values were used for statistics
- Consider if missing data pattern could explain anomaly
- Look for municipalities flagged for data quality issues
- Review skipped indicators list

### 4. Report Data Quality

Always include in reports:
- Overall data completeness score
- Number of skipped indicators
- Number of flagged municipalities
- Recommendations for data collection improvements

## Technical Implementation Notes

### Index Preservation

Critical for accurate missing value handling:

```python
# CORRECT: Preserves original indices
values = df[indicator].dropna()
z_scores_series = pd.Series(z_scores_array, index=values.index)

# INCORRECT: Loses index alignment
values = df[indicator].dropna().values  # Don't do this!
```

### Safe Index Access

Always check index existence before accessing:

```python
# CORRECT: Safe access with validation
if idx not in df.index:
    logger.warning(f"Index {idx} not found, skipping")
    continue
actual_value = df.loc[idx, indicator]

# INCORRECT: Direct access can cause KeyError
actual_value = df.loc[idx, indicator]  # May fail if idx missing
```

### Minimum Data Requirements

Different analyses require different minimum data:

| Analysis Type | Minimum Values | Rationale |
|--------------|----------------|-----------|
| Mean/Std | 3 | Need variance estimate |
| Median/MAD | 3 | Need central tendency |
| IQR | 3 | Need quartile estimates |
| Percentile | 10 | Need distribution shape |
| Correlation | 10 | Need relationship pattern |
| Time series | 3 | Need temporal pattern |
| Group comparison | 5 per group | Need group statistics |

## Validation and Testing

### Unit Tests

Missing value handling is tested in:

```
tests/test_missingness_analysis.py
tests/test_data_loader.py
tests/test_detectors.py
```

Test scenarios include:
- All values missing
- Some values missing (various percentages)
- No values missing
- Systematic missing patterns
- Random missing patterns

### Integration Tests

Full pipeline tests with missing data:

```
tests/test_error_handling_integration.py
```

Validates:
- End-to-end processing with missing values
- Correct filtering and flagging
- Accurate anomaly detection
- Complete reporting

## References

### Statistical Methods

- **Pairwise Deletion**: Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data (3rd ed.)
- **Robust Statistics**: Huber, P. J., & Ronchetti, E. M. (2009). Robust Statistics (2nd ed.)
- **Missing Data Patterns**: Schafer, J. L., & Graham, J. W. (2002). Missing data: Our view of the state of the art

### Implementation

- **Pandas Documentation**: https://pandas.pydata.org/docs/user_guide/missing_data.html
- **NumPy NaN Handling**: https://numpy.org/doc/stable/reference/routines.nan.html
- **SciPy Statistics**: https://docs.scipy.org/doc/scipy/reference/stats.html

## Changelog

### Version 1.0 (Current)

- Implemented pairwise deletion for all statistical calculations
- Added missingness analysis and reporting
- Implemented indicator filtering (>50% missing)
- Implemented municipality flagging (>70% missing)
- Added robust statistics with missing value handling
- Documented methodology comprehensively

### Future Enhancements

Potential improvements for future versions:

1. **Multiple Imputation**: For critical indicators, implement multiple imputation methods
2. **Pattern Analysis**: Detect and report systematic missing data patterns (MCAR vs MAR vs MNAR)
3. **Sensitivity Analysis**: Show how results change with different missing value thresholds
4. **Visualization**: Add missing data heatmaps and patterns to reports
5. **Recommendations**: Automated suggestions for data collection improvements

## Contact and Support

For questions about missing value handling:
- Review this documentation
- Check test files for examples
- Review log files for warnings and errors
- Consult with data quality team for systematic issues
