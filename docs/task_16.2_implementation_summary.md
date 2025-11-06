# Task 16.2: Configuration Validation Implementation

## Overview

Implemented comprehensive configuration validation for the anomaly detection system. The validator ensures configuration files have correct structure, types, and values before the system starts processing data.

## Implementation Details

### New Module: `src/config_validator.py`

Created a comprehensive configuration validator with the following capabilities:

#### 1. Schema Validation
- Validates presence of required top-level fields (`thresholds`, `export`, `data_paths`)
- Validates required threshold categories (statistical, temporal, geographic, cross_source, logical)
- Validates data source structure (sberindex, rosstat, municipal_dict)

#### 2. Type Checking
- Validates field types match specifications (str, int, float, bool)
- Supports nested field validation using dot notation (e.g., `export.top_n_municipalities`)
- Comprehensive type checking for 20+ configuration fields

#### 3. Value Range Validation
- Validates numeric values are within acceptable ranges
- Examples:
  - `z_score`: [0.0, 10.0]
  - `correlation_threshold`: [0.0, 1.0]
  - `target_false_positive_rate`: [0.0, 1.0]
  - `percentile_lower`: [0.0, 50.0]
  - `percentile_upper`: [50.0, 100.0]

#### 4. Enum Validation
- Validates enum fields have valid values
- Supported enums:
  - `detection_profile`: strict, normal, relaxed (or custom profiles)
  - `temporal.aggregation_method`: latest, mean, median
  - `logging.level`: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - `data_processing.handle_missing`: log_and_continue, raise_error, skip

#### 5. Profile Validation
- Validates threshold profiles have correct structure
- Ensures profile thresholds are within valid ranges
- Supports custom profile definitions
- Validates profile references exist

#### 6. Logical Consistency Checks
- Ensures `percentile_lower < percentile_upper`
- Ensures `winsorization_limits[0] < winsorization_limits[1]`
- Ensures `min_anomalies_per_detector < max_anomalies_per_detector`
- Warns if `drop_threshold` is positive (should typically be negative)
- Validates custom profile references

### Integration with Main Pipeline

Updated `main.py` to integrate validation:

```python
from src.config_validator import validate_config

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load and validate configuration from YAML file"""
    # Load YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    is_valid, issues = validate_config(config)
    
    if not is_valid:
        # Print errors and exit
        print("Configuration validation failed:")
        for issue in issues:
            if issue.severity == 'error':
                print(f"  ERROR - {issue.field}: {issue.message}")
        sys.exit(1)
    
    # Log warnings but continue
    warnings = [issue for issue in issues if issue.severity == 'warning']
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  WARNING - {warning.field}: {warning.message}")
    
    return config
```

### Validation Error Reporting

The validator provides detailed error messages with:
- **Field path**: Exact location of the error (e.g., `thresholds.statistical.z_score`)
- **Error message**: Clear description of the problem
- **Severity**: `error` (blocks execution) or `warning` (allows execution)

Example output:
```
Configuration validation failed:
  ERROR - thresholds.statistical.z_score: Value 15.0 is outside valid range [0.0, 10.0]
  ERROR - detection_profile: Invalid value 'invalid_profile'. Must be one of: strict, normal, relaxed
  WARNING - thresholds.temporal.drop_threshold: drop_threshold (50) should typically be negative
```

## Testing

Created comprehensive test suite in `tests/test_config_validator.py`:

### Test Coverage (28 tests)

1. **Valid Configuration Tests**
   - Valid config passes validation
   - Minimal config passes validation

2. **Required Field Tests**
   - Missing required top-level field detected
   - Missing threshold category detected
   - Missing data source detected

3. **Type Validation Tests**
   - Invalid field type detected (e.g., string instead of int)

4. **Range Validation Tests**
   - Value out of range detected (too high)
   - Value out of range detected (too low/negative)
   - Percentile bounds validated
   - Correlation threshold range validated
   - FPR range validated
   - Validation confidence range validated

5. **Enum Validation Tests**
   - Invalid detection profile detected
   - Valid detection profiles pass (strict, normal, relaxed)
   - Invalid aggregation method detected
   - Valid aggregation methods pass (latest, mean, median)

6. **Profile Validation Tests**
   - Threshold profiles validated
   - Profile value ranges checked
   - Undefined profile reference detected
   - Custom profile passes when defined

7. **Logical Consistency Tests**
   - Auto-tuning min/max consistency checked
   - Winsorization limits validated
   - Winsorization limits format validated
   - Drop threshold warning generated

8. **Helper Method Tests**
   - Nested value access works correctly
   - Missing nested values handled
   - Partial path access handled

### Test Results

```
28 passed in 0.07s
```

All tests pass successfully.

## Usage Examples

### Basic Validation

```python
from src.config_validator import validate_config
import yaml

# Load config
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Validate
is_valid, issues = validate_config(config)

if is_valid:
    print("Configuration is valid!")
else:
    print("Configuration has errors:")
    for issue in issues:
        print(f"  {issue.severity.upper()}: {issue.field} - {issue.message}")
```

### Using ConfigValidator Class

```python
from src.config_validator import ConfigValidator

validator = ConfigValidator()
is_valid, issues = validator.validate(config)

# Access errors and warnings separately
errors = [i for i in issues if i.severity == 'error']
warnings = [i for i in issues if i.severity == 'warning']

print(f"Errors: {len(errors)}, Warnings: {len(warnings)}")
```

### Validating Custom Profiles

```python
config = {
    'detection_profile': 'my_custom_profile',
    'threshold_profiles': {
        'my_custom_profile': {
            'statistical': {
                'z_score': 2.8,
                'iqr_multiplier': 1.4,
                # ... other thresholds
            },
            # ... other categories
        }
    },
    # ... rest of config
}

is_valid, issues = validate_config(config)
# Custom profile will be validated
```

## Validation Rules Summary

### Required Fields
- `thresholds` (with all 5 categories)
- `export`
- `data_paths` (with sberindex, rosstat, municipal_dict)

### Type Requirements
- Strings: `detection_profile`, `output_dir`, `timestamp_format`, `aggregation_method`, etc.
- Integers: `top_n_municipalities`, `random_seed`, `urban_population_threshold`, etc.
- Floats: `min_data_completeness`, `indicator_threshold`, `target_false_positive_rate`, etc.
- Booleans: `enabled`, `use_median`, `use_mad`, `check_negative_values`, etc.

### Value Ranges
- Z-scores: [0.0, 10.0]
- Multipliers: [0.0, 10.0]
- Percentiles: [0.0, 100.0] (with lower < upper constraint)
- Correlations: [0.0, 1.0]
- Rates/Confidences: [0.0, 1.0]
- Thresholds: [0.0, 1000.0] (varies by type)
- Intervals: [1, 365] days

### Logical Constraints
- `percentile_lower < percentile_upper`
- `winsorization_limits[0] < winsorization_limits[1]`
- `min_anomalies_per_detector < max_anomalies_per_detector`
- Custom profiles must be defined if referenced

## Benefits

1. **Early Error Detection**: Catches configuration errors before processing starts
2. **Clear Error Messages**: Provides specific, actionable error messages
3. **Type Safety**: Ensures configuration values have correct types
4. **Range Safety**: Prevents invalid threshold values
5. **Profile Validation**: Ensures custom profiles are correctly defined
6. **Graceful Warnings**: Allows execution with warnings for non-critical issues
7. **Comprehensive Coverage**: Validates all critical configuration fields

## Files Modified

1. **Created**: `src/config_validator.py` - Main validator implementation
2. **Created**: `tests/test_config_validator.py` - Comprehensive test suite
3. **Modified**: `main.py` - Integrated validation into config loading

## Requirements Satisfied

✅ **Requirement 14.1**: Configuration validation on load
- Schema validation implemented
- Required field checks implemented
- Value range validation implemented

## Next Steps

This implementation completes task 16.2. The configuration validator is now fully integrated into the system and will catch configuration errors before any data processing begins.

Potential future enhancements:
- Add validation for file path existence
- Add validation for capital cities list format
- Add validation for priority weights sum
- Add schema export for documentation
