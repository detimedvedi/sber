# Task 15.1: Add Auto-Tuning Configuration

## Overview

Added comprehensive auto-tuning configuration section to `config.yaml` to support automatic threshold optimization functionality.

## Implementation Details

### Configuration Section Added

Added `auto_tuning` section to `config.yaml` with the following parameters:

```yaml
auto_tuning:
  # Enable/disable auto-tuning (opt-in, disabled by default)
  enabled: false
  
  # Target false positive rate (0.0 - 1.0)
  target_false_positive_rate: 0.05
  
  # Minimum expected anomalies per detector
  min_anomalies_per_detector: 10
  
  # Maximum expected anomalies per detector
  max_anomalies_per_detector: 1000
  
  # Interval for periodic re-tuning (in days)
  retuning_interval_days: 30
  
  # Minimum data points required for auto-tuning
  min_data_points: 100
  
  # Confidence level for threshold validation (0.0 - 1.0)
  validation_confidence: 0.95
  
  # Export tuned thresholds to file
  export_tuned_config: true
  export_path: "output/tuned_thresholds.yaml"
```

### Key Features

1. **Opt-in by Default**: Auto-tuning is disabled by default (`enabled: false`) to ensure backward compatibility
2. **Target FPR**: System will optimize thresholds to achieve 5% false positive rate
3. **Anomaly Count Bounds**: Ensures detectors produce between 10-1000 anomalies
4. **Periodic Re-tuning**: Automatically re-evaluates thresholds every 30 days
5. **Validation Confidence**: Ensures 95% of normal municipalities are not flagged
6. **Export Capability**: Can export tuned thresholds to a separate YAML file

### Parameter Descriptions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable/disable auto-tuning feature |
| `target_false_positive_rate` | float | `0.05` | Target FPR for optimization (0.0-1.0) |
| `min_anomalies_per_detector` | int | `10` | Minimum expected anomalies per detector |
| `max_anomalies_per_detector` | int | `1000` | Maximum expected anomalies per detector |
| `retuning_interval_days` | int | `30` | Days between automatic re-tuning |
| `min_data_points` | int | `100` | Minimum data points required for tuning |
| `validation_confidence` | float | `0.95` | Confidence level for validation (0.0-1.0) |
| `export_tuned_config` | boolean | `true` | Export tuned thresholds to file |
| `export_path` | string | `"output/tuned_thresholds.yaml"` | Path for exported config |

## Integration with AutoTuner

The configuration is fully compatible with the existing `AutoTuner` class in `src/auto_tuner.py`:

- `target_false_positive_rate` → `AutoTuner.target_fpr`
- `min_anomalies_per_detector` → `AutoTuner.min_anomalies`
- `max_anomalies_per_detector` → `AutoTuner.max_anomalies`
- `retuning_interval_days` → `AutoTuner.retuning_interval_days`

## Validation

### Configuration Validation

The configuration was validated to ensure:
- ✅ Valid YAML syntax
- ✅ All required fields present
- ✅ Correct data types (boolean, float, int, string)
- ✅ Valid value ranges (FPR: 0.0-1.0, confidence: 0.0-1.0)
- ✅ Logical constraints (max >= min anomalies)

### Test Results

All existing tests pass with the new configuration:
- ✅ `test_profile_loading.py` - 11/11 tests passed
- ✅ `test_auto_tuner_fpr.py` - 8/8 tests passed

## Usage Example

### Enable Auto-Tuning

To enable auto-tuning, modify `config.yaml`:

```yaml
auto_tuning:
  enabled: true  # Change from false to true
  target_false_positive_rate: 0.05
  # ... other parameters
```

### Adjust Target FPR

To be more or less strict:

```yaml
auto_tuning:
  enabled: true
  target_false_positive_rate: 0.03  # More strict (fewer false positives)
  # OR
  target_false_positive_rate: 0.10  # More relaxed (more detections)
```

### Change Re-tuning Frequency

```yaml
auto_tuning:
  enabled: true
  retuning_interval_days: 7  # Re-tune weekly instead of monthly
```

## Backward Compatibility

The implementation maintains full backward compatibility:
- Auto-tuning is **disabled by default**
- All parameters have sensible defaults
- Existing configurations work without modification
- AutoTuner gracefully handles missing `auto_tuning` section

## Requirements Satisfied

✅ **Requirement 6.1**: Auto-tuning configuration added to config.yaml  
✅ **Requirement 14.4**: Configuration is opt-in with `enabled: false` by default  
✅ **Task Parameters**: All required tuning parameters included (target FPR, min/max anomalies, re-tuning interval)

## Files Modified

- `config.yaml` - Added `auto_tuning` section with 9 configuration parameters

## Next Steps

This configuration enables the following tasks:
- **Task 15.2**: Implement tuning workflow using these parameters
- **Task 15.3**: Generate recommended configuration using export settings
- **Task 15.4**: Integration tests for auto-tuning pipeline

## Notes

- The configuration includes additional parameters beyond the minimum requirements (`min_data_points`, `validation_confidence`, `export_tuned_config`, `export_path`) to provide more control over the auto-tuning process
- All parameters are documented with inline comments in the config file
- The export path is relative to the workspace root and will be created automatically if it doesn't exist
