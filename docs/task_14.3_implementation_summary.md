# Task 14.3: Integrate Profiles with DetectorManager - Implementation Summary

## Overview

Task 14.3 successfully integrates threshold profiles with DetectorManager, enabling profile-based threshold management and runtime profile switching. The implementation provides three key capabilities:

1. **Profile loading on initialization** - Automatic profile loading when DetectorManager is created
2. **Profile thresholds applied to detectors** - All detectors use profile-specific thresholds
3. **Runtime profile switching** - Dynamic profile changes without restarting the system

## Implementation Details

### 1. Profile Loading on Initialization

The `DetectorManager.__init__()` method automatically loads and applies the profile specified in `config['detection_profile']`:

```python
def __init__(self, config: Dict[str, Any], source_mapping: Optional[Dict[str, str]] = None):
    self.config = config
    self.source_mapping = source_mapping or {}
    self.threshold_manager = ThresholdManager(config)
    
    # Apply profile thresholds to config before initializing detectors
    self._apply_profile_to_config()
    
    # Initialize detectors with profile-adjusted config
    self.detectors = self._initialize_detectors()
```

**Key Method: `_apply_profile_to_config()`**

This method updates `config['thresholds']` with values from the loaded profile:

```python
def _apply_profile_to_config(self):
    """Apply profile thresholds to config."""
    profile_thresholds = self.threshold_manager.profile_thresholds
    
    if profile_thresholds:
        self.config['thresholds'] = profile_thresholds
        self.logger.debug(
            f"Applied profile '{self.threshold_manager.profile}' thresholds to config"
        )
```

### 2. Profile Thresholds Applied to Detectors

When detectors are initialized via `_initialize_detectors()`, they receive the profile-adjusted config:

```python
def _initialize_detectors(self) -> Dict[str, Any]:
    """Initialize all detector instances with current config."""
    detectors = {}
    
    # Each detector is initialized with self.config, which contains
    # profile-specific thresholds after _apply_profile_to_config()
    detector = StatisticalOutlierDetector(self.config)
    detectors['statistical'] = detector
    
    # ... initialize other detectors
    
    return detectors
```

**Threshold Flow:**
1. Profile thresholds loaded by `ThresholdManager`
2. Profile thresholds merged with defaults
3. Profile thresholds applied to `config['thresholds']`
4. Detectors initialized with profile-adjusted config
5. Each detector reads thresholds from config during initialization

### 3. Runtime Profile Switching

The `switch_profile()` method enables dynamic profile changes:

```python
def switch_profile(self, profile_name: str):
    """
    Switch to a different threshold profile at runtime.
    
    Args:
        profile_name: Name of the profile ('strict', 'normal', 'relaxed')
        
    Raises:
        ValueError: If profile_name is not found in configuration
    """
    self.logger.info(
        f"Switching from profile '{self.threshold_manager.profile}' "
        f"to '{profile_name}'"
    )
    
    # Load new profile in threshold manager
    self.threshold_manager.load_profile(profile_name)
    
    # Apply new profile thresholds to config
    self._apply_profile_to_config()
    
    # Reinitialize detectors with new thresholds
    old_detector_count = len(self.detectors)
    self.detectors = self._initialize_detectors()
    new_detector_count = len(self.detectors)
    
    self.logger.info(
        f"Profile switched to '{profile_name}'. "
        f"Reinitialized {new_detector_count}/{old_detector_count} detectors."
    )
```

**Profile Switching Process:**
1. Load new profile via `ThresholdManager.load_profile()`
2. Apply new thresholds to config via `_apply_profile_to_config()`
3. Reinitialize all detectors via `_initialize_detectors()`
4. Previous detection results are preserved
5. New detections use new thresholds

## Supporting Methods

### `get_current_profile()`

Returns the name of the currently active profile:

```python
def get_current_profile(self) -> str:
    """Get the name of the currently active profile."""
    return self.threshold_manager.profile
```

### `get_profile_info()`

Returns detailed information about the current profile:

```python
def get_profile_info(self) -> Dict[str, Any]:
    """
    Get detailed information about the current profile.
    
    Returns:
        Dictionary containing:
        - profile_name: Name of current profile
        - validation: Validation results (completeness, missing params)
        - thresholds: Current threshold values for all detectors
    """
    return self.threshold_manager.get_profile_info()
```

## Configuration

Profiles are defined in `config.yaml`:

```yaml
# Active profile
detection_profile: "normal"

# Profile definitions
threshold_profiles:
  strict:
    statistical:
      z_score: 2.5
      iqr_multiplier: 1.2
    geographic:
      regional_z_score: 1.5
      cluster_threshold: 2.0
    # ... other detector thresholds
  
  normal:
    statistical:
      z_score: 3.0
      iqr_multiplier: 1.5
    geographic:
      regional_z_score: 2.0
      cluster_threshold: 2.5
    # ... other detector thresholds
  
  relaxed:
    statistical:
      z_score: 3.5
      iqr_multiplier: 2.0
    geographic:
      regional_z_score: 3.0
      cluster_threshold: 3.0
    # ... other detector thresholds
```

## Usage Examples

### Example 1: Initialize with Specific Profile

```python
# Set profile in config
config['detection_profile'] = 'strict'

# DetectorManager automatically loads and applies the profile
manager = DetectorManager(config)

# Verify profile is active
print(f"Current profile: {manager.get_current_profile()}")
# Output: Current profile: strict

# Run detection with strict thresholds
results = manager.run_all_detectors(df)
```

### Example 2: Runtime Profile Switching

```python
# Start with normal profile
config['detection_profile'] = 'normal'
manager = DetectorManager(config)

# Run detection with normal thresholds
results_normal = manager.run_all_detectors(df)
print(f"Normal profile: {len(results_normal)} anomaly sets")

# Switch to strict profile
manager.switch_profile('strict')

# Run detection again with strict thresholds
results_strict = manager.run_all_detectors(df)
print(f"Strict profile: {len(results_strict)} anomaly sets")

# Switch to relaxed profile
manager.switch_profile('relaxed')

# Run detection with relaxed thresholds
results_relaxed = manager.run_all_detectors(df)
print(f"Relaxed profile: {len(results_relaxed)} anomaly sets")
```

### Example 3: Profile Comparison

```python
# Compare detection results across profiles
profiles = ['strict', 'normal', 'relaxed']
results_by_profile = {}

manager = DetectorManager(config)

for profile in profiles:
    manager.switch_profile(profile)
    results = manager.run_all_detectors(df)
    
    total_anomalies = sum(len(df) for df in results if df is not None)
    results_by_profile[profile] = total_anomalies
    
    print(f"{profile}: {total_anomalies} anomalies")

# Output:
# strict: 100 anomalies
# normal: 53 anomalies
# relaxed: 37 anomalies
```

## Testing

Comprehensive tests verify all aspects of profile integration:

### Test Coverage

1. **Profile Loading Tests**
   - `test_profile_loaded_on_initialization` - Verifies profile loads on init
   - `test_strict_profile_loaded_on_initialization` - Tests strict profile
   - `test_relaxed_profile_loaded_on_initialization` - Tests relaxed profile

2. **Threshold Application Tests**
   - `test_profile_thresholds_applied_to_detectors` - Verifies detectors use profile thresholds
   - `test_detectors_initialized_with_profile` - Tests all detectors get correct thresholds

3. **Runtime Switching Tests**
   - `test_runtime_profile_switching` - Tests dynamic profile changes
   - `test_profile_switching_preserves_detector_count` - Ensures no detectors lost
   - `test_invalid_profile_raises_error` - Tests error handling

4. **Profile Information Tests**
   - `test_get_profile_info` - Tests profile info retrieval
   - `test_profile_info_after_switching` - Tests info updates after switch

5. **Detection Behavior Tests**
   - `test_detection_with_different_profiles` - Compares results across profiles

### Test Results

All 11 tests pass successfully:

```
tests/test_detector_manager_profiles.py::test_profile_loaded_on_initialization PASSED
tests/test_detector_manager_profiles.py::test_strict_profile_loaded_on_initialization PASSED
tests/test_detector_manager_profiles.py::test_relaxed_profile_loaded_on_initialization PASSED
tests/test_detector_manager_profiles.py::test_profile_thresholds_applied_to_detectors PASSED
tests/test_detector_manager_profiles.py::test_runtime_profile_switching PASSED
tests/test_detector_manager_profiles.py::test_invalid_profile_raises_error PASSED
tests/test_detector_manager_profiles.py::test_get_profile_info PASSED
tests/test_detector_manager_profiles.py::test_detectors_initialized_with_profile PASSED
tests/test_detector_manager_profiles.py::test_profile_switching_preserves_detector_count PASSED
tests/test_detector_manager_profiles.py::test_detection_with_different_profiles PASSED
tests/test_detector_manager_profiles.py::test_profile_info_after_switching PASSED

11 passed in 1.68s
```

## Demo Script

A comprehensive demonstration script is available at `examples/detector_manager_profiles_demo.py` that shows:

1. Initialization with different profiles
2. Threshold application to detectors
3. Runtime profile switching
4. Detection results comparison
5. Profile information retrieval

### Demo Results

The demo shows clear differences in detection behavior:

| Profile | Total Anomalies | Change from Normal |
|---------|----------------|-------------------|
| strict  | 100            | +88.7%            |
| normal  | 53             | baseline          |
| relaxed | 37             | -30.2%            |

**Breakdown by Detection Method:**

| Profile | Z-Score | Regional Outlier | Contradictory |
|---------|---------|------------------|---------------|
| strict  | 28      | 62               | 10            |
| normal  | 17      | 26               | 10            |
| relaxed | 13      | 14               | 10            |

## Benefits

### 1. Flexible Detection Sensitivity

Users can easily adjust detection sensitivity without modifying code:
- **Strict**: Comprehensive data quality audits
- **Normal**: Regular operations (balanced)
- **Relaxed**: High-confidence alerts only

### 2. Runtime Adaptability

Profile switching enables dynamic adjustment based on:
- Data quality variations
- Operational requirements
- User feedback
- Seasonal patterns

### 3. Consistent Configuration

All detectors use the same profile, ensuring:
- Consistent detection behavior
- Predictable results
- Easy troubleshooting
- Clear documentation

### 4. Easy Comparison

Runtime switching enables:
- A/B testing of thresholds
- Sensitivity analysis
- Optimal profile selection
- Performance tuning

## Integration with Other Components

### ThresholdManager

DetectorManager delegates profile management to `ThresholdManager`:
- Profile loading and validation
- Threshold merging with defaults
- Profile completeness checking
- Custom profile support

### Detectors

All detectors automatically use profile thresholds:
- `StatisticalOutlierDetector` - z_score, iqr_multiplier, percentiles
- `GeographicAnomalyDetector` - regional_z_score, cluster_threshold
- `TemporalAnomalyDetector` - spike_threshold, drop_threshold, volatility
- `CrossSourceComparator` - correlation_threshold, discrepancy_threshold
- `LogicalConsistencyChecker` - check flags

### Main Pipeline

The main pipeline can use profiles transparently:

```python
# Load config with profile
config = load_config('config.yaml')

# DetectorManager automatically uses the profile
manager = DetectorManager(config, source_mapping)

# Run detection with profile thresholds
results = manager.run_all_detectors(df)
```

## Requirements Satisfied

This implementation satisfies all requirements from task 14.3:

✅ **Load profile on initialization**
- Profile specified in `config['detection_profile']` is loaded automatically
- Profile thresholds are merged with defaults
- Profile validation ensures completeness

✅ **Apply profile thresholds to detectors**
- Profile thresholds are applied to `config['thresholds']`
- All detectors are initialized with profile-adjusted config
- Each detector uses profile-specific thresholds

✅ **Support runtime profile switching**
- `switch_profile()` method enables dynamic profile changes
- Detectors are automatically reinitialized with new thresholds
- Previous detection results are preserved
- Profile switching is logged for audit trail

✅ **Requirements 9.1, 9.2**
- Requirement 9.1: Multiple configuration profiles supported (strict, normal, relaxed)
- Requirement 9.2: Profile thresholds correctly applied to all detectors

## Files Modified

- `src/detector_manager.py` - Added profile integration methods
- `tests/test_detector_manager_profiles.py` - Comprehensive test suite
- `examples/detector_manager_profiles_demo.py` - Demonstration script
- `config.yaml` - Profile definitions

## Conclusion

Task 14.3 is complete. DetectorManager now provides full profile integration with:
- Automatic profile loading on initialization
- Profile thresholds applied to all detectors
- Runtime profile switching capability
- Comprehensive testing and documentation

The implementation enables flexible, dynamic threshold management while maintaining backward compatibility and ease of use.
