# Configuration Migration Guide

## Overview

Starting with Phase 1 of the anomaly detection improvements, the system supports an enhanced configuration format with new features. The system automatically detects and migrates old configuration files to the new format, ensuring full backward compatibility.

## What's New in the Configuration Format

The new configuration format adds several sections for enhanced functionality:

### New Top-Level Sections

1. **`detection_profile`** - Switch between strict/normal/relaxed detection modes
2. **`temporal`** - Settings for temporal data analysis
3. **`municipality_classification`** - Municipality type classification settings
4. **`threshold_profiles`** - Pre-defined threshold profiles
5. **`auto_tuning`** - Automatic threshold optimization settings
6. **`robust_statistics`** - Robust statistical methods configuration
7. **`priority_weights`** - Anomaly prioritization weights
8. **`missing_value_handling`** - Missing value handling thresholds

### Enhanced Existing Sections

- **`export`** - Added fields for executive summary and dashboard generation
- **`data_processing`** - Added `min_data_completeness` field

## Automatic Migration

### How It Works

The system automatically detects old configuration format and migrates it:

1. **Detection**: System checks for new format indicators
2. **Migration**: Adds new fields with sensible defaults
3. **Validation**: Ensures migrated config is valid
4. **Logging**: Warns about migration and suggests updating

### Migration Process

```python
from src.config_validator import migrate_config_if_needed

# Load your config (old or new format)
config = load_config('config.yaml')

# Automatically migrate if needed
migrated_config, was_migrated = migrate_config_if_needed(config)

if was_migrated:
    print("Config was automatically migrated to new format")
```

### What Gets Added

When migrating an old config, the system adds:

#### 1. Detection Profile
```yaml
detection_profile: "normal"  # strict, normal, or relaxed
```

#### 2. Temporal Settings
```yaml
temporal:
  enabled: false
  aggregation_method: "latest"  # latest, mean, or median
  auto_detect: true
```

#### 3. Municipality Classification
```yaml
municipality_classification:
  enabled: true
  urban_population_threshold: 50000
  capital_cities:
    - "Москва"
    - "Санкт-Петербург"
    # ... more cities
```

#### 4. Threshold Profiles
```yaml
threshold_profiles:
  strict:
    statistical:
      z_score: 2.4  # 80% of normal
      # ... other thresholds
  
  normal:
    statistical:
      z_score: 3.0  # Same as base thresholds
      # ... other thresholds
  
  relaxed:
    statistical:
      z_score: 3.6  # 120% of normal
      # ... other thresholds
```

#### 5. Robust Statistics
```yaml
robust_statistics:
  enabled: true
  use_median: true
  use_mad: true
  winsorization_limits: [0.01, 0.99]
  log_transform_skewness_threshold: 2.0
```

#### 6. Priority Weights
```yaml
priority_weights:
  anomaly_types:
    logical_inconsistency: 1.5
    cross_source_discrepancy: 1.2
    temporal_anomaly: 1.1
    statistical_outlier: 1.0
    geographic_anomaly: 0.8
  
  indicators:
    population: 1.3
    consumption_total: 1.2
    salary: 1.1
    default: 1.0
```

#### 7. Missing Value Handling
```yaml
missing_value_handling:
  indicator_threshold: 50.0  # Skip indicators with >50% missing
  municipality_threshold: 70.0  # Flag municipalities with >70% missing
```

#### 8. Auto-tuning
```yaml
auto_tuning:
  enabled: false  # Opt-in feature
  target_false_positive_rate: 0.05
  min_anomalies_per_detector: 10
  max_anomalies_per_detector: 1000
  retuning_interval_days: 30
  min_data_points: 100
  validation_confidence: 0.95
  export_tuned_config: true
  export_path: "output/tuned_thresholds.yaml"
```

## Backward Compatibility

### Guaranteed Compatibility

✅ **Old configurations work without changes**
- System automatically detects old format
- Migration happens transparently
- No manual intervention required

✅ **All original settings preserved**
- Existing thresholds unchanged
- Export settings maintained
- Data paths preserved
- Custom fields kept

✅ **New features use sensible defaults**
- Conservative default values
- Opt-in for advanced features
- No breaking changes

### Migration Warnings

When an old config is detected, you'll see warnings like:

```
WARNING - Config migration: Added new field 'detection_profile' with default value
WARNING - Config migration: Added new field 'temporal' with default value
WARNING - Config migration: Generated threshold profiles based on existing thresholds
WARNING - Configuration has been automatically migrated from old format. 
          8 changes were made. Consider updating your config.yaml to the new format.
```

These warnings are informational and don't indicate errors.

## Updating Your Configuration

While automatic migration works seamlessly, we recommend updating your `config.yaml` to the new format for better control and clarity.

### Step 1: Run the Demo

```bash
python examples/config_migration_demo.py
```

This shows exactly what gets added during migration.

### Step 2: Review the Current Config

Look at the current `config.yaml` in the repository for a complete example of the new format.

### Step 3: Update Your Config

Add the new sections to your config file. You can:

1. **Copy from template**: Use `config.yaml` as a template
2. **Add incrementally**: Add sections as you need them
3. **Keep defaults**: Use the default values shown above

### Step 4: Validate

```python
from src.config_validator import validate_config

config = load_config('config.yaml')
is_valid, issues = validate_config(config)

if not is_valid:
    for issue in issues:
        if issue.severity == 'error':
            print(f"Error: {issue.field} - {issue.message}")
```

## Testing Migration

### Unit Tests

Run the migration tests:

```bash
pytest tests/test_config_migration.py -v
```

### Integration Test

Test with your actual config:

```python
from src.config_validator import migrate_config_if_needed, validate_config
import yaml

# Load your config
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Migrate if needed
migrated, was_migrated = migrate_config_if_needed(config)

print(f"Migration performed: {was_migrated}")

# Validate
is_valid, issues = validate_config(migrated)
print(f"Valid: {is_valid}")

if issues:
    for issue in issues:
        print(f"{issue.severity.upper()}: {issue.field} - {issue.message}")
```

## FAQ

### Q: Do I need to update my config.yaml?

**A:** No, old configs work automatically. However, updating gives you better control over new features.

### Q: Will migration change my existing thresholds?

**A:** No, all existing thresholds are preserved exactly as-is. New fields are added with defaults.

### Q: What if I have custom fields in my config?

**A:** Custom fields are preserved during migration. The system only adds missing new fields.

### Q: Can I disable automatic migration?

**A:** Migration is automatic and transparent. If you want to use the old format, just don't add the new fields - the system will migrate in-memory without modifying your file.

### Q: How do I know if my config was migrated?

**A:** Check the logs for migration warnings. The system logs each field that was added.

### Q: Will future versions break my old config?

**A:** No, backward compatibility is guaranteed. Old configs will continue to work with automatic migration.

### Q: Can I mix old and new format?

**A:** Yes! If your config has some new fields but not all, the system will add only the missing ones.

### Q: How do I export a migrated config?

**A:** After migration, you can save the migrated config:

```python
import yaml
from src.config_validator import migrate_config_if_needed

# Load and migrate
with open('config.yaml', 'r', encoding='utf-8') as f:
    old_config = yaml.safe_load(f)

migrated, _ = migrate_config_if_needed(old_config)

# Save migrated version
with open('config_new.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(migrated, f, default_flow_style=False, allow_unicode=True)
```

## Examples

### Example 1: Basic Migration

```python
# Old config (minimal)
old_config = {
    'thresholds': {
        'statistical': {'z_score': 3.0},
        'temporal': {'spike_threshold': 100},
        'geographic': {'regional_z_score': 2.0},
        'cross_source': {'correlation_threshold': 0.5},
        'logical': {'check_negative_values': True}
    },
    'export': {'output_dir': 'output'},
    'data_paths': {'sberindex': {}, 'rosstat': {}, 'municipal_dict': {}}
}

# Migrate
from src.config_validator import migrate_config_if_needed
new_config, was_migrated = migrate_config_if_needed(old_config)

# Result: new_config has all 11 top-level fields
assert 'detection_profile' in new_config
assert 'temporal' in new_config
assert 'threshold_profiles' in new_config
```

### Example 2: Partial New Format

```python
# Config with some new fields
partial_config = {
    'detection_profile': 'strict',  # NEW field
    'thresholds': {...},
    'export': {...},
    'data_paths': {...}
    # Missing: temporal, municipality_classification, etc.
}

# Migrate
new_config, was_migrated = migrate_config_if_needed(partial_config)

# Result: Only missing fields are added
assert new_config['detection_profile'] == 'strict'  # Preserved
assert 'temporal' in new_config  # Added
```

### Example 3: Custom Fields Preserved

```python
# Config with custom fields
custom_config = {
    'thresholds': {...},
    'export': {...},
    'data_paths': {...},
    'my_custom_field': 'custom_value',
    'my_custom_section': {'key': 'value'}
}

# Migrate
new_config, was_migrated = migrate_config_if_needed(custom_config)

# Result: Custom fields preserved
assert new_config['my_custom_field'] == 'custom_value'
assert new_config['my_custom_section'] == {'key': 'value'}
```

## Summary

- ✅ Automatic migration ensures backward compatibility
- ✅ Old configs work without any changes
- ✅ New features added with sensible defaults
- ✅ Migration is transparent and non-destructive
- ✅ Custom fields are preserved
- ✅ Comprehensive logging and warnings
- ✅ Full validation of migrated configs

For more information, see:
- `examples/config_migration_demo.py` - Interactive demonstration
- `tests/test_config_migration.py` - Comprehensive test suite
- `src/config_validator.py` - Implementation details
