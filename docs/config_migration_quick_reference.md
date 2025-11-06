# Configuration Migration Quick Reference

## TL;DR

Old configs work automatically. No action required. System migrates transparently.

## Quick Start

```python
from src.config_validator import migrate_config_if_needed

# Your old config works as-is
config = load_config('config.yaml')

# Automatically migrate if needed
migrated, was_migrated = migrate_config_if_needed(config)

# Use migrated config
run_detection(migrated)
```

## What's Different?

### Old Format (Pre-Phase 1)
```yaml
thresholds: {...}
export: {...}
data_paths: {...}
```

### New Format (Phase 1+)
```yaml
detection_profile: "normal"      # NEW
temporal: {...}                  # NEW
municipality_classification: {...}  # NEW
threshold_profiles: {...}        # NEW
auto_tuning: {...}              # NEW
robust_statistics: {...}        # NEW
priority_weights: {...}         # NEW
missing_value_handling: {...}   # NEW

thresholds: {...}               # Same
export: {...}                   # Enhanced
data_paths: {...}               # Same
```

## API Reference

### ConfigMigrator

```python
from src.config_validator import ConfigMigrator

migrator = ConfigMigrator()

# Check format
is_old = migrator.is_old_format(config)

# Migrate
new_config = migrator.migrate(old_config)

# Get warnings
warnings = migrator.get_migration_warnings()
```

### migrate_config_if_needed()

```python
from src.config_validator import migrate_config_if_needed

# Returns (migrated_config, was_migrated)
config, migrated = migrate_config_if_needed(old_config)

if migrated:
    print("Config was migrated")
```

## Common Scenarios

### Scenario 1: Using Old Config
```python
# Your old config.yaml works without changes
config = yaml.safe_load(open('config.yaml'))
config, _ = migrate_config_if_needed(config)
# Use config normally
```

### Scenario 2: Checking If Migration Happened
```python
config, was_migrated = migrate_config_if_needed(old_config)
if was_migrated:
    logger.info("Config was automatically migrated")
```

### Scenario 3: Getting Migration Details
```python
migrator = ConfigMigrator()
if migrator.is_old_format(config):
    new_config = migrator.migrate(config)
    for warning in migrator.get_migration_warnings():
        print(warning)
```

### Scenario 4: Validating Migrated Config
```python
from src.config_validator import migrate_config_if_needed, validate_config

config, _ = migrate_config_if_needed(old_config)
is_valid, issues = validate_config(config)
assert is_valid
```

## Default Values

All new fields get sensible defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `detection_profile` | `"normal"` | Balanced detection mode |
| `temporal.enabled` | `false` | Temporal analysis off by default |
| `municipality_classification.enabled` | `true` | Type-aware comparison on |
| `robust_statistics.enabled` | `true` | Robust stats on |
| `auto_tuning.enabled` | `false` | Auto-tuning opt-in |
| `missing_value_handling.indicator_threshold` | `50.0` | Skip indicators >50% missing |
| `missing_value_handling.municipality_threshold` | `70.0` | Flag municipalities >70% missing |

## Threshold Profiles

Generated automatically from your base thresholds:

| Profile | Scaling | Use Case |
|---------|---------|----------|
| `strict` | 80% of base | More sensitive, more anomalies |
| `normal` | 100% of base | Balanced (your current thresholds) |
| `relaxed` | 120% of base | Less sensitive, fewer anomalies |

Example:
```yaml
# Your base threshold
thresholds:
  statistical:
    z_score: 3.0

# Generated profiles
threshold_profiles:
  strict:
    statistical:
      z_score: 2.4    # 80% of 3.0
  normal:
    statistical:
      z_score: 3.0    # 100% of 3.0
  relaxed:
    statistical:
      z_score: 3.6    # 120% of 3.0
```

## Migration Warnings

You'll see warnings like:
```
WARNING - Config migration: Added new field 'detection_profile' with default value
WARNING - Config migration: Generated threshold profiles based on existing thresholds
WARNING - Configuration has been automatically migrated from old format.
```

These are informational, not errors.

## Testing

```bash
# Test migration
pytest tests/test_config_migration.py -v

# Test with your config
python examples/config_migration_demo.py
```

## FAQ

**Q: Do I need to update my config?**  
A: No, old configs work automatically.

**Q: Will my thresholds change?**  
A: No, all existing thresholds are preserved.

**Q: What if I have custom fields?**  
A: They're preserved during migration.

**Q: Can I disable migration?**  
A: Migration is automatic and transparent. Just use your old config.

**Q: How do I use new features?**  
A: They're added with defaults. Update config.yaml to customize.

## More Info

- Full guide: `docs/config_migration_guide.md`
- Demo: `examples/config_migration_demo.py`
- Tests: `tests/test_config_migration.py`
- Implementation: `src/config_validator.py`
