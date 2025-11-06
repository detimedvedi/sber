"""
Configuration Migration Demo

Demonstrates automatic migration from old configuration format to new format.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.config_validator import ConfigMigrator, migrate_config_if_needed, validate_config


def demo_old_config_format():
    """Demonstrate old configuration format (pre-Phase 1)."""
    print("=" * 80)
    print("OLD CONFIGURATION FORMAT (Pre-Phase 1)")
    print("=" * 80)
    
    old_config = {
        'thresholds': {
            'statistical': {
                'z_score': 3.0,
                'iqr_multiplier': 1.5,
                'percentile_lower': 1,
                'percentile_upper': 99
            },
            'temporal': {
                'spike_threshold': 100,
                'drop_threshold': -50,
                'volatility_multiplier': 2.0
            },
            'geographic': {
                'regional_z_score': 2.0,
                'cluster_threshold': 2.5
            },
            'cross_source': {
                'correlation_threshold': 0.5,
                'discrepancy_threshold': 50
            },
            'logical': {
                'check_negative_values': True,
                'check_impossible_ratios': True
            }
        },
        'export': {
            'output_dir': 'output',
            'timestamp_format': '%Y%m%d_%H%M%S',
            'top_n_municipalities': 50
        },
        'data_paths': {
            'sberindex': {
                'connection': 'connection.parquet',
                'consumption': 'consumption.parquet',
                'market_access': 'market_access.parquet'
            },
            'rosstat': {
                'population': 'rosstat/2_bdmo_population.parquet',
                'migration': 'rosstat/3_bdmo_migration.parquet',
                'salary': 'rosstat/4_bdmo_salary.parquet'
            },
            'municipal_dict': {
                'excel': 't_dict_municipal/t_dict_municipal_districts.xlsx'
            }
        }
    }
    
    print("\nOld config has these top-level fields:")
    for key in old_config.keys():
        print(f"  - {key}")
    
    print("\nOld config is missing new features:")
    print("  - detection_profile")
    print("  - temporal settings")
    print("  - municipality_classification")
    print("  - threshold_profiles")
    print("  - auto_tuning")
    print("  - robust_statistics")
    print("  - priority_weights")
    print("  - missing_value_handling")
    
    return old_config


def demo_format_detection(config):
    """Demonstrate format detection."""
    print("\n" + "=" * 80)
    print("FORMAT DETECTION")
    print("=" * 80)
    
    migrator = ConfigMigrator()
    is_old = migrator.is_old_format(config)
    
    print(f"\nIs old format? {is_old}")
    
    if is_old:
        print("\n✓ Old format detected!")
        print("  - Has basic required fields (thresholds, export, data_paths)")
        print("  - Missing new format indicators")
        print("  - Will be automatically migrated")
    else:
        print("\n✓ New format detected!")
        print("  - Has new format fields")
        print("  - No migration needed")
    
    return is_old


def demo_migration(old_config):
    """Demonstrate configuration migration."""
    print("\n" + "=" * 80)
    print("AUTOMATIC MIGRATION")
    print("=" * 80)
    
    migrator = ConfigMigrator()
    new_config = migrator.migrate(old_config)
    
    print("\nMigration completed!")
    print(f"\nNew config has {len(new_config)} top-level fields:")
    for key in sorted(new_config.keys()):
        if key not in old_config:
            print(f"  + {key} (NEW)")
        else:
            print(f"  - {key}")
    
    print("\nMigration warnings:")
    warnings = migrator.get_migration_warnings()
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    
    return new_config


def demo_new_features(new_config):
    """Demonstrate new features added by migration."""
    print("\n" + "=" * 80)
    print("NEW FEATURES ADDED")
    print("=" * 80)
    
    print("\n1. Detection Profile:")
    print(f"   - Profile: {new_config['detection_profile']}")
    print("   - Allows switching between strict/normal/relaxed modes")
    
    print("\n2. Temporal Settings:")
    print(f"   - Enabled: {new_config['temporal']['enabled']}")
    print(f"   - Aggregation method: {new_config['temporal']['aggregation_method']}")
    print(f"   - Auto-detect: {new_config['temporal']['auto_detect']}")
    
    print("\n3. Municipality Classification:")
    print(f"   - Enabled: {new_config['municipality_classification']['enabled']}")
    print(f"   - Urban threshold: {new_config['municipality_classification']['urban_population_threshold']:,}")
    print(f"   - Capital cities: {len(new_config['municipality_classification']['capital_cities'])} defined")
    
    print("\n4. Threshold Profiles:")
    profiles = new_config['threshold_profiles']
    print(f"   - Profiles available: {', '.join(profiles.keys())}")
    print("\n   Comparison (z_score thresholds):")
    for profile_name in ['strict', 'normal', 'relaxed']:
        z_score = profiles[profile_name]['statistical']['z_score']
        print(f"     - {profile_name:8s}: {z_score:.2f}")
    
    print("\n5. Robust Statistics:")
    print(f"   - Enabled: {new_config['robust_statistics']['enabled']}")
    print(f"   - Use median: {new_config['robust_statistics']['use_median']}")
    print(f"   - Use MAD: {new_config['robust_statistics']['use_mad']}")
    print(f"   - Winsorization limits: {new_config['robust_statistics']['winsorization_limits']}")
    
    print("\n6. Priority Weights:")
    print("   Anomaly type weights:")
    for atype, weight in new_config['priority_weights']['anomaly_types'].items():
        print(f"     - {atype:30s}: {weight:.1f}")
    
    print("\n7. Missing Value Handling:")
    print(f"   - Indicator threshold: {new_config['missing_value_handling']['indicator_threshold']}%")
    print(f"   - Municipality threshold: {new_config['missing_value_handling']['municipality_threshold']}%")
    
    print("\n8. Auto-tuning:")
    print(f"   - Enabled: {new_config['auto_tuning']['enabled']} (opt-in)")
    print(f"   - Target FPR: {new_config['auto_tuning']['target_false_positive_rate']}")
    print(f"   - Re-tuning interval: {new_config['auto_tuning']['retuning_interval_days']} days")


def demo_validation(config):
    """Demonstrate validation of migrated config."""
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    is_valid, issues = validate_config(config)
    
    print(f"\nValidation result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    if issues:
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for error in errors:
                print(f"  ✗ {error.field}: {error.message}")
        
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  ⚠ {warning.field}: {warning.message}")
    else:
        print("\n✓ No validation issues found!")


def demo_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "=" * 80)
    print("CONVENIENCE FUNCTION: migrate_config_if_needed()")
    print("=" * 80)
    
    old_config = {
        'thresholds': {
            'statistical': {'z_score': 3.0, 'iqr_multiplier': 1.5},
            'temporal': {'spike_threshold': 100},
            'geographic': {'regional_z_score': 2.0},
            'cross_source': {'correlation_threshold': 0.5},
            'logical': {'check_negative_values': True}
        },
        'export': {'output_dir': 'output'},
        'data_paths': {'sberindex': {}, 'rosstat': {}, 'municipal_dict': {}}
    }
    
    print("\nCalling migrate_config_if_needed()...")
    migrated, was_migrated = migrate_config_if_needed(old_config)
    
    print(f"\nWas migration performed? {was_migrated}")
    
    if was_migrated:
        print("✓ Config was automatically migrated to new format")
        print(f"  - Original fields: {len(old_config)}")
        print(f"  - Migrated fields: {len(migrated)}")
        print(f"  - New fields added: {len(migrated) - len(old_config)}")
    else:
        print("✓ Config was already in new format, no migration needed")


def demo_backward_compatibility():
    """Demonstrate backward compatibility."""
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY")
    print("=" * 80)
    
    print("\n✓ Old configurations continue to work:")
    print("  1. System detects old format automatically")
    print("  2. Migration happens transparently")
    print("  3. All original settings are preserved")
    print("  4. New features use sensible defaults")
    print("  5. No manual intervention required")
    
    print("\n✓ Migration is non-destructive:")
    print("  1. Original config object is not modified")
    print("  2. All original fields are preserved")
    print("  3. Only new fields are added")
    print("  4. Custom/extra fields are preserved")
    
    print("\n✓ Users can update at their own pace:")
    print("  1. Old configs work immediately")
    print("  2. Migration warnings suggest updating")
    print("  3. New features are opt-in")
    print("  4. No breaking changes")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CONFIGURATION MIGRATION DEMO" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Demo 1: Show old config format
    old_config = demo_old_config_format()
    
    # Demo 2: Detect format
    is_old = demo_format_detection(old_config)
    
    if is_old:
        # Demo 3: Perform migration
        new_config = demo_migration(old_config)
        
        # Demo 4: Show new features
        demo_new_features(new_config)
        
        # Demo 5: Validate migrated config
        demo_validation(new_config)
    
    # Demo 6: Convenience function
    demo_convenience_function()
    
    # Demo 7: Backward compatibility
    demo_backward_compatibility()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✓ Configuration migration is automatic and transparent")
    print("✓ Old configurations continue to work without changes")
    print("✓ New features are added with sensible defaults")
    print("✓ Migration warnings guide users to update their configs")
    print("✓ Full backward compatibility is maintained")
    print("\n")


if __name__ == '__main__':
    main()
