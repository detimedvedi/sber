"""
Configuration Validation Demo

Demonstrates how to use the configuration validator to ensure
configuration files are valid before processing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.config_validator import ConfigValidator, validate_config


def demo_valid_config():
    """Demonstrate validation of a valid configuration."""
    print("=" * 60)
    print("Demo 1: Validating a valid configuration")
    print("=" * 60)
    
    # Load the actual config file
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate using convenience function
    is_valid, issues = validate_config(config)
    
    print(f"\nValidation result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Total issues: {len(issues)}")
    
    if issues:
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
    
    print()


def demo_invalid_config():
    """Demonstrate validation of an invalid configuration."""
    print("=" * 60)
    print("Demo 2: Validating an invalid configuration")
    print("=" * 60)
    
    # Create an intentionally invalid config
    invalid_config = {
        'thresholds': {
            'statistical': {
                'z_score': 15.0,  # Out of range (max is 10.0)
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
                'correlation_threshold': 1.5,  # Out of range (max is 1.0)
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
            'top_n_municipalities': "50"  # Wrong type (should be int)
        }
        # Missing required field: data_paths
    }
    
    # Validate
    is_valid, issues = validate_config(invalid_config)
    
    print(f"\nValidation result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Total issues: {len(issues)}")
    
    # Show errors
    errors = [i for i in issues if i.severity == 'error']
    if errors:
        print(f"\nErrors found: {len(errors)}")
        for error in errors[:5]:  # Show first 5
            print(f"  • {error.field}")
            print(f"    {error.message}")
    
    print()


def demo_custom_profile():
    """Demonstrate validation of custom profile."""
    print("=" * 60)
    print("Demo 3: Validating a custom profile")
    print("=" * 60)
    
    config = {
        'detection_profile': 'my_custom_profile',
        'threshold_profiles': {
            'my_custom_profile': {
                'statistical': {
                    'z_score': 2.8,
                    'iqr_multiplier': 1.4,
                    'percentile_lower': 2,
                    'percentile_upper': 98
                },
                'temporal': {
                    'spike_threshold': 90,
                    'drop_threshold': -45,
                    'volatility_multiplier': 1.8
                },
                'geographic': {
                    'regional_z_score': 2.2,
                    'cluster_threshold': 2.3
                },
                'cross_source': {
                    'correlation_threshold': 0.55,
                    'discrepancy_threshold': 45
                },
                'logical': {
                    'check_negative_values': True,
                    'check_impossible_ratios': True
                }
            }
        },
        'thresholds': {
            'statistical': {'z_score': 3.0, 'iqr_multiplier': 1.5, 'percentile_lower': 1, 'percentile_upper': 99},
            'temporal': {'spike_threshold': 100, 'drop_threshold': -50, 'volatility_multiplier': 2.0},
            'geographic': {'regional_z_score': 2.0, 'cluster_threshold': 2.5},
            'cross_source': {'correlation_threshold': 0.5, 'discrepancy_threshold': 50},
            'logical': {'check_negative_values': True, 'check_impossible_ratios': True}
        },
        'export': {
            'output_dir': 'output',
            'timestamp_format': '%Y%m%d_%H%M%S',
            'top_n_municipalities': 50
        },
        'data_paths': {
            'sberindex': {'connection': 'connection.parquet', 'consumption': 'consumption.parquet', 'market_access': 'market_access.parquet'},
            'rosstat': {'population': 'rosstat/2_bdmo_population.parquet', 'migration': 'rosstat/3_bdmo_migration.parquet', 'salary': 'rosstat/4_bdmo_salary.parquet'},
            'municipal_dict': {'excel': 't_dict_municipal/t_dict_municipal_districts.xlsx', 'geopackage': 't_dict_municipal/t_dict_municipal_districts_poly.gpkg'}
        }
    }
    
    # Validate
    is_valid, issues = validate_config(config)
    
    print(f"\nValidation result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Custom profile 'my_custom_profile' is properly defined and validated")
    
    if issues:
        errors = [i for i in issues if i.severity == 'error']
        warnings = [i for i in issues if i.severity == 'warning']
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len(warnings)}")
    
    print()


def demo_validator_class():
    """Demonstrate using the ConfigValidator class directly."""
    print("=" * 60)
    print("Demo 4: Using ConfigValidator class directly")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create validator instance
    validator = ConfigValidator()
    
    # Validate
    is_valid, issues = validator.validate(config)
    
    print(f"\nValidation result: {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    # Access validator properties
    print(f"\nValidator configuration:")
    print(f"  Required fields: {len(validator.REQUIRED_FIELDS)}")
    print(f"  Required threshold categories: {len(validator.REQUIRED_THRESHOLD_CATEGORIES)}")
    print(f"  Field type specifications: {len(validator.FIELD_TYPES)}")
    print(f"  Value range specifications: {len(validator.VALUE_RANGES)}")
    print(f"  Enum specifications: {len(validator.VALID_VALUES)}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("Configuration Validation Demo")
    print("=" * 60 + "\n")
    
    demo_valid_config()
    demo_invalid_config()
    demo_custom_profile()
    demo_validator_class()
    
    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
