"""
Tests for Configuration Validator

Tests comprehensive validation of configuration including:
- Schema validation
- Required field checks
- Value range validation
- Type checking
- Profile validation
"""

import pytest
import yaml
from src.config_validator import ConfigValidator, validate_config, ValidationError


@pytest.fixture
def valid_config():
    """Load valid configuration from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def minimal_config():
    """Minimal valid configuration."""
    return {
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
                'excel': 't_dict_municipal/t_dict_municipal_districts.xlsx',
                'geopackage': 't_dict_municipal/t_dict_municipal_districts_poly.gpkg'
            }
        }
    }


class TestConfigValidator:
    """Test suite for ConfigValidator class."""
    
    def test_valid_config_passes(self, valid_config):
        """Test that valid configuration passes validation."""
        validator = ConfigValidator()
        is_valid, issues = validator.validate(valid_config)
        
        assert is_valid, f"Valid config should pass validation. Issues: {issues}"
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0, f"Should have no errors, got: {errors}"
    
    def test_minimal_config_passes(self, minimal_config):
        """Test that minimal valid configuration passes."""
        validator = ConfigValidator()
        is_valid, issues = validator.validate(minimal_config)
        
        assert is_valid, f"Minimal config should pass. Issues: {issues}"
    
    def test_missing_required_field(self, minimal_config):
        """Test that missing required field is detected."""
        config = minimal_config.copy()
        del config['thresholds']
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('thresholds' in i.field and i.severity == 'error' for i in issues)
    
    def test_missing_threshold_category(self, minimal_config):
        """Test that missing threshold category is detected."""
        config = minimal_config.copy()
        del config['thresholds']['statistical']
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('statistical' in i.field and i.severity == 'error' for i in issues)
    
    def test_invalid_field_type(self, minimal_config):
        """Test that invalid field type is detected."""
        config = minimal_config.copy()
        config['export']['top_n_municipalities'] = "50"  # Should be int
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('top_n_municipalities' in i.field and i.severity == 'error' for i in issues)
    
    def test_value_out_of_range(self, minimal_config):
        """Test that out-of-range value is detected."""
        config = minimal_config.copy()
        config['thresholds']['statistical']['z_score'] = 15.0  # Max is 10.0
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('z_score' in i.field and i.severity == 'error' for i in issues)
    
    def test_negative_value_out_of_range(self, minimal_config):
        """Test that negative value outside range is detected."""
        config = minimal_config.copy()
        config['thresholds']['statistical']['z_score'] = -1.0  # Min is 0.0
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('z_score' in i.field and i.severity == 'error' for i in issues)
    
    def test_percentile_bounds(self, minimal_config):
        """Test that percentile bounds are validated."""
        config = minimal_config.copy()
        config['thresholds']['statistical']['percentile_lower'] = 60
        config['thresholds']['statistical']['percentile_upper'] = 40
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('percentile' in i.message.lower() and i.severity == 'error' for i in issues)
    
    def test_invalid_detection_profile(self, minimal_config):
        """Test that invalid detection profile is detected."""
        config = minimal_config.copy()
        config['detection_profile'] = 'invalid_profile'
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('detection_profile' in i.field and i.severity == 'error' for i in issues)
    
    def test_valid_detection_profiles(self, minimal_config):
        """Test that valid detection profiles pass."""
        for profile in ['strict', 'normal', 'relaxed']:
            config = minimal_config.copy()
            config['detection_profile'] = profile
            
            validator = ConfigValidator()
            is_valid, issues = validator.validate(config)
            
            errors = [i for i in issues if i.severity == 'error']
            assert is_valid, f"Profile '{profile}' should be valid. Errors: {errors}"
    
    def test_invalid_aggregation_method(self, minimal_config):
        """Test that invalid aggregation method is detected."""
        config = minimal_config.copy()
        config['temporal'] = {
            'enabled': True,
            'aggregation_method': 'invalid_method',
            'auto_detect': True
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('aggregation_method' in i.field and i.severity == 'error' for i in issues)
    
    def test_valid_aggregation_methods(self, minimal_config):
        """Test that valid aggregation methods pass."""
        for method in ['latest', 'mean', 'median']:
            config = minimal_config.copy()
            config['temporal'] = {
                'enabled': True,
                'aggregation_method': method,
                'auto_detect': True
            }
            
            validator = ConfigValidator()
            is_valid, issues = validator.validate(config)
            
            errors = [i for i in issues if i.severity == 'error']
            assert is_valid, f"Method '{method}' should be valid. Errors: {errors}"
    
    def test_auto_tuning_min_max_consistency(self, minimal_config):
        """Test that auto-tuning min/max anomalies are consistent."""
        config = minimal_config.copy()
        config['auto_tuning'] = {
            'enabled': True,
            'min_anomalies_per_detector': 1000,
            'max_anomalies_per_detector': 100,  # Less than min
            'target_false_positive_rate': 0.05,
            'retuning_interval_days': 30,
            'min_data_points': 100,
            'validation_confidence': 0.95,
            'export_tuned_config': True,
            'export_path': 'output/tuned.yaml'
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('min_anomalies' in i.message.lower() and i.severity == 'error' for i in issues)
    
    def test_winsorization_limits_validation(self, minimal_config):
        """Test that winsorization limits are validated."""
        config = minimal_config.copy()
        config['robust_statistics'] = {
            'enabled': True,
            'use_median': True,
            'use_mad': True,
            'winsorization_limits': [0.99, 0.01],  # Wrong order
            'log_transform_skewness_threshold': 2.0
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('winsorization' in i.field and i.severity == 'error' for i in issues)
    
    def test_winsorization_limits_wrong_format(self, minimal_config):
        """Test that winsorization limits format is validated."""
        config = minimal_config.copy()
        config['robust_statistics'] = {
            'enabled': True,
            'use_median': True,
            'use_mad': True,
            'winsorization_limits': [0.01],  # Should have 2 values
            'log_transform_skewness_threshold': 2.0
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('winsorization' in i.field and i.severity == 'error' for i in issues)
    
    def test_threshold_profiles_validation(self, minimal_config):
        """Test that threshold profiles are validated."""
        config = minimal_config.copy()
        config['threshold_profiles'] = {
            'custom': {
                'statistical': {
                    'z_score': 20.0  # Out of range
                }
            }
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('custom' in i.field and 'z_score' in i.field and i.severity == 'error' for i in issues)
    
    def test_missing_data_source(self, minimal_config):
        """Test that missing data source is detected."""
        config = minimal_config.copy()
        del config['data_paths']['rosstat']
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('rosstat' in i.field and i.severity == 'error' for i in issues)
    
    def test_drop_threshold_warning(self, minimal_config):
        """Test that positive drop threshold generates warning."""
        config = minimal_config.copy()
        config['thresholds']['temporal']['drop_threshold'] = 50  # Should be negative
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        # Should pass but with warning
        assert is_valid
        warnings = [i for i in issues if i.severity == 'warning']
        assert any('drop_threshold' in i.field for i in warnings)
    
    def test_correlation_threshold_range(self, minimal_config):
        """Test that correlation threshold is in [0, 1] range."""
        config = minimal_config.copy()
        config['thresholds']['cross_source']['correlation_threshold'] = 1.5
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('correlation_threshold' in i.field and i.severity == 'error' for i in issues)
    
    def test_fpr_range(self, minimal_config):
        """Test that false positive rate is in [0, 1] range."""
        config = minimal_config.copy()
        config['auto_tuning'] = {
            'enabled': True,
            'target_false_positive_rate': 1.5,  # Out of range
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'min_data_points': 100,
            'validation_confidence': 0.95,
            'export_tuned_config': True,
            'export_path': 'output/tuned.yaml'
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('target_false_positive_rate' in i.field and i.severity == 'error' for i in issues)
    
    def test_validation_confidence_range(self, minimal_config):
        """Test that validation confidence is in [0, 1] range."""
        config = minimal_config.copy()
        config['auto_tuning'] = {
            'enabled': True,
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'min_data_points': 100,
            'validation_confidence': 2.0,  # Out of range
            'export_tuned_config': True,
            'export_path': 'output/tuned.yaml'
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('validation_confidence' in i.field and i.severity == 'error' for i in issues)
    
    def test_undefined_profile_reference(self, minimal_config):
        """Test that referencing undefined profile is detected."""
        config = minimal_config.copy()
        config['detection_profile'] = 'custom_profile'
        # No threshold_profiles section with 'custom_profile'
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        assert not is_valid
        assert any('detection_profile' in i.field and 'custom_profile' in i.message for i in issues)
    
    def test_defined_custom_profile_passes(self, minimal_config):
        """Test that custom profile passes when defined."""
        config = minimal_config.copy()
        config['detection_profile'] = 'custom_profile'
        config['threshold_profiles'] = {
            'custom_profile': {
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
        }
        
        validator = ConfigValidator()
        is_valid, issues = validator.validate(config)
        
        errors = [i for i in issues if i.severity == 'error']
        assert is_valid, f"Custom profile should pass. Errors: {errors}"


class TestValidateConfigFunction:
    """Test the convenience validate_config function."""
    
    def test_validate_config_function(self, valid_config):
        """Test that validate_config function works correctly."""
        is_valid, issues = validate_config(valid_config)
        
        assert is_valid
        errors = [i for i in issues if i.severity == 'error']
        assert len(errors) == 0
    
    def test_validate_config_returns_errors(self, minimal_config):
        """Test that validate_config returns errors for invalid config."""
        config = minimal_config.copy()
        del config['thresholds']
        
        is_valid, issues = validate_config(config)
        
        assert not is_valid
        assert len(issues) > 0
        assert any(i.severity == 'error' for i in issues)


class TestNestedValueAccess:
    """Test nested value access helper method."""
    
    def test_get_nested_value_simple(self):
        """Test getting simple nested value."""
        validator = ConfigValidator()
        config = {'a': {'b': {'c': 42}}}
        
        value = validator._get_nested_value(config, 'a.b.c')
        assert value == 42
    
    def test_get_nested_value_missing(self):
        """Test getting missing nested value."""
        validator = ConfigValidator()
        config = {'a': {'b': {}}}
        
        value = validator._get_nested_value(config, 'a.b.c')
        assert value is None
    
    def test_get_nested_value_partial_path(self):
        """Test getting value with partial path."""
        validator = ConfigValidator()
        config = {'a': {'b': 'value'}}
        
        value = validator._get_nested_value(config, 'a.b.c.d')
        assert value is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
