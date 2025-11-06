"""
Tests for Configuration Migration

Tests the automatic migration from old configuration format to new format.
"""

import pytest
from src.config_validator import ConfigMigrator, migrate_config_if_needed


@pytest.fixture
def old_config():
    """Sample old format configuration (pre-Phase 1)."""
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
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'output/anomaly_detection.log'
        },
        'visualization': {
            'figure_size': [12, 8],
            'dpi': 300,
            'style': 'seaborn-v0_8'
        },
        'data_processing': {
            'random_seed': 42,
            'handle_missing': 'log_and_continue'
        }
    }


@pytest.fixture
def new_config():
    """Sample new format configuration (Phase 1+)."""
    return {
        'detection_profile': 'normal',
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
        'temporal': {
            'enabled': False,
            'aggregation_method': 'latest',
            'auto_detect': True
        },
        'municipality_classification': {
            'enabled': True,
            'urban_population_threshold': 50000,
            'capital_cities': ['Москва', 'Санкт-Петербург']
        },
        'export': {
            'output_dir': 'output',
            'timestamp_format': '%Y%m%d_%H%M%S',
            'top_n_municipalities': 50
        },
        'data_paths': {
            'sberindex': {},
            'rosstat': {},
            'municipal_dict': {}
        }
    }


class TestConfigMigrator:
    """Test suite for ConfigMigrator class."""
    
    def test_is_old_format_detects_old_config(self, old_config):
        """Test that old format is correctly detected."""
        migrator = ConfigMigrator()
        
        assert migrator.is_old_format(old_config) is True
    
    def test_is_old_format_detects_new_config(self, new_config):
        """Test that new format is correctly detected."""
        migrator = ConfigMigrator()
        
        assert migrator.is_old_format(new_config) is False
    
    def test_is_old_format_with_partial_new_fields(self, old_config):
        """Test detection when config has some but not all new fields."""
        migrator = ConfigMigrator()
        
        # Add one new field
        old_config['detection_profile'] = 'normal'
        
        # Should be detected as new format (has at least one new field)
        assert migrator.is_old_format(old_config) is False
    
    def test_is_old_format_with_invalid_config(self):
        """Test detection with invalid/incomplete config."""
        migrator = ConfigMigrator()
        
        # Missing required fields
        invalid_config = {'thresholds': {}}
        
        assert migrator.is_old_format(invalid_config) is False
    
    def test_migrate_adds_detection_profile(self, old_config):
        """Test that migration adds detection_profile field."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'detection_profile' in new_config
        assert new_config['detection_profile'] == 'normal'
    
    def test_migrate_adds_temporal_settings(self, old_config):
        """Test that migration adds temporal settings."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'temporal' in new_config
        assert new_config['temporal']['enabled'] is False
        assert new_config['temporal']['aggregation_method'] == 'latest'
        assert new_config['temporal']['auto_detect'] is True
    
    def test_migrate_adds_municipality_classification(self, old_config):
        """Test that migration adds municipality classification."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'municipality_classification' in new_config
        assert new_config['municipality_classification']['enabled'] is True
        assert new_config['municipality_classification']['urban_population_threshold'] == 50000
        assert 'capital_cities' in new_config['municipality_classification']
        assert len(new_config['municipality_classification']['capital_cities']) > 0
    
    def test_migrate_adds_robust_statistics(self, old_config):
        """Test that migration adds robust statistics settings."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'robust_statistics' in new_config
        assert new_config['robust_statistics']['enabled'] is True
        assert new_config['robust_statistics']['use_median'] is True
        assert new_config['robust_statistics']['use_mad'] is True
        assert new_config['robust_statistics']['winsorization_limits'] == [0.01, 0.99]
        assert new_config['robust_statistics']['log_transform_skewness_threshold'] == 2.0
    
    def test_migrate_adds_priority_weights(self, old_config):
        """Test that migration adds priority weights."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'priority_weights' in new_config
        assert 'anomaly_types' in new_config['priority_weights']
        assert 'indicators' in new_config['priority_weights']
        
        # Check specific weights
        assert new_config['priority_weights']['anomaly_types']['logical_inconsistency'] == 1.5
        assert new_config['priority_weights']['indicators']['population'] == 1.3
    
    def test_migrate_adds_missing_value_handling(self, old_config):
        """Test that migration adds missing value handling settings."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'missing_value_handling' in new_config
        assert new_config['missing_value_handling']['indicator_threshold'] == 50.0
        assert new_config['missing_value_handling']['municipality_threshold'] == 70.0
    
    def test_migrate_adds_auto_tuning(self, old_config):
        """Test that migration adds auto-tuning settings."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'auto_tuning' in new_config
        assert new_config['auto_tuning']['enabled'] is False
        assert new_config['auto_tuning']['target_false_positive_rate'] == 0.05
        assert new_config['auto_tuning']['min_anomalies_per_detector'] == 10
        assert new_config['auto_tuning']['max_anomalies_per_detector'] == 1000
        assert new_config['auto_tuning']['retuning_interval_days'] == 30
    
    def test_migrate_generates_threshold_profiles(self, old_config):
        """Test that migration generates threshold profiles."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'threshold_profiles' in new_config
        assert 'strict' in new_config['threshold_profiles']
        assert 'normal' in new_config['threshold_profiles']
        assert 'relaxed' in new_config['threshold_profiles']
    
    def test_migrate_threshold_profiles_normal_matches_base(self, old_config):
        """Test that normal profile matches base thresholds."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        # Normal profile should match original thresholds
        assert new_config['threshold_profiles']['normal'] == old_config['thresholds']
    
    def test_migrate_threshold_profiles_strict_is_lower(self, old_config):
        """Test that strict profile has lower thresholds."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        # Strict should be 80% of normal
        base_z_score = old_config['thresholds']['statistical']['z_score']
        strict_z_score = new_config['threshold_profiles']['strict']['statistical']['z_score']
        
        assert strict_z_score < base_z_score
        assert strict_z_score == pytest.approx(base_z_score * 0.8, rel=0.01)
    
    def test_migrate_threshold_profiles_relaxed_is_higher(self, old_config):
        """Test that relaxed profile has higher thresholds."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        # Relaxed should be 120% of normal
        base_z_score = old_config['thresholds']['statistical']['z_score']
        relaxed_z_score = new_config['threshold_profiles']['relaxed']['statistical']['z_score']
        
        assert relaxed_z_score > base_z_score
        assert relaxed_z_score == pytest.approx(base_z_score * 1.2, rel=0.01)
    
    def test_migrate_preserves_original_fields(self, old_config):
        """Test that migration preserves all original fields."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        # Check that all original fields are still present
        assert new_config['thresholds'] == old_config['thresholds']
        
        # Export will have new fields added, but original fields should be preserved
        for key, value in old_config['export'].items():
            assert new_config['export'][key] == value
        
        assert new_config['data_paths'] == old_config['data_paths']
        assert new_config['logging'] == old_config['logging']
        assert new_config['visualization'] == old_config['visualization']
    
    def test_migrate_does_not_modify_original(self, old_config):
        """Test that migration does not modify the original config."""
        migrator = ConfigMigrator()
        
        original_copy = old_config.copy()
        migrator.migrate(old_config)
        
        # Original should be unchanged
        assert old_config == original_copy
    
    def test_migrate_generates_warnings(self, old_config):
        """Test that migration generates appropriate warnings."""
        migrator = ConfigMigrator()
        
        migrator.migrate(old_config)
        warnings = migrator.get_migration_warnings()
        
        assert len(warnings) > 0
        
        # Check for specific warnings
        warning_text = ' '.join(warnings)
        assert 'detection_profile' in warning_text
        assert 'temporal' in warning_text
        assert 'threshold profiles' in warning_text.lower()
    
    def test_migrate_updates_export_settings(self, old_config):
        """Test that migration adds new export settings."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        # Check new export fields
        assert new_config['export']['generate_executive_summary'] is True
        assert new_config['export']['generate_dashboard'] is True
        assert new_config['export']['use_management_descriptions'] is True
        assert new_config['export']['highlight_critical_threshold'] == 90
        
        # Original export fields should be preserved
        assert new_config['export']['output_dir'] == old_config['export']['output_dir']
        assert new_config['export']['timestamp_format'] == old_config['export']['timestamp_format']
    
    def test_migrate_updates_data_processing(self, old_config):
        """Test that migration adds min_data_completeness to data_processing."""
        migrator = ConfigMigrator()
        
        new_config = migrator.migrate(old_config)
        
        assert 'min_data_completeness' in new_config['data_processing']
        assert new_config['data_processing']['min_data_completeness'] == 0.5
        
        # Original fields preserved
        assert new_config['data_processing']['random_seed'] == old_config['data_processing']['random_seed']


class TestMigrateConfigIfNeeded:
    """Test suite for migrate_config_if_needed convenience function."""
    
    def test_migrate_old_config(self, old_config):
        """Test migrating old configuration."""
        migrated, was_migrated = migrate_config_if_needed(old_config)
        
        assert was_migrated is True
        assert 'detection_profile' in migrated
        assert 'temporal' in migrated
        assert 'municipality_classification' in migrated
    
    def test_no_migration_for_new_config(self, new_config):
        """Test that new config is not migrated."""
        migrated, was_migrated = migrate_config_if_needed(new_config)
        
        assert was_migrated is False
        assert migrated == new_config
    
    def test_migrated_config_is_valid(self, old_config):
        """Test that migrated config passes validation."""
        from src.config_validator import validate_config
        
        migrated, was_migrated = migrate_config_if_needed(old_config)
        
        assert was_migrated is True
        
        # Validate migrated config
        is_valid, errors = validate_config(migrated)
        
        # Should be valid (may have warnings but no errors)
        error_messages = [e.message for e in errors if e.severity == 'error']
        assert is_valid is True, f"Validation errors: {error_messages}"


class TestThresholdScaling:
    """Test suite for threshold scaling logic."""
    
    def test_scale_thresholds_numeric_values(self):
        """Test scaling of numeric threshold values."""
        migrator = ConfigMigrator()
        
        thresholds = {
            'statistical': {
                'z_score': 3.0,
                'iqr_multiplier': 1.5
            }
        }
        
        scaled = migrator._scale_thresholds(thresholds, 0.8)
        
        assert scaled['statistical']['z_score'] == pytest.approx(2.4, rel=0.01)
        assert scaled['statistical']['iqr_multiplier'] == pytest.approx(1.2, rel=0.01)
    
    def test_scale_thresholds_preserves_percentiles(self):
        """Test that percentile scaling is handled specially."""
        migrator = ConfigMigrator()
        
        thresholds = {
            'statistical': {
                'percentile_lower': 1,
                'percentile_upper': 99
            }
        }
        
        # Scale down (strict)
        strict = migrator._scale_thresholds(thresholds, 0.8)
        
        # Lower percentile should scale towards 0
        assert strict['statistical']['percentile_lower'] < thresholds['statistical']['percentile_lower']
        assert strict['statistical']['percentile_lower'] >= 0.5
        
        # Upper percentile should scale towards 100
        assert strict['statistical']['percentile_upper'] > thresholds['statistical']['percentile_upper']
        assert strict['statistical']['percentile_upper'] <= 99.5
    
    def test_scale_thresholds_preserves_booleans(self):
        """Test that boolean values are not scaled."""
        migrator = ConfigMigrator()
        
        thresholds = {
            'logical': {
                'check_negative_values': True,
                'check_impossible_ratios': False
            }
        }
        
        scaled = migrator._scale_thresholds(thresholds, 0.8)
        
        assert scaled['logical']['check_negative_values'] is True
        assert scaled['logical']['check_impossible_ratios'] is False
    
    def test_scale_thresholds_does_not_modify_original(self):
        """Test that scaling does not modify original thresholds."""
        migrator = ConfigMigrator()
        
        thresholds = {
            'statistical': {
                'z_score': 3.0
            }
        }
        
        original_value = thresholds['statistical']['z_score']
        migrator._scale_thresholds(thresholds, 0.8)
        
        # Original should be unchanged
        assert thresholds['statistical']['z_score'] == original_value


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_migrate_empty_config(self):
        """Test migration with empty config."""
        migrator = ConfigMigrator()
        
        empty_config = {}
        
        # Should not crash, but won't add fields without base structure
        result = migrator.migrate(empty_config)
        
        # Should add new fields even to empty config
        assert 'detection_profile' in result
    
    def test_migrate_config_with_extra_fields(self, old_config):
        """Test migration preserves extra/custom fields."""
        migrator = ConfigMigrator()
        
        old_config['custom_field'] = 'custom_value'
        old_config['custom_section'] = {'key': 'value'}
        
        new_config = migrator.migrate(old_config)
        
        # Custom fields should be preserved
        assert new_config['custom_field'] == 'custom_value'
        assert new_config['custom_section'] == {'key': 'value'}
    
    def test_migrate_config_with_partial_thresholds(self):
        """Test migration with incomplete threshold structure."""
        migrator = ConfigMigrator()
        
        partial_config = {
            'thresholds': {
                'statistical': {
                    'z_score': 3.0
                }
                # Missing other categories
            },
            'export': {'output_dir': 'output'},
            'data_paths': {'sberindex': {}, 'rosstat': {}, 'municipal_dict': {}}
        }
        
        new_config = migrator.migrate(partial_config)
        
        # Should still add new fields
        assert 'detection_profile' in new_config
        assert 'temporal' in new_config
        
        # Threshold profiles should be generated even with partial thresholds
        assert 'threshold_profiles' in new_config
