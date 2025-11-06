"""
Tests for threshold profile loading functionality.

Tests profile loading, validation, and merging with defaults.
"""

import pytest
import yaml
from src.detector_manager import ThresholdManager


@pytest.fixture
def base_config():
    """Load base configuration from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def minimal_config():
    """Create minimal configuration with defaults only."""
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
        }
    }


def test_profile_loading_with_full_config(base_config):
    """Test loading profiles from full configuration."""
    # Test normal profile
    base_config['detection_profile'] = 'normal'
    manager = ThresholdManager(base_config)
    
    assert manager.profile == 'normal'
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 3.0
    
    # Test strict profile
    base_config['detection_profile'] = 'strict'
    manager = ThresholdManager(base_config)
    
    assert manager.profile == 'strict'
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 2.5  # Stricter threshold


def test_profile_loading_with_missing_profile(minimal_config):
    """Test loading when profile doesn't exist - should use defaults."""
    minimal_config['detection_profile'] = 'nonexistent'
    manager = ThresholdManager(minimal_config)
    
    # Should fall back to defaults
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 3.0  # Default value


def test_profile_merging_with_defaults(minimal_config):
    """Test that incomplete profiles are merged with defaults."""
    # Create incomplete profile (missing some parameters)
    minimal_config['threshold_profiles'] = {
        'incomplete': {
            'statistical': {
                'z_score': 2.8  # Only one parameter
                # Missing: iqr_multiplier, percentile_lower, percentile_upper
            },
            'temporal': {
                'spike_threshold': 120
                # Missing: drop_threshold, volatility_multiplier
            }
            # Missing: geographic, cross_source, logical
        }
    }
    minimal_config['detection_profile'] = 'incomplete'
    
    manager = ThresholdManager(minimal_config)
    
    # Check that profile value is used
    stat_thresholds = manager.get_thresholds('statistical')
    assert stat_thresholds['z_score'] == 2.8  # From profile
    
    # Check that defaults fill in missing values
    assert stat_thresholds['iqr_multiplier'] == 1.5  # From defaults
    assert stat_thresholds['percentile_lower'] == 1  # From defaults
    assert stat_thresholds['percentile_upper'] == 99  # From defaults
    
    # Check temporal
    temp_thresholds = manager.get_thresholds('temporal')
    assert temp_thresholds['spike_threshold'] == 120  # From profile
    assert temp_thresholds['drop_threshold'] == -50  # From defaults
    
    # Check completely missing detector type uses all defaults
    geo_thresholds = manager.get_thresholds('geographic')
    assert geo_thresholds['regional_z_score'] == 2.0  # From defaults
    assert geo_thresholds['cluster_threshold'] == 2.5  # From defaults


def test_profile_validation(minimal_config):
    """Test profile completeness validation."""
    # Create complete profile
    minimal_config['threshold_profiles'] = {
        'complete': minimal_config['thresholds'].copy()
    }
    minimal_config['detection_profile'] = 'complete'
    
    manager = ThresholdManager(minimal_config)
    profile_info = manager.get_profile_info()
    
    assert profile_info['validation']['is_valid'] is True
    assert profile_info['validation']['completeness_percentage'] == 100.0
    assert len(profile_info['validation']['missing_params']) == 0


def test_profile_validation_incomplete(minimal_config):
    """Test validation of incomplete profile."""
    # Create incomplete profile
    minimal_config['threshold_profiles'] = {
        'incomplete': {
            'statistical': {
                'z_score': 2.8
                # Missing other parameters
            }
        }
    }
    minimal_config['detection_profile'] = 'incomplete'
    
    manager = ThresholdManager(minimal_config)
    profile_info = manager.get_profile_info()
    
    # After merging with defaults, should be complete
    assert profile_info['validation']['is_valid'] is True
    assert profile_info['validation']['completeness_percentage'] == 100.0


def test_runtime_profile_switching(base_config):
    """Test switching profiles at runtime."""
    base_config['detection_profile'] = 'normal'
    manager = ThresholdManager(base_config)
    
    # Initial profile
    assert manager.profile == 'normal'
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 3.0
    
    # Switch to strict
    manager.load_profile('strict')
    assert manager.profile == 'strict'
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 2.5
    
    # Switch to relaxed
    manager.load_profile('relaxed')
    assert manager.profile == 'relaxed'
    thresholds = manager.get_thresholds('statistical')
    assert thresholds['z_score'] == 3.5


def test_runtime_profile_switching_invalid(base_config):
    """Test switching to invalid profile raises error."""
    manager = ThresholdManager(base_config)
    
    with pytest.raises(ValueError, match="Unknown profile"):
        manager.load_profile('nonexistent')


def test_custom_profile_loading(minimal_config):
    """Test loading custom profile."""
    manager = ThresholdManager(minimal_config)
    
    # Load custom profile with partial thresholds
    custom_thresholds = {
        'statistical': {
            'z_score': 2.7,
            'iqr_multiplier': 1.8
        },
        'geographic': {
            'regional_z_score': 2.2
        }
    }
    
    result = manager.load_custom_profile(custom_thresholds, 'my_custom')
    
    assert manager.profile == 'my_custom'
    
    # Check custom values are used
    stat_thresholds = manager.get_thresholds('statistical')
    assert stat_thresholds['z_score'] == 2.7
    assert stat_thresholds['iqr_multiplier'] == 1.8
    
    # Check defaults fill in missing values
    assert stat_thresholds['percentile_lower'] == 1
    assert stat_thresholds['percentile_upper'] == 99
    
    # Check geographic
    geo_thresholds = manager.get_thresholds('geographic')
    assert geo_thresholds['regional_z_score'] == 2.2
    assert geo_thresholds['cluster_threshold'] == 2.5  # From defaults


def test_profile_info(base_config):
    """Test getting profile information."""
    base_config['detection_profile'] = 'strict'
    manager = ThresholdManager(base_config)
    
    info = manager.get_profile_info()
    
    assert info['profile_name'] == 'strict'
    assert 'validation' in info
    assert 'thresholds' in info
    assert info['validation']['is_valid'] is True
    assert 'statistical' in info['thresholds']
    assert 'temporal' in info['thresholds']
    assert 'geographic' in info['thresholds']


def test_all_profiles_are_complete(base_config):
    """Test that all predefined profiles are complete."""
    profiles = ['strict', 'normal', 'relaxed']
    
    for profile_name in profiles:
        base_config['detection_profile'] = profile_name
        manager = ThresholdManager(base_config)
        
        info = manager.get_profile_info()
        assert info['validation']['is_valid'] is True, \
            f"Profile '{profile_name}' is incomplete: {info['validation']['missing_params']}"
        assert info['validation']['completeness_percentage'] == 100.0


def test_profile_thresholds_differ(base_config):
    """Test that different profiles have different threshold values."""
    # Get thresholds for each profile
    profiles_thresholds = {}
    
    for profile_name in ['strict', 'normal', 'relaxed']:
        base_config['detection_profile'] = profile_name
        manager = ThresholdManager(base_config)
        profiles_thresholds[profile_name] = manager.get_thresholds('statistical')
    
    # Strict should have lower z_score than normal
    assert profiles_thresholds['strict']['z_score'] < profiles_thresholds['normal']['z_score']
    
    # Relaxed should have higher z_score than normal
    assert profiles_thresholds['relaxed']['z_score'] > profiles_thresholds['normal']['z_score']
    
    # Order should be: strict < normal < relaxed
    assert (profiles_thresholds['strict']['z_score'] < 
            profiles_thresholds['normal']['z_score'] < 
            profiles_thresholds['relaxed']['z_score'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
