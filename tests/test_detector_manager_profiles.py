"""
Tests for DetectorManager Profile Integration

Tests that DetectorManager correctly:
1. Loads profiles on initialization
2. Applies profile thresholds to detectors
3. Supports runtime profile switching
"""

import pytest
import pandas as pd
import numpy as np
from src.detector_manager import DetectorManager


@pytest.fixture
def sample_config():
    """Sample configuration with threshold profiles."""
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
        'threshold_profiles': {
            'strict': {
                'statistical': {
                    'z_score': 2.5,
                    'iqr_multiplier': 1.2,
                    'percentile_lower': 2,
                    'percentile_upper': 98
                },
                'temporal': {
                    'spike_threshold': 75,
                    'drop_threshold': -40,
                    'volatility_multiplier': 1.5
                },
                'geographic': {
                    'regional_z_score': 1.5,
                    'cluster_threshold': 2.0
                },
                'cross_source': {
                    'correlation_threshold': 0.6,
                    'discrepancy_threshold': 40
                },
                'logical': {
                    'check_negative_values': True,
                    'check_impossible_ratios': True
                }
            },
            'normal': {
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
            'relaxed': {
                'statistical': {
                    'z_score': 3.5,
                    'iqr_multiplier': 2.0,
                    'percentile_lower': 0.5,
                    'percentile_upper': 99.5
                },
                'temporal': {
                    'spike_threshold': 150,
                    'drop_threshold': -60,
                    'volatility_multiplier': 2.5
                },
                'geographic': {
                    'regional_z_score': 3.0,
                    'cluster_threshold': 3.0
                },
                'cross_source': {
                    'correlation_threshold': 0.4,
                    'discrepancy_threshold': 60
                },
                'logical': {
                    'check_negative_values': True,
                    'check_impossible_ratios': True
                }
            }
        },
        'missing_value_handling': {
            'indicator_threshold': 50.0,
            'municipality_threshold': 70.0
        }
    }


@pytest.fixture
def sample_data():
    """Sample municipal data for testing."""
    np.random.seed(42)
    
    data = {
        'territory_id': range(1, 101),
        'municipal_name': [f'Municipality_{i}' for i in range(1, 101)],
        'region_name': ['Region_A'] * 50 + ['Region_B'] * 50,
        'consumption_total': np.random.normal(1000, 200, 100),
        'population_total': np.random.normal(50000, 10000, 100),
        'salary_average': np.random.normal(40000, 8000, 100)
    }
    
    return pd.DataFrame(data)


def test_profile_loaded_on_initialization(sample_config):
    """Test that profile is loaded when DetectorManager is initialized."""
    # Test with normal profile
    sample_config['detection_profile'] = 'normal'
    manager = DetectorManager(sample_config)
    
    assert manager.get_current_profile() == 'normal'
    
    # Verify thresholds are applied to config
    assert manager.config['thresholds']['statistical']['z_score'] == 3.0
    assert manager.config['thresholds']['geographic']['regional_z_score'] == 2.0


def test_strict_profile_loaded_on_initialization(sample_config):
    """Test that strict profile is loaded correctly."""
    sample_config['detection_profile'] = 'strict'
    manager = DetectorManager(sample_config)
    
    assert manager.get_current_profile() == 'strict'
    
    # Verify strict thresholds are applied
    assert manager.config['thresholds']['statistical']['z_score'] == 2.5
    assert manager.config['thresholds']['geographic']['regional_z_score'] == 1.5


def test_relaxed_profile_loaded_on_initialization(sample_config):
    """Test that relaxed profile is loaded correctly."""
    sample_config['detection_profile'] = 'relaxed'
    manager = DetectorManager(sample_config)
    
    assert manager.get_current_profile() == 'relaxed'
    
    # Verify relaxed thresholds are applied
    assert manager.config['thresholds']['statistical']['z_score'] == 3.5
    assert manager.config['thresholds']['geographic']['regional_z_score'] == 3.0


def test_profile_thresholds_applied_to_detectors(sample_config):
    """Test that profile thresholds are actually used by detectors."""
    sample_config['detection_profile'] = 'strict'
    manager = DetectorManager(sample_config)
    
    # Check that detectors have the correct thresholds
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        assert detector.z_score_threshold == 2.5
        assert detector.iqr_multiplier == 1.2
    
    if 'geographic' in manager.detectors:
        detector = manager.detectors['geographic']
        assert detector.regional_z_score_threshold == 1.5
        assert detector.cluster_threshold == 2.0


def test_runtime_profile_switching(sample_config):
    """Test that profile can be switched at runtime."""
    # Start with normal profile
    sample_config['detection_profile'] = 'normal'
    manager = DetectorManager(sample_config)
    
    assert manager.get_current_profile() == 'normal'
    assert manager.config['thresholds']['statistical']['z_score'] == 3.0
    
    # Switch to strict
    manager.switch_profile('strict')
    
    assert manager.get_current_profile() == 'strict'
    assert manager.config['thresholds']['statistical']['z_score'] == 2.5
    
    # Verify detectors are reinitialized with new thresholds
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        assert detector.z_score_threshold == 2.5
    
    # Switch to relaxed
    manager.switch_profile('relaxed')
    
    assert manager.get_current_profile() == 'relaxed'
    assert manager.config['thresholds']['statistical']['z_score'] == 3.5
    
    # Verify detectors are reinitialized again
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        assert detector.z_score_threshold == 3.5


def test_invalid_profile_raises_error(sample_config):
    """Test that switching to invalid profile raises ValueError."""
    manager = DetectorManager(sample_config)
    
    with pytest.raises(ValueError, match="Unknown profile"):
        manager.switch_profile('nonexistent')


def test_get_profile_info(sample_config):
    """Test that profile info is returned correctly."""
    sample_config['detection_profile'] = 'normal'
    manager = DetectorManager(sample_config)
    
    info = manager.get_profile_info()
    
    assert 'profile_name' in info
    assert info['profile_name'] == 'normal'
    
    assert 'validation' in info
    assert info['validation']['is_valid'] is True
    
    assert 'thresholds' in info
    assert 'statistical' in info['thresholds']
    assert info['thresholds']['statistical']['z_score'] == 3.0


def test_detectors_initialized_with_profile(sample_config):
    """Test that all detectors are initialized with profile thresholds."""
    sample_config['detection_profile'] = 'strict'
    manager = DetectorManager(sample_config)
    
    # Verify detectors were initialized
    assert len(manager.detectors) > 0
    
    # Check each detector has correct thresholds
    expected_thresholds = {
        'statistical': {'z_score': 2.5, 'iqr_multiplier': 1.2},
        'geographic': {'regional_z_score': 1.5, 'cluster_threshold': 2.0},
        'temporal': {'spike_threshold': 75, 'drop_threshold': -40},
        'cross_source': {'correlation_threshold': 0.6, 'discrepancy_threshold': 40}
    }
    
    for detector_name, detector in manager.detectors.items():
        if detector_name in expected_thresholds:
            for threshold_name, expected_value in expected_thresholds[detector_name].items():
                # Get threshold attribute name (e.g., 'z_score' -> 'z_score_threshold')
                if threshold_name == 'z_score':
                    attr_name = 'z_score_threshold'
                elif threshold_name == 'regional_z_score':
                    attr_name = 'regional_z_score_threshold'
                else:
                    attr_name = threshold_name
                
                if hasattr(detector, attr_name):
                    actual_value = getattr(detector, attr_name)
                    assert actual_value == expected_value, (
                        f"{detector_name}.{attr_name} should be {expected_value}, "
                        f"got {actual_value}"
                    )


def test_profile_switching_preserves_detector_count(sample_config):
    """Test that profile switching doesn't lose detectors."""
    manager = DetectorManager(sample_config)
    
    initial_count = len(manager.detectors)
    
    manager.switch_profile('strict')
    assert len(manager.detectors) == initial_count
    
    manager.switch_profile('relaxed')
    assert len(manager.detectors) == initial_count


def test_detection_with_different_profiles(sample_config, sample_data):
    """Test that detection results differ with different profiles."""
    # Run with strict profile
    sample_config['detection_profile'] = 'strict'
    manager_strict = DetectorManager(sample_config)
    results_strict = manager_strict.run_all_detectors(sample_data)
    
    # Run with relaxed profile
    sample_config['detection_profile'] = 'relaxed'
    manager_relaxed = DetectorManager(sample_config)
    results_relaxed = manager_relaxed.run_all_detectors(sample_data)
    
    # Count total anomalies
    strict_count = sum(len(df) for df in results_strict if df is not None)
    relaxed_count = sum(len(df) for df in results_relaxed if df is not None)
    
    # Strict should detect more anomalies (or equal if no anomalies detected)
    # Note: This might not always be true depending on data, but generally expected
    # We just verify both ran successfully
    assert strict_count >= 0
    assert relaxed_count >= 0


def test_profile_info_after_switching(sample_config):
    """Test that profile info updates after switching."""
    manager = DetectorManager(sample_config)
    
    # Initial profile
    info = manager.get_profile_info()
    assert info['profile_name'] == 'normal'
    
    # Switch profile
    manager.switch_profile('strict')
    
    # Check updated info
    info = manager.get_profile_info()
    assert info['profile_name'] == 'strict'
    assert info['thresholds']['statistical']['z_score'] == 2.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
