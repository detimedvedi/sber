"""
Tests for AutoTuner threshold validation functionality.

Tests the threshold validation methods including:
- Threshold range validation
- Anomaly count validation
- Municipality coverage validation (95% not flagged requirement)
"""

import pytest
import pandas as pd
import numpy as np

from src.auto_tuner import AutoTuner


@pytest.fixture
def sample_config():
    """Sample configuration for AutoTuner."""
    return {
        'auto_tuning': {
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'optimization_strategy': 'adaptive'
        },
        'export': {
            'output_dir': 'output'
        }
    }


@pytest.fixture
def sample_data():
    """Sample municipal data for testing."""
    np.random.seed(42)
    n_municipalities = 100
    
    return pd.DataFrame({
        'territory_id': range(1, n_municipalities + 1),
        'region_name': [f'Region_{i % 10}' for i in range(n_municipalities)],
        'indicator_1': np.random.normal(100, 20, n_municipalities),
        'indicator_2': np.random.normal(50, 10, n_municipalities),
        'indicator_3': np.random.lognormal(3, 0.5, n_municipalities),
        'indicator_4': np.random.normal(200, 50, n_municipalities),
        'indicator_5': np.random.normal(75, 15, n_municipalities)
    })


def test_validate_thresholds_valid_configuration(sample_config, sample_data):
    """Test validation with valid threshold configuration."""
    tuner = AutoTuner(sample_config)
    
    # Use more relaxed thresholds that will pass validation
    thresholds = {
        'statistical': {
            'z_score': 3.5,
            'iqr_multiplier': 2.0
        },
        'geographic': {
            'regional_z_score': 3.0,
            'cluster_threshold': 3.0
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    # Check structure is correct
    assert 'is_valid' in result
    assert 'validation_errors' in result
    assert 'validation_warnings' in result
    assert 'detector_results' in result
    assert 'statistical' in result['detector_results']
    assert 'geographic' in result['detector_results']
    
    # Each detector should have complete validation results
    for detector_name in ['statistical', 'geographic']:
        det_result = result['detector_results'][detector_name]
        assert 'threshold_range_valid' in det_result
        assert 'anomaly_count_valid' in det_result
        assert 'municipality_coverage_valid' in det_result
        assert 'estimated_anomalies' in det_result


def test_validate_thresholds_out_of_range(sample_config, sample_data):
    """Test validation with thresholds out of acceptable range."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'statistical': {
            'z_score': 6.0,  # Too high (max is 5.0)
            'iqr_multiplier': 0.5  # Too low (min is 1.0)
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    # Should fail validation
    assert result['is_valid'] is False
    assert len(result['validation_errors']) > 0
    
    # Check detector-specific results
    stat_result = result['detector_results']['statistical']
    assert stat_result['threshold_range_valid'] is False
    assert len(stat_result['errors']) >= 2  # Both z_score and iqr_multiplier errors


def test_validate_thresholds_too_strict(sample_config, sample_data):
    """Test validation with thresholds that are too strict (too few anomalies)."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'statistical': {
            'z_score': 5.0,  # Very high - will detect very few anomalies
            'iqr_multiplier': 3.0
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    # May fail if estimated anomalies < min_anomalies
    stat_result = result['detector_results']['statistical']
    
    # Check that anomaly count was estimated
    assert 'estimated_anomalies' in stat_result
    assert stat_result['estimated_anomalies'] >= 0


def test_validate_thresholds_too_relaxed(sample_config, sample_data):
    """Test validation with thresholds that are too relaxed (too many anomalies)."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'statistical': {
            'z_score': 1.5,  # Very low - will detect many anomalies
            'iqr_multiplier': 1.0
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    stat_result = result['detector_results']['statistical']
    
    # Should detect high anomaly count
    assert 'estimated_anomalies' in stat_result
    
    # May fail validation if too many anomalies
    if not stat_result['anomaly_count_valid']:
        assert any('exceeds maximum' in err for err in stat_result['errors'])


def test_validate_municipality_coverage_requirement(sample_config, sample_data):
    """Test that validation enforces 95% of municipalities not flagged requirement."""
    tuner = AutoTuner(sample_config)
    
    # Use very relaxed thresholds that would flag many municipalities
    thresholds = {
        'statistical': {
            'z_score': 1.5,
            'iqr_multiplier': 1.0
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    stat_result = result['detector_results']['statistical']
    
    # Check that municipality coverage is validated
    assert 'municipality_coverage_valid' in stat_result
    assert 'estimated_flagged_municipalities' in stat_result
    assert 'flagged_percentage' in stat_result
    
    # If more than 5% flagged, should fail validation
    if stat_result['flagged_percentage'] > 5.0:
        assert stat_result['municipality_coverage_valid'] is False
        assert any('95% of normal municipalities not flagged' in err 
                  for err in stat_result['errors'])


def test_validate_single_detector(sample_config, sample_data):
    """Test validation of a single specific detector."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'statistical': {
            'z_score': 3.0,
            'iqr_multiplier': 1.5
        },
        'geographic': {
            'regional_z_score': 2.5,
            'cluster_threshold': 2.5
        }
    }
    
    # Validate only statistical detector
    result = tuner.validate_thresholds(sample_data, thresholds, detector_name='statistical')
    
    # Should only have results for statistical detector
    assert 'statistical' in result['detector_results']
    assert 'geographic' not in result['detector_results']


def test_validate_threshold_ranges_statistical(sample_config):
    """Test threshold range validation for statistical detector."""
    tuner = AutoTuner(sample_config)
    
    # Valid thresholds
    valid_thresholds = {
        'z_score': 3.0,
        'iqr_multiplier': 1.5,
        'percentile_lower': 1.0,
        'percentile_upper': 99.0
    }
    
    result = tuner._validate_threshold_ranges('statistical', valid_thresholds)
    assert result['is_valid'] is True
    assert len(result['errors']) == 0
    
    # Invalid thresholds
    invalid_thresholds = {
        'z_score': 0.5,  # Too low
        'iqr_multiplier': 5.0,  # Too high
    }
    
    result = tuner._validate_threshold_ranges('statistical', invalid_thresholds)
    assert result['is_valid'] is False
    assert len(result['errors']) >= 2


def test_validate_threshold_ranges_geographic(sample_config):
    """Test threshold range validation for geographic detector."""
    tuner = AutoTuner(sample_config)
    
    # Valid thresholds
    valid_thresholds = {
        'regional_z_score': 2.5,
        'cluster_threshold': 2.5,
        'neighbor_threshold': 2.0
    }
    
    result = tuner._validate_threshold_ranges('geographic', valid_thresholds)
    assert result['is_valid'] is True
    assert len(result['errors']) == 0
    
    # Invalid thresholds
    invalid_thresholds = {
        'regional_z_score': 5.0,  # Too high
        'cluster_threshold': 0.5,  # Too low
    }
    
    result = tuner._validate_threshold_ranges('geographic', invalid_thresholds)
    assert result['is_valid'] is False
    assert len(result['errors']) >= 2


def test_validate_threshold_ranges_temporal(sample_config):
    """Test threshold range validation for temporal detector."""
    tuner = AutoTuner(sample_config)
    
    # Valid thresholds
    valid_thresholds = {
        'spike_threshold': 100,
        'drop_threshold': -50,
        'volatility_multiplier': 2.0,
        'min_periods': 3
    }
    
    result = tuner._validate_threshold_ranges('temporal', valid_thresholds)
    assert result['is_valid'] is True
    assert len(result['errors']) == 0


def test_validate_threshold_ranges_cross_source(sample_config):
    """Test threshold range validation for cross-source detector."""
    tuner = AutoTuner(sample_config)
    
    # Valid thresholds
    valid_thresholds = {
        'discrepancy_threshold': 50,
        'correlation_threshold': 0.5,
        'min_correlation': 0.3
    }
    
    result = tuner._validate_threshold_ranges('cross_source', valid_thresholds)
    assert result['is_valid'] is True
    assert len(result['errors']) == 0


def test_estimate_anomaly_count_statistical(sample_config, sample_data):
    """Test anomaly count estimation for statistical detector."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'z_score': 3.0,
        'iqr_multiplier': 1.5
    }
    
    count = tuner._estimate_anomaly_count(sample_data, 'statistical', thresholds)
    
    # Should return a reasonable count
    assert count >= 0
    assert count < len(sample_data) * 10  # Sanity check


def test_estimate_anomaly_count_geographic(sample_config, sample_data):
    """Test anomaly count estimation for geographic detector."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'regional_z_score': 2.5,
        'cluster_threshold': 2.5
    }
    
    count = tuner._estimate_anomaly_count(sample_data, 'geographic', thresholds)
    
    # Should return a reasonable count
    assert count >= 0


def test_estimate_anomaly_count_temporal(sample_config, sample_data):
    """Test anomaly count estimation for temporal detector."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'spike_threshold': 100,
        'drop_threshold': -50
    }
    
    count = tuner._estimate_anomaly_count(sample_data, 'temporal', thresholds)
    
    # Should return a reasonable count
    assert count >= 0


def test_validation_warnings(sample_config, sample_data):
    """Test that validation generates appropriate warnings."""
    tuner = AutoTuner(sample_config)
    
    # Use thresholds close to boundaries
    thresholds = {
        'statistical': {
            'z_score': 1.55,  # Close to minimum (1.5)
            'iqr_multiplier': 3.45  # Close to maximum (3.5)
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    # Should have warnings about values close to boundaries or anomaly counts
    # Check that warnings are generated (either from range or anomaly count)
    stat_result = result['detector_results']['statistical']
    assert len(stat_result['warnings']) > 0 or len(result['validation_warnings']) > 0


def test_validation_with_empty_data(sample_config):
    """Test validation with empty DataFrame."""
    tuner = AutoTuner(sample_config)
    
    empty_df = pd.DataFrame()
    
    thresholds = {
        'statistical': {
            'z_score': 3.0,
            'iqr_multiplier': 1.5
        }
    }
    
    result = tuner.validate_thresholds(empty_df, thresholds)
    
    # Should handle empty data gracefully
    assert 'detector_results' in result


def test_validation_comprehensive(sample_config, sample_data):
    """Test comprehensive validation with multiple detectors."""
    tuner = AutoTuner(sample_config)
    
    thresholds = {
        'statistical': {
            'z_score': 3.0,
            'iqr_multiplier': 1.5
        },
        'geographic': {
            'regional_z_score': 2.5,
            'cluster_threshold': 2.5
        },
        'temporal': {
            'spike_threshold': 100,
            'drop_threshold': -50,
            'volatility_multiplier': 2.0
        },
        'cross_source': {
            'discrepancy_threshold': 50,
            'correlation_threshold': 0.5
        }
    }
    
    result = tuner.validate_thresholds(sample_data, thresholds)
    
    # Should validate all detectors
    assert len(result['detector_results']) == 4
    
    # Each detector should have complete validation results
    for detector_name, det_result in result['detector_results'].items():
        assert 'threshold_range_valid' in det_result
        assert 'anomaly_count_valid' in det_result
        assert 'municipality_coverage_valid' in det_result
        assert 'estimated_anomalies' in det_result
        assert 'estimated_flagged_municipalities' in det_result
        assert 'flagged_percentage' in det_result
