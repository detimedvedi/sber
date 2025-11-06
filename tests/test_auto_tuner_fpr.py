"""
Tests for AutoTuner FPR calculation functionality.

Tests the false positive rate calculation methods including:
- Historical FPR calculation
- Threshold sweep analysis
- Optimal threshold identification
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

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
        'indicator_3': np.random.lognormal(3, 0.5, n_municipalities)
    })


@pytest.fixture
def sample_historical_anomalies():
    """Sample historical anomaly detection results."""
    np.random.seed(42)
    
    detectors = ['statistical', 'geographic', 'cross_source', 'logical']
    anomalies = []
    
    for detector in detectors:
        # Generate varying numbers of anomalies per detector
        if detector == 'geographic':
            n_anomalies = 500  # High FPR
        elif detector == 'statistical':
            n_anomalies = 50   # Low FPR
        else:
            n_anomalies = 100  # Moderate FPR
        
        for i in range(n_anomalies):
            anomalies.append({
                'detector_name': detector,
                'territory_id': np.random.randint(1, 101),
                'severity_score': np.random.uniform(30, 95),
                'indicator': f'indicator_{np.random.randint(1, 4)}'
            })
    
    return pd.DataFrame(anomalies)


def test_calculate_fpr_from_historical_results(sample_config, sample_historical_anomalies):
    """Test FPR calculation from historical anomaly results."""
    tuner = AutoTuner(sample_config)
    
    total_municipalities = 100
    fpr_by_detector = tuner.calculate_fpr_from_historical_results(
        sample_historical_anomalies, total_municipalities
    )
    
    # Should return FPR for each detector
    assert 'statistical' in fpr_by_detector
    assert 'geographic' in fpr_by_detector
    assert 'cross_source' in fpr_by_detector
    assert 'logical' in fpr_by_detector
    
    # FPR should be between 0 and 1
    for detector, fpr in fpr_by_detector.items():
        assert 0 <= fpr <= 1, f"{detector} FPR out of range: {fpr}"
    
    # Geographic should have higher FPR (more anomalies)
    assert fpr_by_detector['geographic'] > fpr_by_detector['statistical']


def test_calculate_fpr_by_threshold_sweep(sample_config, sample_data):
    """Test FPR calculation using threshold sweep."""
    tuner = AutoTuner(sample_config)
    
    threshold_range = np.linspace(2.0, 4.0, 11)
    thresholds, fpr_values = tuner.calculate_fpr_by_threshold_sweep(
        sample_data, 'statistical', 'z_score', threshold_range
    )
    
    # Should return arrays of same length
    assert len(thresholds) == len(fpr_values)
    assert len(fpr_values) == 11
    
    # FPR should decrease as threshold increases
    assert fpr_values[0] > fpr_values[-1]
    
    # All FPR values should be valid
    assert all(0 <= fpr <= 1 for fpr in fpr_values)


def test_identify_optimal_threshold(sample_config):
    """Test optimal threshold identification."""
    tuner = AutoTuner(sample_config)
    
    threshold_range = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    fpr_values = np.array([0.10, 0.07, 0.05, 0.03, 0.01])
    
    # Find threshold for target FPR of 0.05
    optimal = tuner.identify_optimal_threshold(threshold_range, fpr_values, target_fpr=0.05)
    
    # Should select threshold closest to target FPR
    assert optimal == 3.0


def test_load_historical_results_empty_directory(sample_config):
    """Test loading historical results when no files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = sample_config.copy()
        config['export']['output_dir'] = tmpdir
        
        tuner = AutoTuner(config)
        df = tuner.load_historical_results()
        
        # Should return empty DataFrame
        assert df.empty


def test_analyze_historical_fpr(sample_config, sample_data, sample_historical_anomalies):
    """Test comprehensive historical FPR analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = sample_config.copy()
        config['export']['output_dir'] = tmpdir
        
        # Save historical results to file
        results_file = Path(tmpdir) / 'anomalies_test.csv'
        sample_historical_anomalies.to_csv(results_file, index=False, encoding='utf-8')
        
        tuner = AutoTuner(config)
        analysis = tuner.analyze_historical_fpr(sample_data, str(results_file))
        
        # Should return analysis for each detector
        assert 'statistical' in analysis
        assert 'geographic' in analysis
        
        # Each analysis should have required fields
        for detector, info in analysis.items():
            assert 'estimated_fpr' in info
            assert 'detection_rate' in info
            assert 'flagged_municipalities' in info
            assert 'total_anomalies' in info
            assert 'avg_severity' in info
            assert 'severity_distribution' in info
            assert 'recommendation' in info
            assert 'adjustment' in info
            assert 'meets_target' in info
            
            # Validate ranges
            assert 0 <= info['estimated_fpr'] <= 1
            assert 0 <= info['detection_rate'] <= 1
            assert info['flagged_municipalities'] >= 0
            assert info['total_anomalies'] > 0


def test_fpr_calculation_with_empty_data(sample_config):
    """Test FPR calculation with empty historical data."""
    tuner = AutoTuner(sample_config)
    
    empty_df = pd.DataFrame()
    fpr_by_detector = tuner.calculate_fpr_from_historical_results(empty_df, 100)
    
    # Should return empty dict
    assert fpr_by_detector == {}


def test_threshold_sweep_geographic_detector(sample_config, sample_data):
    """Test threshold sweep for geographic detector."""
    tuner = AutoTuner(sample_config)
    
    threshold_range = np.linspace(1.5, 3.5, 11)
    thresholds, fpr_values = tuner.calculate_fpr_by_threshold_sweep(
        sample_data, 'geographic', 'regional_z_score', threshold_range
    )
    
    # Should return valid results
    assert len(thresholds) == len(fpr_values)
    assert all(0 <= fpr <= 1 for fpr in fpr_values)


def test_fpr_recommendations(sample_config, sample_data, sample_historical_anomalies):
    """Test that FPR analysis generates appropriate recommendations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = sample_config.copy()
        config['export']['output_dir'] = tmpdir
        
        # Save historical results
        results_file = Path(tmpdir) / 'anomalies_test.csv'
        sample_historical_anomalies.to_csv(results_file, index=False, encoding='utf-8')
        
        tuner = AutoTuner(config)
        analysis = tuner.analyze_historical_fpr(sample_data, str(results_file))
        
        # Geographic detector should recommend increasing thresholds (high FPR)
        assert analysis['geographic']['adjustment'] in ['increase_high', 'increase_moderate']
        
        # All detectors should have valid adjustment recommendations
        valid_adjustments = ['increase_high', 'increase_moderate', 'decrease', 'maintain']
        for detector, info in analysis.items():
            assert info['adjustment'] in valid_adjustments, \
                f"{detector} has invalid adjustment: {info['adjustment']}"
