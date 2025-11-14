"""
Tests for DetectorManager Conditional Detector Loading

Tests that DetectorManager correctly:
1. Checks config.detectors section for enabled flags
2. Only initializes detectors where enabled flag is true
3. Logs warning when CrossSourceComparator is enabled
4. Logs info when detectors are disabled
"""

import pytest
import logging
from src.detector_manager import DetectorManager


@pytest.fixture
def base_config():
    """Base configuration with all required threshold sections."""
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
        'missing_value_handling': {
            'indicator_threshold': 50.0,
            'municipality_threshold': 70.0
        }
    }


def test_all_detectors_enabled_by_default(base_config):
    """Test that all detectors except cross_source are enabled by default."""
    # No detectors section - should use defaults
    manager = DetectorManager(base_config)
    
    # Statistical, temporal, geographic, logical should be enabled by default
    assert 'statistical' in manager.detectors
    assert 'temporal' in manager.detectors
    assert 'geographic' in manager.detectors
    assert 'logical' in manager.detectors
    
    # CrossSource should be disabled by default
    assert 'cross_source' not in manager.detectors


def test_cross_source_disabled_by_default(base_config):
    """Test that CrossSourceComparator is disabled by default."""
    base_config['detectors'] = {
        'cross_source': {
            'enabled': False
        }
    }
    
    manager = DetectorManager(base_config)
    
    assert 'cross_source' not in manager.detectors


def test_cross_source_can_be_enabled(base_config, caplog):
    """Test that CrossSourceComparator can be enabled and logs warning."""
    base_config['detectors'] = {
        'cross_source': {
            'enabled': True
        }
    }
    
    with caplog.at_level(logging.WARNING):
        manager = DetectorManager(base_config)
    
    # Should be initialized
    assert 'cross_source' in manager.detectors
    
    # Should log warning
    assert any('CrossSourceComparator enabled' in record.message for record in caplog.records)
    assert any('verify metric pairs are valid' in record.message for record in caplog.records)


def test_statistical_detector_can_be_disabled(base_config, caplog):
    """Test that StatisticalOutlierDetector can be disabled."""
    base_config['detectors'] = {
        'statistical': {
            'enabled': False
        }
    }
    
    with caplog.at_level(logging.INFO):
        manager = DetectorManager(base_config)
    
    # Should not be initialized
    assert 'statistical' not in manager.detectors
    
    # Should log info message
    assert any('StatisticalOutlierDetector disabled' in record.message for record in caplog.records)


def test_temporal_detector_can_be_disabled(base_config, caplog):
    """Test that TemporalAnomalyDetector can be disabled."""
    base_config['detectors'] = {
        'temporal': {
            'enabled': False
        }
    }
    
    with caplog.at_level(logging.INFO):
        manager = DetectorManager(base_config)
    
    # Should not be initialized
    assert 'temporal' not in manager.detectors
    
    # Should log info message
    assert any('TemporalAnomalyDetector disabled' in record.message for record in caplog.records)


def test_geographic_detector_can_be_disabled(base_config, caplog):
    """Test that GeographicAnomalyDetector can be disabled."""
    base_config['detectors'] = {
        'geographic': {
            'enabled': False
        }
    }
    
    with caplog.at_level(logging.INFO):
        manager = DetectorManager(base_config)
    
    # Should not be initialized
    assert 'geographic' not in manager.detectors
    
    # Should log info message
    assert any('GeographicAnomalyDetector disabled' in record.message for record in caplog.records)


def test_logical_detector_can_be_disabled(base_config, caplog):
    """Test that LogicalConsistencyChecker can be disabled."""
    base_config['detectors'] = {
        'logical': {
            'enabled': False
        }
    }
    
    with caplog.at_level(logging.INFO):
        manager = DetectorManager(base_config)
    
    # Should not be initialized
    assert 'logical' not in manager.detectors
    
    # Should log info message
    assert any('LogicalConsistencyChecker disabled' in record.message for record in caplog.records)


def test_multiple_detectors_can_be_disabled(base_config):
    """Test that multiple detectors can be disabled simultaneously."""
    base_config['detectors'] = {
        'statistical': {'enabled': False},
        'temporal': {'enabled': False},
        'cross_source': {'enabled': False}
    }
    
    manager = DetectorManager(base_config)
    
    # Only geographic and logical should be enabled
    assert 'statistical' not in manager.detectors
    assert 'temporal' not in manager.detectors
    assert 'cross_source' not in manager.detectors
    assert 'geographic' in manager.detectors
    assert 'logical' in manager.detectors


def test_all_detectors_can_be_explicitly_enabled(base_config):
    """Test that all detectors can be explicitly enabled."""
    base_config['detectors'] = {
        'statistical': {'enabled': True},
        'temporal': {'enabled': True},
        'geographic': {'enabled': True},
        'cross_source': {'enabled': True},
        'logical': {'enabled': True}
    }
    
    manager = DetectorManager(base_config)
    
    # All detectors should be initialized
    assert 'statistical' in manager.detectors
    assert 'temporal' in manager.detectors
    assert 'geographic' in manager.detectors
    assert 'cross_source' in manager.detectors
    assert 'logical' in manager.detectors


def test_detector_count_reflects_enabled_detectors(base_config):
    """Test that detector count matches number of enabled detectors."""
    # Enable only 2 detectors
    base_config['detectors'] = {
        'statistical': {'enabled': True},
        'temporal': {'enabled': False},
        'geographic': {'enabled': True},
        'cross_source': {'enabled': False},
        'logical': {'enabled': False}
    }
    
    manager = DetectorManager(base_config)
    
    # Should have exactly 2 detectors
    assert len(manager.detectors) == 2


def test_empty_detectors_config_uses_defaults(base_config):
    """Test that empty detectors config uses default enabled states."""
    base_config['detectors'] = {}
    
    manager = DetectorManager(base_config)
    
    # Should use defaults: all enabled except cross_source
    assert 'statistical' in manager.detectors
    assert 'temporal' in manager.detectors
    assert 'geographic' in manager.detectors
    assert 'logical' in manager.detectors
    assert 'cross_source' not in manager.detectors


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
