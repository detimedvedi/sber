"""
Tests for anomaly count warnings functionality.

This module tests the check_anomaly_count_warnings method in ResultsAggregator
which validates anomaly counts and generates warnings with recommendations.
"""

import pytest
import pandas as pd
from src.results_aggregator import ResultsAggregator


class TestAnomalyCountWarnings:
    """Test suite for anomaly count warnings"""
    
    @pytest.fixture
    def aggregator(self):
        """Create ResultsAggregator instance"""
        config = {
            'priority_weights': {
                'anomaly_types': {
                    'logical_inconsistency': 1.5,
                    'cross_source_discrepancy': 1.2,
                    'temporal_anomaly': 1.1,
                    'statistical_outlier': 1.0,
                    'geographic_anomaly': 0.8
                },
                'indicators': {
                    'population': 1.3,
                    'consumption_total': 1.2,
                    'salary': 1.1,
                    'default': 1.0
                }
            }
        }
        return ResultsAggregator(config)
    
    def test_no_warnings_normal_metrics(self, aggregator):
        """Test that no warnings are generated for normal metrics"""
        metrics = {
            'total_anomalies': 500,
            'municipalities_affected': 150,
            'municipalities_affected_pct': 15.0,
            'anomaly_rate_per_1000': 500.0,
            'avg_anomalies_per_municipality': 3.3,
            'anomalies_by_severity': {
                'critical': 50,
                'high': 100,
                'medium': 200,
                'low': 150
            },
            'anomalies_by_type': {
                'geographic_anomaly': 200,
                'cross_source_discrepancy': 150,
                'logical_inconsistency': 100,
                'statistical_outlier': 50
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have no warnings for normal metrics
        assert isinstance(warnings, list)
        # May have info-level warnings, but no critical or warning level
        critical_warnings = [w for w in warnings if w['severity'] == 'critical']
        warning_level = [w for w in warnings if w['severity'] == 'warning']
        assert len(critical_warnings) == 0
        assert len(warning_level) == 0
    
    def test_low_anomaly_count_warning(self, aggregator):
        """Test warning for very low anomaly count"""
        metrics = {
            'total_anomalies': 5,  # Below minimum of 10
            'municipalities_affected': 5,
            'municipalities_affected_pct': 5.0,
            'anomaly_rate_per_1000': 50.0,
            'avg_anomalies_per_municipality': 1.0,
            'anomalies_by_severity': {
                'critical': 1,
                'high': 1,
                'medium': 2,
                'low': 1
            },
            'anomalies_by_type': {
                'geographic_anomaly': 5
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have warning for low anomaly count
        assert len(warnings) > 0
        low_count_warnings = [w for w in warnings if w['warning_type'] == 'low_anomaly_count']
        assert len(low_count_warnings) == 1
        
        warning = low_count_warnings[0]
        assert warning['severity'] == 'warning'
        assert 'мало аномалий' in warning['message'].lower()
        assert 'снижение порогов' in warning['recommendation'].lower()
        assert warning['current_value'] == 5
    
    def test_high_anomaly_count_warning(self, aggregator):
        """Test warning for very high anomaly count"""
        metrics = {
            'total_anomalies': 6000,  # Above maximum of 5000
            'municipalities_affected': 2000,
            'municipalities_affected_pct': 66.7,
            'anomaly_rate_per_1000': 6000.0,
            'avg_anomalies_per_municipality': 3.0,
            'anomalies_by_severity': {
                'critical': 600,
                'high': 1200,
                'medium': 2400,
                'low': 1800
            },
            'anomalies_by_type': {
                'geographic_anomaly': 6000
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have warning for high anomaly count
        assert len(warnings) > 0
        high_count_warnings = [w for w in warnings if w['warning_type'] == 'high_anomaly_count']
        assert len(high_count_warnings) == 1
        
        warning = high_count_warnings[0]
        assert warning['severity'] == 'warning'
        assert 'слишком много аномалий' in warning['message'].lower()
        assert 'повышение порогов' in warning['recommendation'].lower()
        assert warning['current_value'] == 6000
    
    def test_high_municipalities_affected_critical(self, aggregator):
        """Test critical warning for very high percentage of municipalities affected"""
        metrics = {
            'total_anomalies': 3000,
            'municipalities_affected': 2800,
            'municipalities_affected_pct': 93.3,  # Above maximum of 90%
            'anomaly_rate_per_1000': 1000.0,
            'avg_anomalies_per_municipality': 1.1,
            'anomalies_by_severity': {
                'critical': 300,
                'high': 600,
                'medium': 1200,
                'low': 900
            },
            'anomalies_by_type': {
                'geographic_anomaly': 3000
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have critical warning
        assert len(warnings) > 0
        critical_warnings = [w for w in warnings if w['warning_type'] == 'high_municipalities_affected']
        assert len(critical_warnings) == 1
        
        warning = critical_warnings[0]
        assert warning['severity'] == 'critical'
        assert 'слишком много муниципалитетов' in warning['message'].lower()
        assert warning['current_value'] == 93.3
    
    def test_high_critical_percentage_warning(self, aggregator):
        """Test warning for high percentage of critical anomalies"""
        metrics = {
            'total_anomalies': 1000,
            'municipalities_affected': 300,
            'municipalities_affected_pct': 30.0,
            'anomaly_rate_per_1000': 1000.0,
            'avg_anomalies_per_municipality': 3.3,
            'anomalies_by_severity': {
                'critical': 600,  # 60% critical - above 50% max
                'high': 200,
                'medium': 150,
                'low': 50
            },
            'anomalies_by_type': {
                'geographic_anomaly': 1000
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have warning for high critical percentage
        assert len(warnings) > 0
        critical_pct_warnings = [w for w in warnings if w['warning_type'] == 'high_critical_percentage']
        assert len(critical_pct_warnings) == 1
        
        warning = critical_pct_warnings[0]
        assert warning['severity'] == 'warning'
        assert 'слишком много критических' in warning['message'].lower()
        assert warning['current_value'] == 60.0
    
    def test_low_critical_percentage_info(self, aggregator):
        """Test info message for low percentage of critical anomalies"""
        metrics = {
            'total_anomalies': 1000,
            'municipalities_affected': 300,
            'municipalities_affected_pct': 30.0,
            'anomaly_rate_per_1000': 1000.0,
            'avg_anomalies_per_municipality': 3.3,
            'anomalies_by_severity': {
                'critical': 5,  # 0.5% critical - below 1% min
                'high': 200,
                'medium': 400,
                'low': 395
            },
            'anomalies_by_type': {
                'geographic_anomaly': 1000
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have info for low critical percentage
        assert len(warnings) > 0
        low_critical_warnings = [w for w in warnings if w['warning_type'] == 'low_critical_percentage']
        assert len(low_critical_warnings) == 1
        
        warning = low_critical_warnings[0]
        assert warning['severity'] == 'info'
        assert 'мало критических' in warning['message'].lower()
    
    def test_high_avg_anomalies_per_municipality(self, aggregator):
        """Test warning for high average anomalies per municipality"""
        metrics = {
            'total_anomalies': 2500,
            'municipalities_affected': 100,
            'municipalities_affected_pct': 10.0,
            'anomaly_rate_per_1000': 2500.0,
            'avg_anomalies_per_municipality': 25.0,  # Above maximum of 20
            'anomalies_by_severity': {
                'critical': 250,
                'high': 500,
                'medium': 1000,
                'low': 750
            },
            'anomalies_by_type': {
                'geographic_anomaly': 2500
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have warning for high average
        assert len(warnings) > 0
        high_avg_warnings = [w for w in warnings if w['warning_type'] == 'high_avg_anomalies_per_municipality']
        assert len(high_avg_warnings) == 1
        
        warning = high_avg_warnings[0]
        assert warning['severity'] == 'warning'
        assert 'слишком много аномалий на муниципалитет' in warning['message'].lower()
        assert warning['current_value'] == 25.0
    
    def test_dominant_anomaly_type_info(self, aggregator):
        """Test info message for dominant anomaly type"""
        metrics = {
            'total_anomalies': 1000,
            'municipalities_affected': 300,
            'municipalities_affected_pct': 30.0,
            'anomaly_rate_per_1000': 1000.0,
            'avg_anomalies_per_municipality': 3.3,
            'anomalies_by_severity': {
                'critical': 100,
                'high': 200,
                'medium': 400,
                'low': 300
            },
            'anomalies_by_type': {
                'geographic_anomaly': 800,  # 80% - dominates
                'cross_source_discrepancy': 100,
                'logical_inconsistency': 100
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have info for dominant type
        assert len(warnings) > 0
        dominant_type_warnings = [w for w in warnings if w['warning_type'] == 'dominant_anomaly_type']
        assert len(dominant_type_warnings) == 1
        
        warning = dominant_type_warnings[0]
        assert warning['severity'] == 'info'
        assert 'доминирует' in warning['message'].lower()
        assert 'geographic_anomaly' in warning['message']
    
    def test_multiple_warnings(self, aggregator):
        """Test that multiple warnings can be generated simultaneously"""
        metrics = {
            'total_anomalies': 6000,  # High count
            'municipalities_affected': 2800,
            'municipalities_affected_pct': 93.3,  # High percentage
            'anomaly_rate_per_1000': 6000.0,  # High rate
            'avg_anomalies_per_municipality': 2.1,
            'anomalies_by_severity': {
                'critical': 3600,  # 60% critical
                'high': 1200,
                'medium': 900,
                'low': 300
            },
            'anomalies_by_type': {
                'geographic_anomaly': 6000  # 100% dominant
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have multiple warnings
        assert len(warnings) >= 4
        
        # Check that different warning types are present
        warning_types = [w['warning_type'] for w in warnings]
        assert 'high_anomaly_count' in warning_types
        assert 'high_municipalities_affected' in warning_types
        assert 'high_critical_percentage' in warning_types
        assert 'dominant_anomaly_type' in warning_types
    
    def test_custom_expected_ranges(self, aggregator):
        """Test with custom expected ranges in config"""
        custom_config = {
            'expected_anomaly_ranges': {
                'total_anomalies': {
                    'min': 100,
                    'max': 1000,
                    'optimal_min': 200,
                    'optimal_max': 800
                }
            }
        }
        
        metrics = {
            'total_anomalies': 1500,  # Above custom max of 1000
            'municipalities_affected': 300,
            'municipalities_affected_pct': 30.0,
            'anomaly_rate_per_1000': 1500.0,
            'avg_anomalies_per_municipality': 5.0,
            'anomalies_by_severity': {
                'critical': 150,
                'high': 300,
                'medium': 600,
                'low': 450
            },
            'anomalies_by_type': {
                'geographic_anomaly': 1500
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics, custom_config)
        
        # Should have warning based on custom range
        assert len(warnings) > 0
        high_count_warnings = [w for w in warnings if w['warning_type'] == 'high_anomaly_count']
        assert len(high_count_warnings) == 1
        assert high_count_warnings[0]['expected_range'] == '100-1000'
    
    def test_warning_structure(self, aggregator):
        """Test that warnings have correct structure"""
        metrics = {
            'total_anomalies': 6000,
            'municipalities_affected': 300,
            'municipalities_affected_pct': 30.0,
            'anomaly_rate_per_1000': 6000.0,
            'avg_anomalies_per_municipality': 20.0,
            'anomalies_by_severity': {
                'critical': 600,
                'high': 1200,
                'medium': 2400,
                'low': 1800
            },
            'anomalies_by_type': {
                'geographic_anomaly': 6000
            }
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Check structure of each warning
        for warning in warnings:
            assert 'warning_type' in warning
            assert 'severity' in warning
            assert 'message' in warning
            assert 'recommendation' in warning
            assert 'affected_metric' in warning
            assert 'current_value' in warning
            assert 'expected_range' in warning
            
            # Check severity values
            assert warning['severity'] in ['info', 'warning', 'critical']
            
            # Check that messages and recommendations are not empty
            assert len(warning['message']) > 0
            assert len(warning['recommendation']) > 0
    
    def test_empty_metrics(self, aggregator):
        """Test with empty metrics dictionary"""
        metrics = {}
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should handle empty metrics gracefully
        assert isinstance(warnings, list)
        # May have warnings about missing data
    
    def test_zero_anomalies(self, aggregator):
        """Test with zero anomalies"""
        metrics = {
            'total_anomalies': 0,
            'municipalities_affected': 0,
            'municipalities_affected_pct': 0.0,
            'anomaly_rate_per_1000': 0.0,
            'avg_anomalies_per_municipality': 0.0,
            'anomalies_by_severity': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'anomalies_by_type': {}
        }
        
        warnings = aggregator.check_anomaly_count_warnings(metrics)
        
        # Should have warning for zero anomalies
        assert len(warnings) > 0
        low_count_warnings = [w for w in warnings if w['warning_type'] == 'low_anomaly_count']
        assert len(low_count_warnings) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
