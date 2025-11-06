"""
Tests for detection metrics calculation functionality.

This module tests the calculate_detection_metrics method in ResultsAggregator
which provides validation metrics for the anomaly detection system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.results_aggregator import ResultsAggregator


class TestDetectionMetrics:
    """Test suite for detection metrics calculation"""
    
    @pytest.fixture
    def sample_anomalies(self):
        """Create sample anomalies DataFrame for testing"""
        return pd.DataFrame({
            'anomaly_id': [f'anom_{i}' for i in range(100)],
            'territory_id': [i % 30 for i in range(100)],  # 30 unique municipalities
            'municipal_name': [f'Municipality_{i % 30}' for i in range(100)],
            'region_name': [f'Region_{i % 5}' for i in range(100)],
            'indicator': [f'indicator_{i % 10}' for i in range(100)],
            'anomaly_type': [
                'geographic_anomaly' if i < 40 else
                'cross_source_discrepancy' if i < 70 else
                'logical_inconsistency' if i < 90 else
                'statistical_outlier'
                for i in range(100)
            ],
            'severity_score': [
                95 if i < 10 else  # 10 critical
                80 if i < 30 else  # 20 high
                60 if i < 60 else  # 30 medium
                40  # 40 low
                for i in range(100)
            ],
            'actual_value': np.random.uniform(100, 1000, 100),
            'expected_value': np.random.uniform(100, 1000, 100),
            'deviation': np.random.uniform(10, 100, 100),
            'deviation_pct': np.random.uniform(5, 50, 100),
            'z_score': np.random.uniform(2, 5, 100),
            'data_source': ['sberindex' if i % 2 == 0 else 'rosstat' for i in range(100)],
            'detection_method': ['method_1' for _ in range(100)],
            'description': [f'Anomaly description {i}' for i in range(100)],
            'detected_at': [datetime.now() for _ in range(100)]
        })
    
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
    
    def test_calculate_detection_metrics_basic(self, aggregator, sample_anomalies):
        """Test basic detection metrics calculation"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Check that all expected keys are present
        assert 'total_anomalies' in metrics
        assert 'anomalies_by_type' in metrics
        assert 'anomalies_by_severity' in metrics
        assert 'municipalities_affected' in metrics
        assert 'municipalities_affected_pct' in metrics
        assert 'anomaly_rate_per_1000' in metrics
        assert 'anomaly_rate_per_1000_by_type' in metrics
        assert 'avg_anomalies_per_municipality' in metrics
        assert 'severity_distribution' in metrics
        
        # Check total anomalies
        assert metrics['total_anomalies'] == 100
        
        # Check municipalities affected (30 unique territory_ids)
        assert metrics['municipalities_affected'] == 30
        
        # Check percentage of municipalities affected
        expected_pct = (30 / 100) * 100
        assert metrics['municipalities_affected_pct'] == round(expected_pct, 2)
        
        # Check anomaly rate per 1000
        expected_rate = (100 / 100) * 1000
        assert metrics['anomaly_rate_per_1000'] == round(expected_rate, 2)
    
    def test_calculate_detection_metrics_by_type(self, aggregator, sample_anomalies):
        """Test anomalies by type counting"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Check anomalies by type
        assert 'geographic_anomaly' in metrics['anomalies_by_type']
        assert 'cross_source_discrepancy' in metrics['anomalies_by_type']
        assert 'logical_inconsistency' in metrics['anomalies_by_type']
        assert 'statistical_outlier' in metrics['anomalies_by_type']
        
        # Verify counts
        assert metrics['anomalies_by_type']['geographic_anomaly'] == 40
        assert metrics['anomalies_by_type']['cross_source_discrepancy'] == 30
        assert metrics['anomalies_by_type']['logical_inconsistency'] == 20
        assert metrics['anomalies_by_type']['statistical_outlier'] == 10
    
    def test_calculate_detection_metrics_by_severity(self, aggregator, sample_anomalies):
        """Test anomalies by severity counting"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Check severity categories
        assert 'critical' in metrics['anomalies_by_severity']
        assert 'high' in metrics['anomalies_by_severity']
        assert 'medium' in metrics['anomalies_by_severity']
        assert 'low' in metrics['anomalies_by_severity']
        
        # Verify counts (based on sample data)
        assert metrics['anomalies_by_severity']['critical'] == 10  # severity >= 90
        assert metrics['anomalies_by_severity']['high'] == 20  # 70 <= severity < 90
        assert metrics['anomalies_by_severity']['medium'] == 30  # 50 <= severity < 70
        assert metrics['anomalies_by_severity']['low'] == 40  # severity < 50
    
    def test_calculate_detection_metrics_rate_per_1000_by_type(self, aggregator, sample_anomalies):
        """Test anomaly rate per 1000 municipalities by type"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Check rate per 1000 by type
        assert 'geographic_anomaly' in metrics['anomaly_rate_per_1000_by_type']
        assert 'cross_source_discrepancy' in metrics['anomaly_rate_per_1000_by_type']
        
        # Verify rates
        expected_rate_geographic = (40 / 100) * 1000
        assert metrics['anomaly_rate_per_1000_by_type']['geographic_anomaly'] == round(expected_rate_geographic, 2)
        
        expected_rate_cross_source = (30 / 100) * 1000
        assert metrics['anomaly_rate_per_1000_by_type']['cross_source_discrepancy'] == round(expected_rate_cross_source, 2)
    
    def test_calculate_detection_metrics_avg_per_municipality(self, aggregator, sample_anomalies):
        """Test average anomalies per affected municipality"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # 100 anomalies across 30 municipalities
        expected_avg = 100 / 30
        assert metrics['avg_anomalies_per_municipality'] == round(expected_avg, 2)
    
    def test_calculate_detection_metrics_severity_distribution(self, aggregator, sample_anomalies):
        """Test severity distribution with percentages"""
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Check severity distribution structure
        assert 'critical' in metrics['severity_distribution']
        assert 'count' in metrics['severity_distribution']['critical']
        assert 'percentage' in metrics['severity_distribution']['critical']
        
        # Verify percentages
        assert metrics['severity_distribution']['critical']['count'] == 10
        assert metrics['severity_distribution']['critical']['percentage'] == 10.0
        
        assert metrics['severity_distribution']['high']['count'] == 20
        assert metrics['severity_distribution']['high']['percentage'] == 20.0
        
        assert metrics['severity_distribution']['medium']['count'] == 30
        assert metrics['severity_distribution']['medium']['percentage'] == 30.0
        
        assert metrics['severity_distribution']['low']['count'] == 40
        assert metrics['severity_distribution']['low']['percentage'] == 40.0
    
    def test_calculate_detection_metrics_empty_dataframe(self, aggregator):
        """Test detection metrics with empty DataFrame"""
        empty_df = pd.DataFrame()
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            empty_df,
            total_municipalities
        )
        
        # All metrics should be zero or empty
        assert metrics['total_anomalies'] == 0
        assert metrics['anomalies_by_type'] == {}
        assert metrics['anomalies_by_severity']['critical'] == 0
        assert metrics['municipalities_affected'] == 0
        assert metrics['municipalities_affected_pct'] == 0.0
        assert metrics['anomaly_rate_per_1000'] == 0.0
        assert metrics['avg_anomalies_per_municipality'] == 0.0
    
    def test_calculate_detection_metrics_zero_municipalities(self, aggregator, sample_anomalies):
        """Test detection metrics with zero total municipalities"""
        total_municipalities = 0
        
        metrics = aggregator.calculate_detection_metrics(
            sample_anomalies,
            total_municipalities
        )
        
        # Should handle division by zero gracefully
        assert metrics['total_anomalies'] == 100
        assert metrics['municipalities_affected_pct'] == 0.0
        assert metrics['anomaly_rate_per_1000'] == 0.0
        assert metrics['anomaly_rate_per_1000_by_type'] == {}
    
    def test_calculate_detection_metrics_missing_columns(self, aggregator):
        """Test detection metrics with missing columns"""
        # DataFrame without severity_score and anomaly_type
        incomplete_df = pd.DataFrame({
            'anomaly_id': ['anom_1', 'anom_2'],
            'territory_id': [1, 2],
            'indicator': ['ind_1', 'ind_2']
        })
        
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            incomplete_df,
            total_municipalities
        )
        
        # Should handle missing columns gracefully
        assert metrics['total_anomalies'] == 2
        assert metrics['anomalies_by_type'] == {}
        assert metrics['anomalies_by_severity']['critical'] == 0
        assert metrics['municipalities_affected'] == 2
    
    def test_calculate_detection_metrics_high_concentration(self, aggregator):
        """Test detection metrics with high concentration in few municipalities"""
        # 100 anomalies in just 5 municipalities
        concentrated_anomalies = pd.DataFrame({
            'anomaly_id': [f'anom_{i}' for i in range(100)],
            'territory_id': [i % 5 for i in range(100)],  # Only 5 unique municipalities
            'anomaly_type': ['geographic_anomaly' for _ in range(100)],
            'severity_score': [75 for _ in range(100)]
        })
        
        total_municipalities = 1000
        
        metrics = aggregator.calculate_detection_metrics(
            concentrated_anomalies,
            total_municipalities
        )
        
        # Check concentration metrics
        assert metrics['municipalities_affected'] == 5
        assert metrics['municipalities_affected_pct'] == 0.5  # 5/1000 * 100
        assert metrics['avg_anomalies_per_municipality'] == 20.0  # 100/5
        assert metrics['anomaly_rate_per_1000'] == 100.0  # 100/1000 * 1000
    
    def test_calculate_detection_metrics_all_critical(self, aggregator):
        """Test detection metrics when all anomalies are critical"""
        critical_anomalies = pd.DataFrame({
            'anomaly_id': [f'anom_{i}' for i in range(50)],
            'territory_id': [i for i in range(50)],
            'anomaly_type': ['logical_inconsistency' for _ in range(50)],
            'severity_score': [95 for _ in range(50)]  # All critical
        })
        
        total_municipalities = 100
        
        metrics = aggregator.calculate_detection_metrics(
            critical_anomalies,
            total_municipalities
        )
        
        # All should be critical
        assert metrics['anomalies_by_severity']['critical'] == 50
        assert metrics['anomalies_by_severity']['high'] == 0
        assert metrics['anomalies_by_severity']['medium'] == 0
        assert metrics['anomalies_by_severity']['low'] == 0
        
        # Check severity distribution
        assert metrics['severity_distribution']['critical']['percentage'] == 100.0
    
    def test_calculate_detection_metrics_realistic_scenario(self, aggregator):
        """Test detection metrics with realistic scenario"""
        # Simulate realistic detection results
        realistic_anomalies = pd.DataFrame({
            'anomaly_id': [f'anom_{i}' for i in range(500)],
            'territory_id': [i % 150 for i in range(500)],  # 150 municipalities affected
            'anomaly_type': [
                'geographic_anomaly' if i % 4 == 0 else
                'cross_source_discrepancy' if i % 4 == 1 else
                'logical_inconsistency' if i % 4 == 2 else
                'statistical_outlier'
                for i in range(500)
            ],
            'severity_score': [
                np.random.uniform(90, 100) if i % 10 == 0 else  # 10% critical
                np.random.uniform(70, 90) if i % 5 == 0 else  # 20% high
                np.random.uniform(50, 70) if i % 3 == 0 else  # ~33% medium
                np.random.uniform(20, 50)  # ~37% low
                for i in range(500)
            ]
        })
        
        total_municipalities = 3000  # Realistic total
        
        metrics = aggregator.calculate_detection_metrics(
            realistic_anomalies,
            total_municipalities
        )
        
        # Verify realistic metrics
        assert metrics['total_anomalies'] == 500
        assert metrics['municipalities_affected'] == 150
        assert metrics['municipalities_affected_pct'] == 5.0  # 150/3000 * 100
        assert metrics['anomaly_rate_per_1000'] == round((500 / 3000) * 1000, 2)
        assert metrics['avg_anomalies_per_municipality'] == round(500 / 150, 2)
        
        # Verify type distribution
        assert len(metrics['anomalies_by_type']) == 4
        assert sum(metrics['anomalies_by_type'].values()) == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
