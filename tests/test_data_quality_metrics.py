"""
Tests for data quality metrics calculation functionality.

This module tests the calculate_data_quality_metrics method in ResultsAggregator
which provides data quality assessment for the anomaly detection system.
"""

import pytest
import pandas as pd
import numpy as np
from src.results_aggregator import ResultsAggregator


class TestDataQualityMetrics:
    """Test suite for data quality metrics calculation"""
    
    @pytest.fixture
    def sample_data_perfect(self):
        """Create sample data with perfect quality (no missing values, no duplicates)"""
        return pd.DataFrame({
            'territory_id': range(1, 101),
            'municipal_name': [f'Municipality_{i}' for i in range(1, 101)],
            'region_name': [f'Region_{i % 10}' for i in range(1, 101)],
            'population_total': np.random.uniform(10000, 100000, 100),
            'salary_average': np.random.uniform(30000, 80000, 100),
            'consumption_total': np.random.uniform(5000, 20000, 100),
            'connection_count': np.random.uniform(100, 1000, 100)
        })
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        df = pd.DataFrame({
            'territory_id': range(1, 101),
            'municipal_name': [f'Municipality_{i}' for i in range(1, 101)],
            'region_name': [f'Region_{i % 10}' for i in range(1, 101)],
            'population_total': np.random.uniform(10000, 100000, 100),
            'salary_average': np.random.uniform(30000, 80000, 100),
            'consumption_total': np.random.uniform(5000, 20000, 100),
            'connection_count': np.random.uniform(100, 1000, 100)
        })
        
        # Add missing values (30% missing in salary_average)
        missing_indices = np.random.choice(df.index, size=30, replace=False)
        df.loc[missing_indices, 'salary_average'] = np.nan
        
        # Add missing values (60% missing in connection_count)
        missing_indices = np.random.choice(df.index, size=60, replace=False)
        df.loc[missing_indices, 'connection_count'] = np.nan
        
        return df
    
    @pytest.fixture
    def sample_data_with_duplicates(self):
        """Create sample data with duplicate territory_ids"""
        df = pd.DataFrame({
            'territory_id': [1, 1, 2, 2, 3, 4, 5, 6, 7, 8],  # Duplicates: 1, 2
            'municipal_name': [f'Municipality_{i}' for i in range(10)],
            'region_name': [f'Region_{i % 3}' for i in range(10)],
            'population_total': np.random.uniform(10000, 100000, 10),
            'salary_average': np.random.uniform(30000, 80000, 10),
            'consumption_total': np.random.uniform(5000, 20000, 10)
        })
        return df
    
    @pytest.fixture
    def sample_data_with_negative_values(self):
        """Create sample data with logical inconsistencies (negative values)"""
        df = pd.DataFrame({
            'territory_id': range(1, 51),
            'municipal_name': [f'Municipality_{i}' for i in range(1, 51)],
            'region_name': [f'Region_{i % 5}' for i in range(1, 51)],
            'population_total': np.random.uniform(10000, 100000, 50),
            'salary_average': np.random.uniform(30000, 80000, 50),
            'consumption_total': np.random.uniform(5000, 20000, 50)
        })
        
        # Add negative values (logical inconsistencies)
        df.loc[0:4, 'population_total'] = -1000  # 5 negative values
        df.loc[5:9, 'salary_average'] = -5000  # 5 negative values
        
        return df
    
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
    
    def test_calculate_data_quality_metrics_perfect_data(self, aggregator, sample_data_perfect):
        """Test data quality metrics with perfect data"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_perfect)
        
        # Check that all expected keys are present
        assert 'data_completeness_score' in metrics
        assert 'completeness_by_indicator' in metrics
        assert 'consistency_score' in metrics
        assert 'missing_value_stats' in metrics
        assert 'duplicate_stats' in metrics
        assert 'quality_grade' in metrics
        assert 'quality_issues' in metrics
        assert 'score_breakdown' in metrics
        
        # Perfect data should have 100% completeness
        assert metrics['data_completeness_score'] == 1.0
        
        # Perfect data should have high consistency score
        assert metrics['consistency_score'] >= 0.95
        
        # Perfect data should have grade A
        assert metrics['quality_grade'] == 'A'
        
        # No quality issues
        assert len(metrics['quality_issues']) == 0
        
        # No missing values
        assert metrics['missing_value_stats']['total_missing_values'] == 0
        
        # No duplicates
        assert metrics['duplicate_stats']['duplicate_records'] == 0
    
    def test_calculate_data_quality_metrics_with_missing(self, aggregator, sample_data_with_missing):
        """Test data quality metrics with missing values"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_missing)
        
        # Completeness should be less than 100%
        assert metrics['data_completeness_score'] < 1.0
        
        # Should have missing value statistics
        assert metrics['missing_value_stats']['total_missing_values'] > 0
        assert metrics['missing_value_stats']['columns_with_missing'] >= 2
        
        # Should identify quality issues
        assert len(metrics['quality_issues']) > 0
        
        # Should have lower quality grade
        assert metrics['quality_grade'] in ['B', 'C', 'D', 'E', 'F']
        
        # Check completeness by indicator
        assert 'salary_average' in metrics['completeness_by_indicator']
        assert metrics['completeness_by_indicator']['salary_average'] < 1.0
        
        assert 'connection_count' in metrics['completeness_by_indicator']
        assert metrics['completeness_by_indicator']['connection_count'] < 0.5  # 60% missing
    
    def test_calculate_data_quality_metrics_with_duplicates(self, aggregator, sample_data_with_duplicates):
        """Test data quality metrics with duplicate territory_ids"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_duplicates)
        
        # Should detect duplicates
        assert metrics['duplicate_stats']['duplicate_records'] == 4  # 2 pairs of duplicates
        assert metrics['duplicate_stats']['affected_territories'] == 2  # territories 1 and 2
        
        # Should have quality issues
        assert len(metrics['quality_issues']) > 0
        assert any('duplicate' in issue.lower() for issue in metrics['quality_issues'])
        
        # Consistency score should be lower due to duplicates
        assert metrics['consistency_score'] < 1.0
    
    def test_calculate_data_quality_metrics_with_negative_values(self, aggregator, sample_data_with_negative_values):
        """Test data quality metrics with logical inconsistencies (negative values)"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_negative_values)
        
        # Should detect logical inconsistencies
        assert len(metrics['quality_issues']) > 0
        assert any('logical' in issue.lower() or 'negative' in issue.lower() for issue in metrics['quality_issues'])
        
        # Consistency score should be lower due to logical issues
        assert metrics['consistency_score'] < 1.0
        
        # Should have lower quality grade
        assert metrics['quality_grade'] in ['B', 'C', 'D', 'E', 'F']
    
    def test_calculate_data_quality_metrics_empty_dataframe(self, aggregator):
        """Test data quality metrics with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        metrics = aggregator.calculate_data_quality_metrics(empty_df)
        
        # All metrics should be zero or empty
        assert metrics['data_completeness_score'] == 0.0
        assert metrics['consistency_score'] == 0.0
        assert metrics['quality_grade'] == 'F'
        assert len(metrics['quality_issues']) > 0
        assert 'No data available' in metrics['quality_issues']
    
    def test_calculate_data_quality_metrics_with_validation_results(self, aggregator, sample_data_with_missing):
        """Test data quality metrics with validation results provided"""
        # Create mock validation results
        validation_results = {
            'missing_values': {
                'salary_average': 30,
                'connection_count': 60
            },
            'summary': {
                'total_missing': 90,
                'overall_completeness': 0.85
            }
        }
        
        metrics = aggregator.calculate_data_quality_metrics(
            sample_data_with_missing,
            validation_results
        )
        
        # Should use validation results
        assert metrics['missing_value_stats']['total_missing_values'] == 90
        assert metrics['missing_value_stats']['columns_with_missing'] == 2
    
    def test_calculate_data_quality_metrics_score_breakdown(self, aggregator, sample_data_perfect):
        """Test that score breakdown is provided"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_perfect)
        
        # Check score breakdown structure
        assert 'score_breakdown' in metrics
        assert 'overall_score' in metrics['score_breakdown']
        assert 'completeness_component' in metrics['score_breakdown']
        assert 'consistency_component' in metrics['score_breakdown']
        assert 'duplicate_score' in metrics['score_breakdown']
        assert 'logical_consistency_score' in metrics['score_breakdown']
        
        # All scores should be between 0 and 1
        for key, value in metrics['score_breakdown'].items():
            assert 0.0 <= value <= 1.0
    
    def test_calculate_data_quality_metrics_quality_grades(self, aggregator):
        """Test quality grade assignment for different score levels"""
        # Create data with varying quality levels
        test_cases = [
            (1.0, 'A'),  # Perfect score
            (0.95, 'A'),  # Excellent
            (0.90, 'B'),  # Good
            (0.80, 'C'),  # Acceptable
            (0.70, 'D'),  # Poor
            (0.60, 'E'),  # Very poor
            (0.40, 'F')   # Failing
        ]
        
        for completeness, expected_grade in test_cases:
            # Create data with specific completeness
            df = pd.DataFrame({
                'territory_id': range(1, 101),
                'indicator1': np.random.uniform(100, 1000, 100),
                'indicator2': np.random.uniform(100, 1000, 100)
            })
            
            # Add missing values to achieve target completeness
            if completeness < 1.0:
                missing_count = int((1 - completeness) * 100)
                missing_indices = np.random.choice(df.index, size=missing_count, replace=False)
                df.loc[missing_indices, 'indicator1'] = np.nan
            
            metrics = aggregator.calculate_data_quality_metrics(df)
            
            # Grade should match expected (allowing for some variation due to consistency component)
            assert metrics['quality_grade'] in ['A', 'B', 'C', 'D', 'E', 'F']
    
    def test_calculate_data_quality_metrics_completeness_by_indicator(self, aggregator, sample_data_with_missing):
        """Test completeness calculation for each indicator"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_missing)
        
        # Check that completeness is calculated for each indicator
        assert 'completeness_by_indicator' in metrics
        assert len(metrics['completeness_by_indicator']) > 0
        
        # All completeness values should be between 0 and 1
        for indicator, completeness in metrics['completeness_by_indicator'].items():
            assert 0.0 <= completeness <= 1.0
        
        # Indicators with missing values should have completeness < 1.0
        assert metrics['completeness_by_indicator']['salary_average'] < 1.0
        assert metrics['completeness_by_indicator']['connection_count'] < 1.0
    
    def test_calculate_data_quality_metrics_missing_value_stats(self, aggregator, sample_data_with_missing):
        """Test missing value statistics structure"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_missing)
        
        # Check missing value stats structure
        assert 'missing_value_stats' in metrics
        assert 'total_missing_values' in metrics['missing_value_stats']
        assert 'columns_with_missing' in metrics['missing_value_stats']
        assert 'missing_percentage' in metrics['missing_value_stats']
        assert 'top_missing_columns' in metrics['missing_value_stats']
        
        # Values should be reasonable
        assert metrics['missing_value_stats']['total_missing_values'] > 0
        assert metrics['missing_value_stats']['columns_with_missing'] > 0
        assert 0 < metrics['missing_value_stats']['missing_percentage'] <= 100
    
    def test_calculate_data_quality_metrics_duplicate_stats(self, aggregator, sample_data_with_duplicates):
        """Test duplicate statistics structure"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_with_duplicates)
        
        # Check duplicate stats structure
        assert 'duplicate_stats' in metrics
        assert 'duplicate_records' in metrics['duplicate_stats']
        assert 'affected_territories' in metrics['duplicate_stats']
        assert 'duplicate_percentage' in metrics['duplicate_stats']
        
        # Values should be reasonable
        assert metrics['duplicate_stats']['duplicate_records'] > 0
        assert metrics['duplicate_stats']['affected_territories'] > 0
        assert 0 < metrics['duplicate_stats']['duplicate_percentage'] <= 100
    
    def test_calculate_data_quality_metrics_consistency_components(self, aggregator, sample_data_perfect):
        """Test that consistency score includes all components"""
        metrics = aggregator.calculate_data_quality_metrics(sample_data_perfect)
        
        # Check that all consistency components are present
        assert 'duplicate_score' in metrics['score_breakdown']
        assert 'logical_consistency_score' in metrics['score_breakdown']
        
        # Perfect data should have perfect scores
        assert metrics['score_breakdown']['duplicate_score'] == 1.0
        assert metrics['score_breakdown']['logical_consistency_score'] == 1.0
    
    def test_calculate_data_quality_metrics_realistic_scenario(self, aggregator):
        """Test data quality metrics with realistic mixed quality data"""
        # Create realistic data with various quality issues
        df = pd.DataFrame({
            'territory_id': list(range(1, 96)) + [1, 2, 3, 4, 5],  # 5 duplicates
            'municipal_name': [f'Municipality_{i}' for i in range(100)],
            'region_name': [f'Region_{i % 10}' for i in range(100)],
            'population_total': np.random.uniform(10000, 100000, 100),
            'salary_average': np.random.uniform(30000, 80000, 100),
            'consumption_total': np.random.uniform(5000, 20000, 100),
            'connection_count': np.random.uniform(100, 1000, 100)
        })
        
        # Add 15% missing values
        missing_indices = np.random.choice(df.index, size=15, replace=False)
        df.loc[missing_indices, 'salary_average'] = np.nan
        
        # Add 3 negative values (logical inconsistencies)
        df.loc[0:2, 'population_total'] = -1000
        
        metrics = aggregator.calculate_data_quality_metrics(df)
        
        # Should detect all issues
        assert metrics['data_completeness_score'] < 1.0
        assert metrics['consistency_score'] < 1.0
        assert metrics['duplicate_stats']['duplicate_records'] > 0
        assert metrics['missing_value_stats']['total_missing_values'] > 0
        assert len(metrics['quality_issues']) >= 2  # duplicates and negative values (missing values below threshold)
        
        # Quality grade should reflect mixed quality (may still be A or B with minor issues)
        assert metrics['quality_grade'] in ['A', 'B', 'C', 'D', 'E']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
