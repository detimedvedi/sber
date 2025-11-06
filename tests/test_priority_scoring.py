"""
Tests for priority scoring functionality in ResultsAggregator
"""

import pytest
import pandas as pd
import numpy as np
from src.results_aggregator import ResultsAggregator


class TestPriorityScoring:
    """Test priority score calculation functionality"""
    
    @pytest.fixture
    def config(self):
        """Sample configuration with priority weights"""
        return {
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
    
    @pytest.fixture
    def aggregator(self, config):
        """Create ResultsAggregator with config"""
        return ResultsAggregator(config)
    
    def test_calculate_priority_score_logical_inconsistency(self, aggregator):
        """Test priority score for logical inconsistency anomaly"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'logical_inconsistency',
            'indicator': 'some_indicator'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 1.5 (type weight) * 1.0 (default indicator weight) = 120.0
        assert priority == 120.0
    
    def test_calculate_priority_score_geographic_anomaly(self, aggregator):
        """Test priority score for geographic anomaly (lower priority)"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'geographic_anomaly',
            'indicator': 'some_indicator'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 0.8 (type weight) * 1.0 (default indicator weight) = 64.0
        assert priority == 64.0
    
    def test_calculate_priority_score_population_indicator(self, aggregator):
        """Test priority score for population indicator (high priority)"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'statistical_outlier',
            'indicator': 'population_total'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 1.0 (type weight) * 1.3 (population weight) = 104.0
        assert priority == 104.0
    
    def test_calculate_priority_score_consumption_total(self, aggregator):
        """Test priority score for consumption_total indicator"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'statistical_outlier',
            'indicator': 'consumption_total'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 1.0 (type weight) * 1.2 (consumption weight) = 96.0
        assert priority == 96.0
    
    def test_calculate_priority_score_salary_indicator(self, aggregator):
        """Test priority score for salary indicator"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'cross_source_discrepancy',
            'indicator': 'salary_Финансы'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 1.2 (type weight) * 1.1 (salary weight) = 105.6
        assert pytest.approx(priority, rel=1e-9) == 105.6
    
    def test_calculate_priority_score_combined_high_priority(self, aggregator):
        """Test priority score with both high-priority type and indicator"""
        anomaly = {
            'severity_score': 90.0,
            'anomaly_type': 'logical_inconsistency',
            'indicator': 'population_total'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 90.0 * 1.5 (type weight) * 1.3 (population weight) = 175.5
        assert priority == 175.5
    
    def test_calculate_priority_score_unknown_type(self, aggregator):
        """Test priority score with unknown anomaly type (should use default 1.0)"""
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'unknown_type',
            'indicator': 'some_indicator'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 80.0 * 1.0 (default type weight) * 1.0 (default indicator weight) = 80.0
        assert priority == 80.0
    
    def test_get_indicator_weight_population(self, aggregator):
        """Test indicator weight detection for population indicators"""
        assert aggregator._get_indicator_weight('population_total') == 1.3
        assert aggregator._get_indicator_weight('Population_2023') == 1.3
        assert aggregator._get_indicator_weight('POPULATION') == 1.3
    
    def test_get_indicator_weight_consumption_total(self, aggregator):
        """Test indicator weight detection for consumption_total"""
        assert aggregator._get_indicator_weight('consumption_total') == 1.2
        assert aggregator._get_indicator_weight('CONSUMPTION_TOTAL') == 1.2
    
    def test_get_indicator_weight_salary(self, aggregator):
        """Test indicator weight detection for salary indicators"""
        assert aggregator._get_indicator_weight('salary_Финансы') == 1.1
        assert aggregator._get_indicator_weight('Salary_IT') == 1.1
        assert aggregator._get_indicator_weight('SALARY_AVERAGE') == 1.1
    
    def test_get_indicator_weight_default(self, aggregator):
        """Test indicator weight detection for unknown indicators"""
        assert aggregator._get_indicator_weight('some_random_indicator') == 1.0
        assert aggregator._get_indicator_weight('connection_speed') == 1.0
    
    def test_add_priority_scores_to_dataframe(self, aggregator):
        """Test adding priority scores to a DataFrame of anomalies"""
        # Create sample anomalies DataFrame
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 90.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 2,
                'municipal_name': 'City B',
                'indicator': 'consumption_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 85.0
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 3,
                'municipal_name': 'City C',
                'indicator': 'salary_IT',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 75.0
            }
        ])
        
        # Add priority scores
        result_df = aggregator.add_priority_scores(anomalies_df)
        
        # Check that priority_score column was added
        assert 'priority_score' in result_df.columns
        
        # Check that all rows have priority scores
        assert result_df['priority_score'].notna().all()
        
        # Check that DataFrame is sorted by priority score (descending)
        assert result_df['priority_score'].is_monotonic_decreasing
        
        # Verify specific priority scores
        # A1: 90.0 * 1.5 * 1.3 = 175.5 (highest)
        # A2: 85.0 * 0.8 * 1.2 = 81.6
        # A3: 75.0 * 1.2 * 1.1 = 99.0
        
        assert result_df.iloc[0]['anomaly_id'] == 'A1'  # Highest priority
        assert result_df.iloc[0]['priority_score'] == 175.5
        
        assert result_df.iloc[1]['anomaly_id'] == 'A3'  # Second highest
        assert pytest.approx(result_df.iloc[1]['priority_score'], rel=1e-9) == 99.0
        
        assert result_df.iloc[2]['anomaly_id'] == 'A2'  # Lowest priority
        assert result_df.iloc[2]['priority_score'] == 81.6
    
    def test_add_priority_scores_empty_dataframe(self, aggregator):
        """Test adding priority scores to empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result_df = aggregator.add_priority_scores(empty_df)
        
        # Should return empty DataFrame without errors
        assert result_df.empty
    
    def test_rank_anomalies_uses_priority_score(self, aggregator):
        """Test that rank_anomalies uses priority_score when available"""
        # Create sample anomalies DataFrame with priority scores
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'region_name': 'Region 1',
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 70.0,
                'priority_score': 136.5  # 70 * 1.5 * 1.3
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 2,
                'municipal_name': 'City B',
                'region_name': 'Region 1',
                'indicator': 'consumption_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 90.0,  # Higher severity but lower priority
                'priority_score': 86.4  # 90 * 0.8 * 1.2
            }
        ])
        
        # Rank anomalies
        ranked_df = aggregator.rank_anomalies(anomalies_df)
        
        # Check that ranking columns were added
        assert 'overall_rank' in ranked_df.columns
        assert 'rank_in_type' in ranked_df.columns
        assert 'rank_in_region' in ranked_df.columns
        
        # Check that A1 is ranked higher despite lower severity (due to higher priority)
        assert ranked_df.iloc[0]['anomaly_id'] == 'A1'
        assert ranked_df.iloc[0]['overall_rank'] == 1
        
        assert ranked_df.iloc[1]['anomaly_id'] == 'A2'
        assert ranked_df.iloc[1]['overall_rank'] == 2
    
    def test_aggregator_without_config_uses_defaults(self):
        """Test that aggregator works without config (uses default weights)"""
        aggregator = ResultsAggregator()
        
        anomaly = {
            'severity_score': 80.0,
            'anomaly_type': 'logical_inconsistency',
            'indicator': 'population_total'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Should use default weights
        # Expected: 80.0 * 1.5 * 1.3 = 156.0
        assert priority == 156.0
    
    def test_priority_score_with_zero_severity(self, aggregator):
        """Test priority score calculation with zero severity"""
        anomaly = {
            'severity_score': 0.0,
            'anomaly_type': 'logical_inconsistency',
            'indicator': 'population_total'
        }
        
        priority = aggregator.calculate_priority_score(anomaly)
        
        # Expected: 0.0 * 1.5 * 1.3 = 0.0
        assert priority == 0.0
    
    def test_priority_score_preserves_relative_ordering(self, aggregator):
        """Test that priority scoring preserves relative importance"""
        # Create anomalies with same severity but different types/indicators
        anomalies = [
            {'severity_score': 80.0, 'anomaly_type': 'logical_inconsistency', 'indicator': 'population_total'},
            {'severity_score': 80.0, 'anomaly_type': 'cross_source_discrepancy', 'indicator': 'consumption_total'},
            {'severity_score': 80.0, 'anomaly_type': 'statistical_outlier', 'indicator': 'some_indicator'},
            {'severity_score': 80.0, 'anomaly_type': 'geographic_anomaly', 'indicator': 'some_indicator'}
        ]
        
        priorities = [aggregator.calculate_priority_score(a) for a in anomalies]
        
        # Verify ordering: logical > cross_source > statistical > geographic
        assert priorities[0] > priorities[1]  # logical > cross_source
        assert priorities[1] > priorities[2]  # cross_source > statistical
        assert priorities[2] > priorities[3]  # statistical > geographic


class TestAnomalyGrouping:
    """Test anomaly grouping functionality"""
    
    @pytest.fixture
    def aggregator(self):
        """Create ResultsAggregator"""
        return ResultsAggregator()
    
    def test_group_related_anomalies_single_anomaly(self, aggregator):
        """Test grouping with single anomaly per territory"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 2,
                'municipal_name': 'City B',
                'indicator': 'consumption_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 75.0
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # Check that grouping columns were added
        assert 'anomaly_group_id' in result_df.columns
        assert 'pattern_type' in result_df.columns
        assert 'pattern_description' in result_df.columns
        assert 'related_anomaly_count' in result_df.columns
        
        # Check that each territory has a unique group ID
        assert result_df.loc[0, 'anomaly_group_id'] != result_df.loc[1, 'anomaly_group_id']
        
        # Check that single anomalies are marked as such
        assert result_df.loc[0, 'pattern_type'] == 'single_anomaly'
        assert result_df.loc[1, 'pattern_type'] == 'single_anomaly'
        assert result_df.loc[0, 'related_anomaly_count'] == 1
        assert result_df.loc[1, 'related_anomaly_count'] == 1
    
    def test_group_related_anomalies_same_indicator_multiple_detectors(self, aggregator):
        """Test grouping when same indicator is detected by multiple detectors"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 75.0
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 85.0
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # All anomalies should have the same group ID
        assert result_df['anomaly_group_id'].nunique() == 1
        
        # Pattern should be same_indicator_multiple_detectors
        assert all(result_df['pattern_type'] == 'same_indicator_multiple_detectors')
        
        # All should have count of 3
        assert all(result_df['related_anomaly_count'] == 3)
        
        # Description should mention the indicator
        assert 'population_total' in result_df.loc[0, 'pattern_description']
    
    def test_group_related_anomalies_related_indicators(self, aggregator):
        """Test grouping when multiple related indicators are affected"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'salary_Финансы',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'salary_IT',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 75.0
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'salary_Образование',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 70.0
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # All anomalies should have the same group ID
        assert result_df['anomaly_group_id'].nunique() == 1
        
        # Pattern should be related_indicators
        assert all(result_df['pattern_type'] == 'related_indicators')
        
        # All should have count of 3
        assert all(result_df['related_anomaly_count'] == 3)
        
        # Description should mention salary category
        assert 'salary' in result_df.loc[0, 'pattern_description']
    
    def test_group_related_anomalies_multiple_indicators(self, aggregator):
        """Test grouping when multiple unrelated indicators are affected"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'consumption_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 75.0
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'connection_speed',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 70.0
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # All anomalies should have the same group ID
        assert result_df['anomaly_group_id'].nunique() == 1
        
        # Pattern should be multiple_indicators (systemic issue)
        assert all(result_df['pattern_type'] == 'multiple_indicators')
        
        # All should have count of 3
        assert all(result_df['related_anomaly_count'] == 3)
        
        # Description should mention systemic issue
        assert 'Системная проблема' in result_df.loc[0, 'pattern_description']
    
    def test_group_related_anomalies_multiple_territories(self, aggregator):
        """Test grouping with multiple territories"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 75.0
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 2,
                'municipal_name': 'City B',
                'indicator': 'salary_IT',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 70.0
            },
            {
                'anomaly_id': 'A4',
                'territory_id': 2,
                'municipal_name': 'City B',
                'indicator': 'salary_Финансы',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 65.0
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # Should have 2 unique group IDs (one per territory)
        assert result_df['anomaly_group_id'].nunique() == 2
        
        # Territory 1 anomalies should have same group ID
        territory_1_groups = result_df[result_df['territory_id'] == 1]['anomaly_group_id'].unique()
        assert len(territory_1_groups) == 1
        
        # Territory 2 anomalies should have same group ID (different from territory 1)
        territory_2_groups = result_df[result_df['territory_id'] == 2]['anomaly_group_id'].unique()
        assert len(territory_2_groups) == 1
        assert territory_1_groups[0] != territory_2_groups[0]
        
        # Territory 1 should have pattern: same_indicator_multiple_detectors
        territory_1_pattern = result_df[result_df['territory_id'] == 1]['pattern_type'].iloc[0]
        assert territory_1_pattern == 'same_indicator_multiple_detectors'
        
        # Territory 2 should have pattern: related_indicators (both salary_*)
        territory_2_pattern = result_df[result_df['territory_id'] == 2]['pattern_type'].iloc[0]
        assert territory_2_pattern == 'related_indicators'
    
    def test_group_related_anomalies_empty_dataframe(self, aggregator):
        """Test grouping with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result_df = aggregator.group_related_anomalies(empty_df)
        
        # Should return empty DataFrame without errors
        assert result_df.empty
    
    def test_group_related_anomalies_preserves_original_columns(self, aggregator):
        """Test that grouping preserves all original columns"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'region_name': 'Region 1',
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0,
                'actual_value': 100000,
                'expected_value': 50000
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # Check that all original columns are preserved
        for col in anomalies_df.columns:
            assert col in result_df.columns
        
        # Check that original values are unchanged
        assert result_df.loc[0, 'anomaly_id'] == 'A1'
        assert result_df.loc[0, 'territory_id'] == 1
        assert result_df.loc[0, 'severity_score'] == 80.0
    
    def test_group_related_anomalies_adds_root_cause(self, aggregator):
        """Test that grouping adds root cause analysis"""
        anomalies_df = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'population_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 80.0,
                'description': 'Cross-source discrepancy detected'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'consumption_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 75.0,
                'description': 'Cross-source discrepancy detected'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'municipal_name': 'City A',
                'indicator': 'salary_IT',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 70.0,
                'description': 'Cross-source discrepancy detected'
            }
        ])
        
        result_df = aggregator.group_related_anomalies(anomalies_df)
        
        # Check that root_cause column was added
        assert 'root_cause' in result_df.columns
        
        # Check that root cause is not None
        assert result_df['root_cause'].notna().all()
        
        # With 3 cross-source discrepancies, should identify systematic issue
        root_cause = result_df.loc[0, 'root_cause']
        assert 'Систематическое расхождение' in root_cause or 'источник' in root_cause


class TestRootCauseAnalysis:
    """Test root cause identification functionality"""
    
    @pytest.fixture
    def aggregator(self):
        """Create ResultsAggregator"""
        return ResultsAggregator()
    
    def test_identify_root_cause_missing_data(self, aggregator):
        """Test root cause identification for missing data pattern"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 80.0,
                'description': 'Данные отсутствуют для показателя'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 75.0,
                'description': 'Пропущенные значения обнаружены'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify missing data
        assert 'Данные отсутствуют' in root_cause or 'неполные' in root_cause
    
    def test_identify_root_cause_duplicates(self, aggregator):
        """Test root cause identification for duplicate records"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'temporal_anomaly',
                'severity_score': 70.0,
                'description': 'Дубликаты записей для территории'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'data_quality_issue',
                'severity_score': 65.0,
                'description': 'Повторяющиеся записи обнаружены'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify duplicates
        assert 'Дубликат' in root_cause or 'дубликат' in root_cause
    
    def test_identify_root_cause_systematic_discrepancies(self, aggregator):
        """Test root cause identification for systematic cross-source discrepancies"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 80.0,
                'description': 'Discrepancy between sources'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 75.0,
                'description': 'Discrepancy between sources'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'indicator': 'salary_IT',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 70.0,
                'description': 'Discrepancy between sources'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify systematic cross-source issue
        assert 'Систематическое расхождение' in root_cause
        assert 'источник' in root_cause
    
    def test_identify_root_cause_extreme_values(self, aggregator):
        """Test root cause identification for extreme values"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 95.0,
                'description': 'Extreme outlier detected'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 90.0,
                'description': 'Extreme outlier detected'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify extreme values
        assert 'Экстремальные значения' in root_cause
    
    def test_identify_root_cause_geographic_anomalies(self, aggregator):
        """Test root cause identification for geographic anomalies"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 75.0,
                'description': 'Regional outlier'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 70.0,
                'description': 'Regional outlier'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'indicator': 'salary_IT',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 68.0,
                'description': 'Regional outlier'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify geographic difference
        assert 'региональн' in root_cause or 'отличие' in root_cause
    
    def test_identify_root_cause_multiple_high_severity(self, aggregator):
        """Test root cause identification for multiple high-severity anomalies"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 90.0,
                'description': 'Critical issue'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'cross_source_discrepancy',
                'severity_score': 85.0,
                'description': 'Critical issue'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'indicator': 'salary_IT',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 80.0,
                'description': 'Critical issue'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify multiple critical anomalies
        assert 'Множественные' in root_cause or 'критические' in root_cause
    
    def test_identify_root_cause_single_type(self, aggregator):
        """Test root cause identification for single anomaly type"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 70.0,
                'description': 'Logical issue'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'logical_inconsistency',
                'severity_score': 65.0,
                'description': 'Logical issue'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify the specific type
        assert 'логические несоответствия' in root_cause
    
    def test_identify_root_cause_related_indicators(self, aggregator):
        """Test root cause identification for related indicators (same category)"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'salary_Финансы',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 75.0,
                'description': 'Outlier'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'salary_IT',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 70.0,
                'description': 'Outlier'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'indicator': 'salary_Образование',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 68.0,
                'description': 'Outlier'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify salary category issue
        assert 'зарплат' in root_cause or 'salary' in root_cause.lower()
    
    def test_identify_root_cause_empty_dataframe(self, aggregator):
        """Test root cause identification with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        root_cause = aggregator.identify_root_cause(empty_df)
        
        # Should return unknown cause
        assert root_cause == "Неизвестная причина"
    
    def test_identify_root_cause_complex_pattern(self, aggregator):
        """Test root cause identification for complex pattern (fallback)"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 60.0,
                'description': 'Issue'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'consumption_total',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 55.0,
                'description': 'Issue'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should return complex problem description
        assert 'Комплексная проблема' in root_cause or 'аномали' in root_cause
    
    def test_identify_root_cause_population_category(self, aggregator):
        """Test root cause identification for population category indicators"""
        territory_anomalies = pd.DataFrame([
            {
                'anomaly_id': 'A1',
                'territory_id': 1,
                'indicator': 'population_total',
                'anomaly_type': 'statistical_outlier',
                'severity_score': 70.0,
                'description': 'Issue'
            },
            {
                'anomaly_id': 'A2',
                'territory_id': 1,
                'indicator': 'population_density',
                'anomaly_type': 'geographic_anomaly',
                'severity_score': 65.0,
                'description': 'Issue'
            },
            {
                'anomaly_id': 'A3',
                'territory_id': 1,
                'indicator': 'population_growth',
                'anomaly_type': 'temporal_anomaly',
                'severity_score': 60.0,
                'description': 'Issue'
            }
        ])
        
        root_cause = aggregator.identify_root_cause(territory_anomalies)
        
        # Should identify population category issue
        assert 'населен' in root_cause or 'population' in root_cause.lower()
