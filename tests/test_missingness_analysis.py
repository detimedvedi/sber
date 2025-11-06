"""
Tests for missingness analysis functionality.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessor import MissingnessAnalyzer, MissingnessReport, DataPreprocessor


class TestMissingnessAnalyzer:
    """Tests for MissingnessAnalyzer class."""
    
    def test_analyze_no_missing_values(self):
        """Test analysis with complete data (no missing values)."""
        # Create sample data with no missing values
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, 20, 30, 40, 50],
            'indicator2': [100, 200, 300, 400, 500],
            'indicator3': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        
        analyzer = MissingnessAnalyzer()
        report = analyzer.analyze(df, indicators=['indicator1', 'indicator2', 'indicator3'])
        
        # Verify no missing values detected
        assert report.overall_completeness == 1.0
        assert len(report.indicators_with_high_missing) == 0
        assert len(report.municipalities_with_high_missing) == 0
        assert all(pct == 0.0 for pct in report.missing_pct_per_indicator.values())
        assert all(pct == 0.0 for pct in report.missing_pct_per_municipality.values())
    
    def test_analyze_with_missing_values(self):
        """Test analysis with some missing values."""
        # Create sample data with missing values
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, 30, np.nan, 50],  # 40% missing
            'indicator2': [100, 200, np.nan, 400, 500],  # 20% missing
            'indicator3': [1.5, 2.5, 3.5, 4.5, 5.5]      # 0% missing
        })
        
        analyzer = MissingnessAnalyzer()
        report = analyzer.analyze(df, indicators=['indicator1', 'indicator2', 'indicator3'])
        
        # Verify missing percentages per indicator
        assert report.missing_pct_per_indicator['indicator1'] == 40.0
        assert report.missing_pct_per_indicator['indicator2'] == 20.0
        assert report.missing_pct_per_indicator['indicator3'] == 0.0
        
        # Verify overall completeness
        # Total: 15 cells, missing: 3 cells -> 12/15 = 0.8
        assert report.overall_completeness == pytest.approx(0.8, rel=0.01)
        
        # Verify no indicators flagged as high missing (threshold is 50%)
        assert len(report.indicators_with_high_missing) == 0
    
    def test_analyze_high_missing_indicators(self):
        """Test detection of indicators with high missing values."""
        # Create sample data with high missing values in some indicators
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, np.nan, np.nan, 50],  # 60% missing
            'indicator2': [100, 200, 300, 400, 500],          # 0% missing
            'indicator3': [np.nan, np.nan, np.nan, np.nan, 5.5]  # 80% missing
        })
        
        analyzer = MissingnessAnalyzer(high_missing_indicator_threshold=50.0)
        report = analyzer.analyze(df, indicators=['indicator1', 'indicator2', 'indicator3'])
        
        # Verify high missing indicators detected
        assert len(report.indicators_with_high_missing) == 2
        assert 'indicator1' in report.indicators_with_high_missing
        assert 'indicator3' in report.indicators_with_high_missing
        assert 'indicator2' not in report.indicators_with_high_missing
    
    def test_analyze_high_missing_municipalities(self):
        """Test detection of municipalities with high missing indicators."""
        # Create sample data where some municipalities have many missing indicators
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, 30, np.nan, 50],
            'indicator2': [100, np.nan, 300, np.nan, 500],
            'indicator3': [1.5, np.nan, 3.5, 4.5, 5.5],
            'indicator4': [5, np.nan, 15, 20, 25]
        })
        
        analyzer = MissingnessAnalyzer(high_missing_municipality_threshold=70.0)
        report = analyzer.analyze(df, indicators=['indicator1', 'indicator2', 'indicator3', 'indicator4'])
        
        # Territory 2 has 4/4 = 100% missing
        # Territory 4 has 2/4 = 50% missing
        assert len(report.municipalities_with_high_missing) >= 1
        assert 2 in report.municipalities_with_high_missing
        
        # Verify missing percentages
        assert report.missing_pct_per_municipality[2] == 100.0
        assert report.missing_pct_per_municipality[4] == 50.0
    
    def test_analyze_empty_dataframe(self):
        """Test analysis with empty DataFrame."""
        df = pd.DataFrame({
            'territory_id': [],
            'indicator1': [],
            'indicator2': []
        })
        
        analyzer = MissingnessAnalyzer()
        report = analyzer.analyze(df, indicators=['indicator1', 'indicator2'])
        
        # Should handle empty DataFrame gracefully
        assert report.total_municipalities == 0
        assert report.overall_completeness == 1.0  # No data means no missing data
    
    def test_analyze_auto_detect_indicators(self):
        """Test automatic detection of numeric indicators."""
        df = pd.DataFrame({
            'territory_id': [1, 2, 3],
            'name': ['City A', 'City B', 'City C'],  # Non-numeric, should be excluded
            'indicator1': [10, 20, 30],
            'indicator2': [100, np.nan, 300],
            'oktmo': [1001, 1002, 1003]  # ID column, should be excluded
        })
        
        analyzer = MissingnessAnalyzer()
        # Don't specify indicators - should auto-detect
        report = analyzer.analyze(df)
        
        # Should only analyze indicator1 and indicator2 (not territory_id, name, or oktmo)
        assert report.total_indicators == 2
        assert 'indicator1' in report.missing_pct_per_indicator
        assert 'indicator2' in report.missing_pct_per_indicator
        assert 'territory_id' not in report.missing_pct_per_indicator
        assert 'oktmo' not in report.missing_pct_per_indicator


class TestDataPreprocessorMissingness:
    """Tests for missingness analysis integration in DataPreprocessor."""
    
    def test_analyze_missingness_method(self):
        """Test that DataPreprocessor.analyze_missingness works correctly."""
        config = {
            'missing_value_handling': {
                'indicator_threshold': 50.0,
                'municipality_threshold': 70.0
            }
        }
        
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, np.nan, np.nan, 50],  # 60% missing
            'indicator2': [100, 200, 300, 400, 500]           # 0% missing
        })
        
        report = preprocessor.analyze_missingness(df, indicators=['indicator1', 'indicator2'])
        
        # Verify report is returned correctly
        assert isinstance(report, MissingnessReport)
        assert report.total_indicators == 2
        assert len(report.indicators_with_high_missing) == 1
        assert 'indicator1' in report.indicators_with_high_missing
    
    def test_custom_thresholds(self):
        """Test that custom thresholds from config are applied."""
        config = {
            'missing_value_handling': {
                'indicator_threshold': 30.0,  # Lower threshold
                'municipality_threshold': 50.0
            }
        }
        
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, 30, np.nan, 50],  # 40% missing
            'indicator2': [100, 200, 300, 400, 500]       # 0% missing
        })
        
        report = preprocessor.analyze_missingness(df, indicators=['indicator1', 'indicator2'])
        
        # With 30% threshold, indicator1 (40% missing) should be flagged
        assert len(report.indicators_with_high_missing) == 1
        assert 'indicator1' in report.indicators_with_high_missing


class TestIndicatorFiltering:
    """Tests for indicator filtering functionality."""
    
    def test_filter_indicators_no_missing(self):
        """Test filtering when no indicators have high missing values."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, 20, 30, 40, 50],
            'indicator2': [100, 200, 300, 400, 500],
            'indicator3': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df, 
            indicators=['indicator1', 'indicator2', 'indicator3'],
            threshold=50.0
        )
        
        # All indicators should be valid
        assert len(valid) == 3
        assert len(skipped) == 0
        assert 'indicator1' in valid
        assert 'indicator2' in valid
        assert 'indicator3' in valid
    
    def test_filter_indicators_with_high_missing(self):
        """Test filtering indicators with >50% missing values."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, np.nan, np.nan, 50],  # 60% missing
            'indicator2': [100, 200, 300, 400, 500],          # 0% missing
            'indicator3': [np.nan, np.nan, np.nan, np.nan, 5.5]  # 80% missing
        })
        
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df,
            indicators=['indicator1', 'indicator2', 'indicator3'],
            threshold=50.0
        )
        
        # Only indicator2 should be valid
        assert len(valid) == 1
        assert len(skipped) == 2
        assert 'indicator2' in valid
        assert 'indicator1' in skipped
        assert 'indicator3' in skipped
    
    def test_filter_indicators_custom_threshold(self):
        """Test filtering with custom threshold."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'indicator1': [10, np.nan, 30, np.nan, 50],  # 40% missing
            'indicator2': [100, 200, 300, 400, 500],      # 0% missing
            'indicator3': [1.5, np.nan, 3.5, 4.5, 5.5]    # 20% missing
        })
        
        # With 30% threshold, indicator1 should be skipped
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df,
            indicators=['indicator1', 'indicator2', 'indicator3'],
            threshold=30.0
        )
        
        assert len(valid) == 2
        assert len(skipped) == 1
        assert 'indicator2' in valid
        assert 'indicator3' in valid
        assert 'indicator1' in skipped
    
    def test_filter_indicators_auto_detect(self):
        """Test filtering with automatic indicator detection."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'name': ['City A', 'City B', 'City C', 'City D', 'City E'],
            'indicator1': [10, np.nan, np.nan, np.nan, 50],  # 60% missing
            'indicator2': [100, 200, 300, 400, 500],          # 0% missing
            'oktmo': [1001, 1002, 1003, 1004, 1005]
        })
        
        # Don't specify indicators - should auto-detect numeric columns
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df,
            threshold=50.0
        )
        
        # Should only consider indicator1 and indicator2 (not territory_id, name, or oktmo)
        assert 'indicator2' in valid
        assert 'indicator1' in skipped
        assert 'territory_id' not in valid and 'territory_id' not in skipped
        assert 'oktmo' not in valid and 'oktmo' not in skipped
    
    def test_filter_indicators_empty_dataframe(self):
        """Test filtering with empty DataFrame."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [],
            'indicator1': [],
            'indicator2': []
        })
        
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df,
            indicators=['indicator1', 'indicator2'],
            threshold=50.0
        )
        
        # Should handle empty DataFrame gracefully
        # All indicators should be returned as valid since there's no data to evaluate
        assert len(valid) == 2
        assert len(skipped) == 0
    
    def test_filter_indicators_exactly_at_threshold(self):
        """Test filtering when missing percentage equals threshold."""
        config = {}
        preprocessor = DataPreprocessor(config)
        
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4],
            'indicator1': [10, np.nan, 30, np.nan],  # Exactly 50% missing
            'indicator2': [100, 200, 300, 400]        # 0% missing
        })
        
        valid, skipped = preprocessor.filter_indicators_by_missingness(
            df,
            indicators=['indicator1', 'indicator2'],
            threshold=50.0
        )
        
        # Exactly at threshold should be kept (not skipped)
        assert len(valid) == 2
        assert len(skipped) == 0
        assert 'indicator1' in valid
        assert 'indicator2' in valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
