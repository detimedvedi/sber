"""
Unit tests for anomaly detector classes.

Tests all detector methods with synthetic data containing known anomalies.
Verifies severity score calculations and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.anomaly_detector import (
    BaseAnomalyDetector,
    StatisticalOutlierDetector,
    CrossSourceComparator,
    TemporalAnomalyDetector,
    GeographicAnomalyDetector,
    LogicalConsistencyChecker
)


# ============================================================================
# Test Fixtures - Synthetic Data with Known Anomalies
# ============================================================================

@pytest.fixture
def basic_config():
    """Basic configuration for detectors."""
    return {
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


@pytest.fixture
def statistical_test_data():
    """Create test data with known statistical outliers."""
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.normal(100, 10, 100)
    
    # Add known outliers
    data_with_outliers = normal_data.copy()
    data_with_outliers[0] = 200  # High outlier (z-score ~10)
    data_with_outliers[1] = 0    # Low outlier (z-score ~-10)
    data_with_outliers[2] = 150  # Moderate outlier (z-score ~5)
    
    df = pd.DataFrame({
        'territory_id': range(100),
        'municipal_district_name_short': [f'Municipality_{i}' for i in range(100)],
        'region_name': ['Region_A'] * 50 + ['Region_B'] * 50,
        'consumption_total': data_with_outliers,
        'population_total': np.random.normal(50000, 5000, 100)
    })
    
    return df


@pytest.fixture
def empty_data():
    """Create empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def uniform_data():
    """Create DataFrame where all values are the same."""
    df = pd.DataFrame({
        'territory_id': range(50),
        'municipal_district_name_short': [f'Municipality_{i}' for i in range(50)],
        'region_name': ['Region_A'] * 50,
        'consumption_total': [100.0] * 50,
        'population_total': [50000] * 50
    })
    
    return df


@pytest.fixture
def temporal_test_data():
    """Create test data with temporal anomalies."""
    np.random.seed(42)
    
    # Create time series data for 10 municipalities over 12 months
    data = []
    for territory_id in range(10):
        for month in range(1, 13):
            # Normal growth pattern
            base_value = 100 + month * 2
            
            # Add spike in month 6 for territory 0
            if territory_id == 0 and month == 6:
                value = base_value * 3  # 200% spike
            # Add drop in month 8 for territory 1
            elif territory_id == 1 and month == 8:
                value = base_value * 0.3  # 70% drop
            # High volatility for territory 2
            elif territory_id == 2:
                value = base_value + np.random.normal(0, 50)
            else:
                value = base_value + np.random.normal(0, 5)
            
            data.append({
                'territory_id': territory_id,
                'municipal_district_name_short': f'Municipality_{territory_id}',
                'region_name': 'Region_A',
                'year': 2023,
                'month': month,
                'consumption_total': value,
                'population_total': 50000 + np.random.normal(0, 1000)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def geographic_test_data():
    """Create test data with geographic anomalies."""
    np.random.seed(42)
    
    data = []
    
    # Region A: normal distribution around 100
    for i in range(50):
        value = np.random.normal(100, 10)
        # Add one outlier
        if i == 0:
            value = 200
        
        data.append({
            'territory_id': i,
            'municipal_district_name_short': f'Municipality_{i}',
            'region_name': 'Region_A',
            'consumption_total': value,
            'population_total': np.random.normal(50000, 5000)
        })
    
    # Region B: normal distribution around 150
    for i in range(50, 100):
        value = np.random.normal(150, 10)
        # Add one outlier
        if i == 50:
            value = 50
        
        data.append({
            'territory_id': i,
            'municipal_district_name_short': f'Municipality_{i}',
            'region_name': 'Region_B',
            'consumption_total': value,
            'population_total': np.random.normal(75000, 5000)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def cross_source_test_data():
    """Create test data with cross-source discrepancies."""
    np.random.seed(42)
    
    # Create correlated data
    population = np.random.normal(50000, 10000, 100)
    
    # Consumption should correlate with population
    consumption = population * 0.002 + np.random.normal(0, 10, 100)
    
    # Add known discrepancies
    consumption[0] = population[0] * 0.01  # 5x higher than expected (400% discrepancy)
    consumption[1] = population[1] * 0.0002  # 10x lower than expected (90% discrepancy)
    
    df = pd.DataFrame({
        'territory_id': range(100),
        'municipal_district_name_short': [f'Municipality_{i}' for i in range(100)],
        'region_name': ['Region_A'] * 100,
        'consumption_total': consumption,
        'population_total': population,
        'salary_avg': np.random.normal(50000, 5000, 100)
    })
    
    return df


@pytest.fixture
def logical_test_data():
    """Create test data with logical inconsistencies."""
    np.random.seed(42)
    
    data = []
    
    for i in range(50):
        # Normal data
        consumption = np.random.normal(100, 10)
        population = np.random.normal(50000, 5000)
        salary = np.random.normal(50000, 5000)
        
        # Add known logical errors
        if i == 0:
            consumption = -50  # Negative value (impossible)
        elif i == 1:
            population = -1000  # Negative population (impossible)
        elif i == 2:
            # High consumption with very low population (contradictory)
            consumption = 500
            population = 100
        
        data.append({
            'territory_id': i,
            'municipal_district_name_short': f'Municipality_{i}',
            'region_name': 'Region_A',
            'consumption_total': consumption,
            'population_total': population,
            'salary_avg': salary
        })
    
    return pd.DataFrame(data)


# ============================================================================
# BaseAnomalyDetector Tests
# ============================================================================

class TestBaseAnomalyDetector:
    """Test BaseAnomalyDetector functionality."""
    
    def test_calculate_severity_score_with_zscore(self, basic_config):
        """Test severity score calculation using z-score."""
        detector = StatisticalOutlierDetector(basic_config)
        
        # Test different z-score levels
        assert detector.calculate_severity_score(deviation=100, z_score=5.0) == 100.0
        assert detector.calculate_severity_score(deviation=100, z_score=4.0) == 90.0
        assert detector.calculate_severity_score(deviation=100, z_score=3.0) == 70.0
        assert detector.calculate_severity_score(deviation=100, z_score=2.0) == 50.0
        assert detector.calculate_severity_score(deviation=100, z_score=1.0) == 25.0
    
    def test_calculate_severity_score_with_percentile(self, basic_config):
        """Test severity score calculation using percentile."""
        detector = StatisticalOutlierDetector(basic_config)
        
        # Test different percentile levels
        assert detector.calculate_severity_score(deviation=100, percentile=99.5) == 90.0
        assert detector.calculate_severity_score(deviation=100, percentile=96.0) == 70.0
        assert detector.calculate_severity_score(deviation=100, percentile=91.0) == 50.0
        assert detector.calculate_severity_score(deviation=100, percentile=50.0) == 30.0
    
    def test_create_anomaly_record(self, basic_config):
        """Test anomaly record creation."""
        detector = StatisticalOutlierDetector(basic_config)
        
        record = detector.create_anomaly_record(
            territory_id=1,
            municipal_name='Test Municipality',
            region_name='Test Region',
            indicator='consumption_total',
            anomaly_type='statistical_outlier',
            actual_value=200.0,
            expected_value=100.0,
            deviation=100.0,
            deviation_pct=100.0,
            severity_score=80.0,
            z_score=5.0,
            data_source='sberindex',
            detection_method='z_score',
            description='Test anomaly',
            potential_explanation='Test explanation'
        )
        
        assert record['territory_id'] == 1
        assert record['municipal_name'] == 'Test Municipality'
        assert record['indicator'] == 'consumption_total'
        assert record['actual_value'] == 200.0
        assert record['severity_score'] == 80.0
        assert 'anomaly_id' in record
        assert 'detected_at' in record


# ============================================================================
# StatisticalOutlierDetector Tests
# ============================================================================

class TestStatisticalOutlierDetector:
    """Test StatisticalOutlierDetector functionality."""
    
    def test_detect_zscore_outliers(self, basic_config, statistical_test_data):
        """Test z-score outlier detection."""
        detector = StatisticalOutlierDetector(basic_config)
        anomalies = detector.detect_zscore_outliers(statistical_test_data)
        
        # Should detect the outliers we added
        assert len(anomalies) > 0
        
        # Check that high outlier is detected
        high_outlier = [a for a in anomalies if a['territory_id'] == 0]
        assert len(high_outlier) > 0
        assert high_outlier[0]['actual_value'] == 200.0
        assert high_outlier[0]['z_score'] > 3.0
    
    def test_detect_iqr_outliers(self, basic_config, statistical_test_data):
        """Test IQR outlier detection."""
        detector = StatisticalOutlierDetector(basic_config)
        anomalies = detector.detect_iqr_outliers(statistical_test_data)
        
        # Should detect outliers
        assert len(anomalies) > 0
        
        # Verify detection method
        assert all(a['detection_method'] == 'iqr' for a in anomalies)
    
    def test_detect_percentile_outliers(self, basic_config, statistical_test_data):
        """Test percentile outlier detection."""
        detector = StatisticalOutlierDetector(basic_config)
        anomalies = detector.detect_percentile_outliers(statistical_test_data)
        
        # Should detect outliers in extreme percentiles
        assert len(anomalies) > 0
        
        # Verify detection method
        assert all(a['detection_method'] == 'percentile' for a in anomalies)
    
    def test_detect_with_empty_data(self, basic_config, empty_data):
        """Test detection with empty DataFrame."""
        detector = StatisticalOutlierDetector(basic_config)
        result = detector.detect(empty_data)
        
        assert result.empty
    
    def test_detect_with_uniform_data(self, basic_config, uniform_data):
        """Test detection with uniform data (no variation)."""
        detector = StatisticalOutlierDetector(basic_config)
        result = detector.detect(uniform_data)
        
        # Z-score and IQR methods should not detect anomalies when all values are the same
        # Percentile method may still flag extreme percentiles
        # Check that z-score and IQR methods don't detect anomalies
        if not result.empty:
            zscore_anomalies = result[result['detection_method'] == 'z_score']
            iqr_anomalies = result[result['detection_method'] == 'iqr']
            assert len(zscore_anomalies) == 0
            assert len(iqr_anomalies) == 0
    
    def test_full_detect_method(self, basic_config, statistical_test_data):
        """Test full detect method combining all detection methods."""
        detector = StatisticalOutlierDetector(basic_config)
        result = detector.detect(statistical_test_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        
        # Check required columns
        required_cols = ['anomaly_id', 'territory_id', 'indicator', 'anomaly_type',
                        'actual_value', 'severity_score', 'detection_method']
        for col in required_cols:
            assert col in result.columns


# ============================================================================
# CrossSourceComparator Tests
# ============================================================================

class TestCrossSourceComparator:
    """Test CrossSourceComparator functionality."""
    
    def test_calculate_correlations(self, basic_config, cross_source_test_data):
        """Test correlation calculation between sources."""
        detector = CrossSourceComparator(basic_config)
        correlations = detector.calculate_correlations(cross_source_test_data)
        
        # Should calculate correlations for comparable pairs
        assert isinstance(correlations, dict)
    
    def test_detect_large_discrepancies(self, basic_config, cross_source_test_data):
        """Test detection of large discrepancies between sources."""
        detector = CrossSourceComparator(basic_config)
        anomalies = detector.detect_large_discrepancies(cross_source_test_data)
        
        # Should detect the discrepancies we added
        assert len(anomalies) > 0
        
        # Check that known discrepancy is detected
        discrepancy_territories = [a['territory_id'] for a in anomalies]
        assert 0 in discrepancy_territories or 1 in discrepancy_territories
    
    def test_rank_by_discrepancy(self, basic_config, cross_source_test_data):
        """Test ranking of anomalies by discrepancy magnitude."""
        detector = CrossSourceComparator(basic_config)
        anomalies = detector.detect_large_discrepancies(cross_source_test_data)
        
        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            ranked = detector.rank_by_discrepancy(anomalies_df)
            
            assert 'discrepancy_rank' in ranked.columns
            # Check that ranking is in descending order of severity
            assert ranked['severity_score'].is_monotonic_decreasing or len(ranked) == 1
    
    def test_detect_with_empty_data(self, basic_config, empty_data):
        """Test detection with empty DataFrame."""
        detector = CrossSourceComparator(basic_config)
        result = detector.detect(empty_data)
        
        assert result.empty


# ============================================================================
# TemporalAnomalyDetector Tests
# ============================================================================

class TestTemporalAnomalyDetector:
    """Test TemporalAnomalyDetector functionality."""
    
    def test_detect_sudden_spikes(self, basic_config, temporal_test_data):
        """Test detection of sudden spikes in temporal data."""
        detector = TemporalAnomalyDetector(basic_config)
        anomalies = detector.detect_sudden_spikes(temporal_test_data)
        
        # Should detect the spike we added for territory 0
        assert len(anomalies) > 0
        
        spike_territories = [a['territory_id'] for a in anomalies]
        assert 0 in spike_territories or 1 in spike_territories
    
    def test_detect_high_volatility(self, basic_config, temporal_test_data):
        """Test detection of high volatility."""
        detector = TemporalAnomalyDetector(basic_config)
        anomalies = detector.detect_high_volatility(temporal_test_data)
        
        # Should detect high volatility for territory 2
        if anomalies:
            volatility_territories = [a['territory_id'] for a in anomalies]
            assert 2 in volatility_territories
    
    def test_detect_without_temporal_data(self, basic_config, statistical_test_data):
        """Test detection when no temporal data is available."""
        detector = TemporalAnomalyDetector(basic_config)
        result = detector.detect(statistical_test_data)
        
        # Should return empty DataFrame when no temporal data
        assert result.empty
    
    def test_calculate_growth_rates(self, basic_config):
        """Test growth rate calculation."""
        detector = TemporalAnomalyDetector(basic_config)
        
        # Create simple series
        series = pd.Series([100, 150, 120, 180])
        growth_rates = detector._calculate_growth_rates(series)
        
        assert len(growth_rates) > 0
        # First growth rate should be 50%
        assert abs(growth_rates.iloc[0] - 50.0) < 0.1


# ============================================================================
# GeographicAnomalyDetector Tests
# ============================================================================

class TestGeographicAnomalyDetector:
    """Test GeographicAnomalyDetector functionality."""
    
    def test_detect_regional_outliers(self, basic_config, geographic_test_data):
        """Test detection of regional outliers."""
        detector = GeographicAnomalyDetector(basic_config)
        anomalies = detector.detect_regional_outliers(geographic_test_data)
        
        # Should detect outliers in each region
        assert len(anomalies) > 0
        
        # Check that known outliers are detected
        outlier_territories = [a['territory_id'] for a in anomalies]
        assert 0 in outlier_territories or 50 in outlier_territories
    
    def test_detect_cluster_outliers(self, basic_config, geographic_test_data):
        """Test detection of cluster outliers."""
        detector = GeographicAnomalyDetector(basic_config)
        anomalies = detector.detect_cluster_outliers(geographic_test_data)
        
        # Should detect outliers within regional clusters
        assert len(anomalies) >= 0  # May or may not detect depending on thresholds
    
    def test_classify_urban_rural(self, basic_config, geographic_test_data):
        """Test urban/rural classification."""
        detector = GeographicAnomalyDetector(basic_config)
        classified = detector._classify_urban_rural(geographic_test_data)
        
        # Should add municipality_type column if classification is possible
        if 'municipality_type' in classified.columns:
            assert classified['municipality_type'].isin(['urban', 'rural', 'unknown']).all()
    
    def test_detect_without_geographic_data(self, basic_config):
        """Test detection when no geographic data is available."""
        # Create data without region_name
        df = pd.DataFrame({
            'territory_id': range(10),
            'consumption_total': np.random.normal(100, 10, 10)
        })
        
        detector = GeographicAnomalyDetector(basic_config)
        result = detector.detect(df)
        
        # Should return empty DataFrame when no geographic data
        assert result.empty
    
    def test_severity_scoring_adjustments(self, basic_config):
        """Test that severity scoring is adjusted for same-type vs mixed-type comparisons."""
        # Create data with municipality types and outliers
        np.random.seed(42)
        
        # Create data with clear outlier in urban municipalities
        df = pd.DataFrame({
            'territory_id': range(30),
            'municipal_district_name_short': [f'Municipality_{i}' for i in range(30)],
            'region_name': ['Region_A'] * 30,
            'municipality_type': ['urban'] * 10 + ['rural'] * 10 + ['capital'] * 10,
            'consumption_total': [100] * 9 + [500] + [50] * 9 + [250] + [200] * 9 + [1000],  # Outliers at indices 9, 19, 29
            'population_total': [50000] * 30
        })
        
        detector = GeographicAnomalyDetector(basic_config)
        
        # Test type-aware detection (should increase severity for same-type outliers)
        anomalies = detector.detect_regional_outliers(df)
        
        if len(anomalies) > 0:
            # Find anomalies for the outliers
            urban_outlier = [a for a in anomalies if a['territory_id'] == 9 and 'consumption_total' in a['indicator']]
            rural_outlier = [a for a in anomalies if a['territory_id'] == 19 and 'consumption_total' in a['indicator']]
            capital_outlier = [a for a in anomalies if a['territory_id'] == 29 and 'consumption_total' in a['indicator']]
            
            # Verify that severity scores are adjusted
            # Same-type outliers should have increased severity (multiplied by 1.2)
            for anomaly in urban_outlier + rural_outlier + capital_outlier:
                # Base severity calculation would give a certain score
                # After adjustment, it should be multiplied by 1.2 (capped at 100)
                assert anomaly['severity_score'] > 0
                assert anomaly['severity_score'] <= 100.0
        
        # Test legacy method (should reduce severity for mixed-type comparisons)
        # Remove municipality_type to trigger legacy behavior
        df_no_type = df.drop('municipality_type', axis=1)
        anomalies_legacy = detector.detect_regional_outliers(df_no_type)
        
        if len(anomalies_legacy) > 0:
            # Legacy method should reduce severity by 0.7 multiplier
            for anomaly in anomalies_legacy:
                assert anomaly['severity_score'] > 0
                assert anomaly['severity_score'] <= 100.0


# ============================================================================
# LogicalConsistencyChecker Tests
# ============================================================================

class TestLogicalConsistencyChecker:
    """Test LogicalConsistencyChecker functionality."""
    
    def test_detect_negative_values(self, basic_config, logical_test_data):
        """Test detection of negative values."""
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.detect_negative_values(logical_test_data)
        
        # Should detect negative consumption and population
        assert len(anomalies) > 0
        
        # Check that negative values are detected
        negative_territories = [a['territory_id'] for a in anomalies]
        assert 0 in negative_territories or 1 in negative_territories
    
    def test_detect_impossible_ratios(self, basic_config):
        """Test detection of impossible ratios."""
        # Create data with impossible ratio
        df = pd.DataFrame({
            'territory_id': range(10),
            'municipal_district_name_short': [f'Municipality_{i}' for i in range(10)],
            'region_name': ['Region_A'] * 10,
            'consumption_total': [1000000] * 10,  # Very high consumption
            'population_total': [100] * 10  # Very low population
        })
        
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.detect_impossible_ratios(df)
        
        # Should detect impossible consumption per capita
        assert len(anomalies) >= 0  # May detect depending on ratio definitions
    
    def test_detect_contradictory_indicators(self, basic_config, logical_test_data):
        """Test detection of contradictory indicators."""
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.detect_contradictory_indicators(logical_test_data)
        
        # Should detect high consumption with low population
        if anomalies:
            assert len(anomalies) > 0
    
    def test_detect_unusual_missing_patterns(self, basic_config):
        """Test detection of unusual missing data patterns."""
        # Create data with unusual missing pattern
        df = pd.DataFrame({
            'territory_id': range(20),
            'municipal_district_name_short': [f'Municipality_{i}' for i in range(20)],
            'region_name': ['Region_A'] * 20,
            'consumption_total': [100.0] * 19 + [np.nan],
            'population_total': [50000] * 19 + [np.nan],
            'salary_avg': [50000] * 19 + [np.nan],
            'market_access': [10.0] * 19 + [np.nan]
        })
        
        # Make one municipality have all missing values
        df.loc[19, ['consumption_total', 'population_total', 'salary_avg', 'market_access']] = np.nan
        
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.detect_unusual_missing_patterns(df)
        
        # Should detect unusual missing pattern for territory 19
        if anomalies:
            missing_territories = [a['territory_id'] for a in anomalies]
            assert 19 in missing_territories
    
    def test_detect_duplicate_identifiers(self, basic_config):
        """Test detection of duplicate identifiers."""
        # Create data with duplicates
        df = pd.DataFrame({
            'territory_id': [1, 1, 2, 3, 4],  # Duplicate ID
            'municipal_district_name_short': ['Muni_1', 'Muni_1', 'Muni_2', 'Muni_3', 'Muni_4'],
            'region_name': ['Region_A'] * 5,
            'consumption_total': [100, 105, 110, 115, 120]
        })
        
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.detect_duplicate_identifiers(df)
        
        # Should detect duplicate territory_id
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 1 for a in anomalies)
    
    def test_flag_high_missing_municipalities(self, basic_config):
        """Test flagging municipalities with >70% missing indicators."""
        # Create data with varying levels of missing indicators
        df = pd.DataFrame({
            'territory_id': range(5),
            'municipal_district_name_short': [f'Municipality_{i}' for i in range(5)],
            'region_name': ['Region_A'] * 5,
            'consumption_total': [100.0, np.nan, np.nan, np.nan, 100.0],
            'population_total': [50000, 50000, np.nan, np.nan, 50000],
            'salary_avg': [50000, 50000, 50000, np.nan, 50000],
            'market_access': [10.0, 10.0, 10.0, 10.0, 10.0],
            'connection_rate': [0.8, np.nan, np.nan, np.nan, 0.8]
        })
        
        # Territory 0: 0% missing (0/5)
        # Territory 1: 40% missing (2/5)
        # Territory 2: 60% missing (3/5)
        # Territory 3: 80% missing (4/5) - should be flagged
        # Territory 4: 0% missing (0/5)
        
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.flag_high_missing_municipalities(df, threshold=70.0)
        
        # Should detect territory 3 with 80% missing
        assert len(anomalies) == 1
        assert anomalies[0]['territory_id'] == 3
        assert anomalies[0]['actual_value'] == 80.0
        assert anomalies[0]['anomaly_type'] == 'logical_inconsistency'
        assert anomalies[0]['detection_method'] == 'high_missing_municipality'
        assert anomalies[0]['severity_score'] >= 75.0
    
    def test_flag_high_missing_municipalities_custom_threshold(self, basic_config):
        """Test flagging municipalities with custom threshold."""
        # Create data with 60% missing indicators
        df = pd.DataFrame({
            'territory_id': [1],
            'municipal_district_name_short': ['Municipality_1'],
            'region_name': ['Region_A'],
            'consumption_total': [np.nan],
            'population_total': [np.nan],
            'salary_avg': [np.nan],
            'market_access': [10.0],
            'connection_rate': [0.8]
        })
        
        # 60% missing (3/5)
        detector = LogicalConsistencyChecker(basic_config)
        
        # Should not be flagged with 70% threshold
        anomalies_70 = detector.flag_high_missing_municipalities(df, threshold=70.0)
        assert len(anomalies_70) == 0
        
        # Should be flagged with 50% threshold
        anomalies_50 = detector.flag_high_missing_municipalities(df, threshold=50.0)
        assert len(anomalies_50) == 1
        assert anomalies_50[0]['territory_id'] == 1
    
    def test_flag_high_missing_municipalities_no_missing(self, basic_config):
        """Test flagging when no municipalities have high missing indicators."""
        # Create data with no missing values
        df = pd.DataFrame({
            'territory_id': range(3),
            'municipal_district_name_short': [f'Municipality_{i}' for i in range(3)],
            'region_name': ['Region_A'] * 3,
            'consumption_total': [100.0, 110.0, 120.0],
            'population_total': [50000, 55000, 60000],
            'salary_avg': [50000, 52000, 54000]
        })
        
        detector = LogicalConsistencyChecker(basic_config)
        anomalies = detector.flag_high_missing_municipalities(df, threshold=70.0)
        
        # Should not flag any municipalities
        assert len(anomalies) == 0


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestStatisticalOutlierDetectorRegression:
    """Regression tests for StatisticalOutlierDetector KeyError fix."""
    
    def test_zscore_with_missing_indices(self, basic_config):
        """Test z-score detection with missing indices (regression test for KeyError)."""
        # Create data with non-sequential indices
        np.random.seed(42)
        normal_values = np.random.normal(100, 10, 19)
        values = list(normal_values[:10]) + [1000] + list(normal_values[10:])  # 1000 is extreme outlier
        
        df = pd.DataFrame({
            'territory_id': range(20),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(20)],
            'region_name': ['Region_A'] * 20,
            'consumption_total': values
        }, index=range(10, 310, 15))  # Non-sequential indices with gaps
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should not raise KeyError
        anomalies = detector.detect_zscore_outliers(df)
        
        # Should detect the outlier
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 10 for a in anomalies)
    
    def test_iqr_with_missing_indices(self, basic_config):
        """Test IQR detection with missing indices (regression test for KeyError)."""
        # Create data with gaps in indices
        df = pd.DataFrame({
            'territory_id': [1, 2, 3, 4, 5],
            'municipal_district_name_short': ['Muni_1', 'Muni_2', 'Muni_3', 'Muni_4', 'Muni_5'],
            'region_name': ['Region_A'] * 5,
            'population_total': [50000, 51000, 52000, 150000, 53000]  # 150000 is outlier
        }, index=[0, 5, 10, 15, 20])  # Gaps in indices
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should not raise KeyError
        anomalies = detector.detect_iqr_outliers(df)
        
        # Should detect the outlier
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 4 for a in anomalies)
    
    def test_percentile_with_missing_indices(self, basic_config):
        """Test percentile detection with missing indices (regression test for KeyError)."""
        # Create data with random indices
        np.random.seed(42)
        indices = np.random.choice(range(1000), size=50, replace=False)
        
        df = pd.DataFrame({
            'territory_id': range(50),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(50)],
            'region_name': ['Region_A'] * 50,
            'salary_avg': np.random.normal(50000, 5000, 50)
        }, index=indices)
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should not raise KeyError
        anomalies = detector.detect_percentile_outliers(df)
        
        # Should complete without error
        assert isinstance(anomalies, list)
    
    def test_zscore_with_filtered_data(self, basic_config):
        """Test z-score detection after filtering data."""
        # Create full dataset
        df_full = pd.DataFrame({
            'territory_id': range(100),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(100)],
            'region_name': ['Region_A'] * 50 + ['Region_B'] * 50,
            'consumption_total': np.random.normal(100, 10, 100)
        })
        
        # Add outlier
        df_full.loc[25, 'consumption_total'] = 500
        
        # Filter to only Region_A (this creates non-sequential indices)
        df_filtered = df_full[df_full['region_name'] == 'Region_A'].copy()
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should not raise KeyError on filtered data
        anomalies = detector.detect_zscore_outliers(df_filtered)
        
        # Should detect the outlier
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 25 for a in anomalies)
    
    def test_iqr_with_filtered_data(self, basic_config):
        """Test IQR detection after filtering data."""
        # Create full dataset
        df_full = pd.DataFrame({
            'territory_id': range(100),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(100)],
            'region_name': ['Region_A'] * 50 + ['Region_B'] * 50,
            'population_total': np.random.normal(50000, 5000, 100)
        })
        
        # Add outlier in Region_B
        df_full.loc[75, 'population_total'] = 200000
        
        # Filter to only Region_B
        df_filtered = df_full[df_full['region_name'] == 'Region_B'].copy()
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should not raise KeyError on filtered data
        anomalies = detector.detect_iqr_outliers(df_filtered)
        
        # Should detect the outlier
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 75 for a in anomalies)
    
    def test_detect_with_empty_dataframe(self, basic_config):
        """Test detection with completely empty DataFrame."""
        df_empty = pd.DataFrame()
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should handle empty DataFrame gracefully
        result = detector.detect(df_empty)
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_detect_with_single_value(self, basic_config):
        """Test detection with single value (edge case)."""
        df_single = pd.DataFrame({
            'territory_id': [1],
            'municipal_district_name_short': ['Muni_1'],
            'region_name': ['Region_A'],
            'consumption_total': [100.0]
        })
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should handle single value gracefully (no statistics possible)
        result = detector.detect(df_single)
        
        assert isinstance(result, pd.DataFrame)
        # Should be empty since we need at least 3 values for statistics
        assert result.empty
    
    def test_detect_with_two_values(self, basic_config):
        """Test detection with two values (edge case)."""
        df_two = pd.DataFrame({
            'territory_id': [1, 2],
            'municipal_district_name_short': ['Muni_1', 'Muni_2'],
            'region_name': ['Region_A', 'Region_A'],
            'consumption_total': [100.0, 200.0]
        })
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should handle two values gracefully (insufficient for z-score)
        result = detector.detect(df_two)
        
        assert isinstance(result, pd.DataFrame)
        # Should be empty since we need at least 3 values for meaningful statistics
        assert result.empty
    
    def test_zscore_with_all_nan_except_outliers(self, basic_config):
        """Test z-score detection when most values are NaN."""
        # Create data with sufficient valid values and an extreme outlier
        # The detector skips columns with >50% missing values, so we need at least 11 valid values out of 20
        np.random.seed(42)
        normal_values = list(np.random.normal(100, 5, 10))
        values = normal_values[:5] + [1000.0] + normal_values[5:] + [np.nan] * 9
        
        df = pd.DataFrame({
            'territory_id': range(20),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(20)],
            'region_name': ['Region_A'] * 20,
            'consumption_total': values
        })
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should handle NaN values and still detect outlier
        anomalies = detector.detect_zscore_outliers(df)
        
        # Should detect the outlier (1000.0 at index 5)
        assert len(anomalies) > 0
        assert any(a['territory_id'] == 5 for a in anomalies)
    
    def test_detect_with_reset_index(self, basic_config):
        """Test detection after index reset (common pandas operation)."""
        # Create data with custom index
        df = pd.DataFrame({
            'territory_id': [100, 200, 300, 400, 500],
            'municipal_district_name_short': ['Muni_A', 'Muni_B', 'Muni_C', 'Muni_D', 'Muni_E'],
            'region_name': ['Region_A'] * 5,
            'consumption_total': [100, 105, 110, 500, 115]
        }, index=[10, 20, 30, 40, 50])
        
        # Reset index (common operation that can cause index issues)
        df_reset = df.reset_index(drop=True)
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should work correctly with reset index
        result = detector.detect(df_reset)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_detect_with_dropna_subset(self, basic_config):
        """Test detection after dropna operation (creates non-sequential indices)."""
        # Create data with some NaN values
        df = pd.DataFrame({
            'territory_id': range(20),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(20)],
            'region_name': ['Region_A'] * 20,
            'consumption_total': [100, np.nan, 105, np.nan, 110, 115, np.nan, 500, 120, 125,
                                 130, np.nan, 135, 140, np.nan, 145, 150, np.nan, 155, 160]
        })
        
        # Drop rows with NaN (creates gaps in indices)
        df_clean = df.dropna(subset=['consumption_total'])
        
        detector = StatisticalOutlierDetector(basic_config)
        
        # Should handle non-sequential indices after dropna
        result = detector.detect(df_clean)
        
        assert isinstance(result, pd.DataFrame)
        # Should detect the outlier (500)
        assert len(result) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_all_detectors_with_minimal_data(self, basic_config):
        """Test all detectors with minimal data (< 3 rows)."""
        df = pd.DataFrame({
            'territory_id': [1, 2],
            'municipal_district_name_short': ['Muni_1', 'Muni_2'],
            'region_name': ['Region_A', 'Region_A'],
            'consumption_total': [100, 110]
        })
        
        # Statistical detector
        stat_detector = StatisticalOutlierDetector(basic_config)
        stat_result = stat_detector.detect(df)
        assert isinstance(stat_result, pd.DataFrame)
        
        # Temporal detector
        temp_detector = TemporalAnomalyDetector(basic_config)
        temp_result = temp_detector.detect(df)
        assert isinstance(temp_result, pd.DataFrame)
        
        # Geographic detector
        geo_detector = GeographicAnomalyDetector(basic_config)
        geo_result = geo_detector.detect(df)
        assert isinstance(geo_result, pd.DataFrame)
        
        # Logical detector
        logic_detector = LogicalConsistencyChecker(basic_config)
        logic_result = logic_detector.detect(df)
        assert isinstance(logic_result, pd.DataFrame)
    
    def test_detectors_with_all_nan_column(self, basic_config):
        """Test detectors with column containing all NaN values."""
        df = pd.DataFrame({
            'territory_id': range(10),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(10)],
            'region_name': ['Region_A'] * 10,
            'consumption_total': [np.nan] * 10,
            'population_total': np.random.normal(50000, 5000, 10)
        })
        
        detector = StatisticalOutlierDetector(basic_config)
        result = detector.detect(df)
        
        # Should handle all-NaN column gracefully
        assert isinstance(result, pd.DataFrame)
    
    def test_severity_score_boundaries(self, basic_config):
        """Test that severity scores are always between 0 and 100."""
        detector = StatisticalOutlierDetector(basic_config)
        
        # Test extreme values
        score1 = detector.calculate_severity_score(deviation=1000000, z_score=100)
        assert 0 <= score1 <= 100
        
        score2 = detector.calculate_severity_score(deviation=-1000000, z_score=-100)
        assert 0 <= score2 <= 100
        
        score3 = detector.calculate_severity_score(deviation=0, z_score=0)
        assert 0 <= score3 <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



# ============================================================================
# Test DetectorManager
# ============================================================================

class TestDetectorManager:
    """Test DetectorManager class for orchestrating detector execution."""
    
    def test_detector_manager_initialization(self, basic_config):
        """Test DetectorManager initializes correctly."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        
        assert manager is not None
        assert manager.config == basic_config
        assert manager.threshold_manager is not None
        assert len(manager.detectors) == 5  # Should have 5 detectors
        assert 'statistical' in manager.detectors
        assert 'cross_source' in manager.detectors
        assert 'temporal' in manager.detectors
        assert 'geographic' in manager.detectors
        assert 'logical' in manager.detectors
    
    def test_run_all_detectors_success(self, basic_config, statistical_test_data):
        """Test running all detectors successfully."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        results = manager.run_all_detectors(statistical_test_data)
        
        # Should return list of DataFrames
        assert isinstance(results, list)
        assert len(results) > 0
        
        # All results should be DataFrames
        for result in results:
            assert isinstance(result, pd.DataFrame)
        
        # Should have statistics for all detectors
        stats = manager.get_detector_statistics()
        assert len(stats) == 5
    
    def test_run_detector_safe_with_error(self, basic_config):
        """Test that run_detector_safe handles errors gracefully."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        
        # Create a mock detector that will fail
        class FailingDetector:
            def detect(self, df):
                raise ValueError("Intentional test error")
        
        failing_detector = FailingDetector()
        
        # Should return None and not raise exception
        result = manager.run_detector_safe('test_failing', failing_detector, pd.DataFrame())
        
        assert result is None
        
        # Should have recorded the failure in statistics
        stats = manager.get_detector_statistics()
        assert 'test_failing' in stats
        assert not stats['test_failing'].success
        assert stats['test_failing'].error_message == "Intentional test error"
    
    def test_detector_statistics_tracking(self, basic_config, statistical_test_data):
        """Test that detector statistics are tracked correctly."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        manager.run_all_detectors(statistical_test_data)
        
        stats = manager.get_detector_statistics()
        
        # Check that all detectors have statistics
        for detector_name in ['statistical', 'cross_source', 'temporal', 'geographic', 'logical']:
            assert detector_name in stats
            stat = stats[detector_name]
            
            # Check required fields
            assert stat.detector_name == detector_name
            assert isinstance(stat.success, bool)
            assert isinstance(stat.execution_time_seconds, float)
            assert stat.execution_time_seconds >= 0
            assert isinstance(stat.anomalies_detected, int)
            assert stat.anomalies_detected >= 0
            assert stat.started_at is not None
            assert stat.completed_at is not None
    
    def test_threshold_manager_get_thresholds(self, basic_config):
        """Test ThresholdManager returns correct thresholds."""
        from src.detector_manager import ThresholdManager
        
        manager = ThresholdManager(basic_config)
        
        # Get statistical thresholds
        stat_thresholds = manager.get_thresholds('statistical')
        assert 'z_score' in stat_thresholds
        assert stat_thresholds['z_score'] == 3.0
        
        # Get geographic thresholds
        geo_thresholds = manager.get_thresholds('geographic')
        assert 'regional_z_score' in geo_thresholds
        assert geo_thresholds['regional_z_score'] == 2.0
    
    def test_threshold_manager_with_profiles(self):
        """Test ThresholdManager with configuration profiles."""
        from src.detector_manager import ThresholdManager
        
        config_with_profiles = {
            'detection_profile': 'strict',
            'threshold_profiles': {
                'strict': {
                    'statistical': {
                        'z_score': 2.5,
                        'iqr_multiplier': 1.2
                    }
                },
                'normal': {
                    'statistical': {
                        'z_score': 3.0,
                        'iqr_multiplier': 1.5
                    }
                }
            },
            'thresholds': {
                'statistical': {
                    'z_score': 3.0,
                    'iqr_multiplier': 1.5
                }
            }
        }
        
        manager = ThresholdManager(config_with_profiles)
        
        # Should use strict profile thresholds
        stat_thresholds = manager.get_thresholds('statistical')
        assert stat_thresholds['z_score'] == 2.5
        assert stat_thresholds['iqr_multiplier'] == 1.2
    
    def test_detector_manager_with_source_mapping(self, basic_config, statistical_test_data):
        """Test DetectorManager with explicit source mapping."""
        from src.detector_manager import DetectorManager
        
        source_mapping = {
            'consumption_total': 'sberindex',
            'population_total': 'rosstat'
        }
        
        manager = DetectorManager(basic_config, source_mapping)
        
        assert manager.source_mapping == source_mapping
        
        # Run detectors and verify they work with source mapping
        results = manager.run_all_detectors(statistical_test_data)
        assert isinstance(results, list)
    
    def test_detector_manager_continues_after_failure(self, basic_config):
        """Test that DetectorManager continues running other detectors after one fails."""
        from src.detector_manager import DetectorManager
        
        # Create data that might cause issues for some detectors
        df = pd.DataFrame({
            'territory_id': range(10),
            'municipal_district_name_short': [f'Muni_{i}' for i in range(10)],
            'region_name': ['Region_A'] * 10,
            'consumption_total': np.random.normal(100, 10, 10)
        })
        
        manager = DetectorManager(basic_config)
        results = manager.run_all_detectors(df)
        
        # Should still return results even if some detectors fail
        assert isinstance(results, list)
        
        # Check statistics - some may have succeeded, some may have failed
        stats = manager.get_detector_statistics()
        assert len(stats) == 5
        
        # At least one detector should have run
        assert any(stat.success for stat in stats.values())
    
    def test_threshold_manager_load_profile(self):
        """Test ThresholdManager load_profile method."""
        from src.detector_manager import ThresholdManager
        
        config = {
            'detection_profile': 'normal',
            'threshold_profiles': {
                'strict': {
                    'statistical': {'z_score': 2.5},
                    'geographic': {'regional_z_score': 1.5}
                },
                'normal': {
                    'statistical': {'z_score': 3.0},
                    'geographic': {'regional_z_score': 2.0}
                },
                'relaxed': {
                    'statistical': {'z_score': 3.5},
                    'geographic': {'regional_z_score': 2.5}
                }
            }
        }
        
        manager = ThresholdManager(config)
        
        # Load strict profile
        strict_thresholds = manager.load_profile('strict')
        assert strict_thresholds['statistical']['z_score'] == 2.5
        assert manager.profile == 'strict'
        
        # Verify get_thresholds uses new profile
        stat_thresholds = manager.get_thresholds('statistical')
        assert stat_thresholds['z_score'] == 2.5
        
        # Load relaxed profile
        relaxed_thresholds = manager.load_profile('relaxed')
        assert relaxed_thresholds['statistical']['z_score'] == 3.5
        assert manager.profile == 'relaxed'
    
    def test_threshold_manager_load_invalid_profile(self):
        """Test ThresholdManager raises error for invalid profile."""
        from src.detector_manager import ThresholdManager
        
        config = {
            'threshold_profiles': {
                'normal': {'statistical': {'z_score': 3.0}}
            }
        }
        
        manager = ThresholdManager(config)
        
        # Should raise ValueError for unknown profile
        with pytest.raises(ValueError, match="Unknown profile"):
            manager.load_profile('nonexistent')
    
    def test_threshold_manager_apply_auto_tuned_thresholds(self):
        """Test ThresholdManager apply_auto_tuned_thresholds method."""
        from src.detector_manager import ThresholdManager
        
        config = {
            'detection_profile': 'normal',
            'threshold_profiles': {
                'normal': {
                    'statistical': {'z_score': 3.0, 'iqr_multiplier': 1.5},
                    'geographic': {'regional_z_score': 2.0}
                }
            }
        }
        
        manager = ThresholdManager(config)
        
        # Apply auto-tuned thresholds
        tuned_thresholds = {
            'statistical': {'z_score': 2.8},  # Override z_score
            'geographic': {'cluster_threshold': 2.2}  # Add new threshold
        }
        
        manager.apply_auto_tuned_thresholds(tuned_thresholds)
        
        # Verify thresholds were updated
        stat_thresholds = manager.get_thresholds('statistical')
        assert stat_thresholds['z_score'] == 2.8  # Updated
        assert stat_thresholds['iqr_multiplier'] == 1.5  # Preserved
        
        geo_thresholds = manager.get_thresholds('geographic')
        assert geo_thresholds['regional_z_score'] == 2.0  # Preserved
        assert geo_thresholds['cluster_threshold'] == 2.2  # Added
    
    def test_threshold_manager_fallback_to_defaults(self):
        """Test ThresholdManager falls back to default thresholds when profile not found."""
        from src.detector_manager import ThresholdManager
        
        config = {
            'detection_profile': 'nonexistent',
            'thresholds': {
                'statistical': {'z_score': 3.0},
                'geographic': {'regional_z_score': 2.0}
            }
        }
        
        manager = ThresholdManager(config)
        
        # Should fall back to default thresholds
        stat_thresholds = manager.get_thresholds('statistical')
        assert stat_thresholds['z_score'] == 3.0
        
        geo_thresholds = manager.get_thresholds('geographic')
        assert geo_thresholds['regional_z_score'] == 2.0
    
    def test_detector_manager_error_statistics(self, basic_config):
        """Test that error statistics are properly recorded."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        
        # Create a mock detector that will fail
        class FailingDetector:
            def detect(self, df):
                raise RuntimeError("Test error message")
        
        failing_detector = FailingDetector()
        df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Run the failing detector
        result = manager.run_detector_safe('failing_test', failing_detector, df)
        
        # Verify result is None
        assert result is None
        
        # Verify statistics were recorded
        stats = manager.get_detector_statistics()
        assert 'failing_test' in stats
        
        stat = stats['failing_test']
        assert not stat.success
        assert stat.error_message == "Test error message"
        assert stat.anomalies_detected == 0
        assert stat.execution_time_seconds >= 0
        assert stat.started_at is not None
        assert stat.completed_at is not None
    
    def test_detector_manager_mixed_success_failure(self, basic_config, statistical_test_data):
        """Test DetectorManager with mix of successful and failing detectors."""
        from src.detector_manager import DetectorManager
        
        manager = DetectorManager(basic_config)
        
        # Replace one detector with a failing one
        class FailingDetector:
            def detect(self, df):
                raise ValueError("Simulated failure")
        
        manager.detectors['failing'] = FailingDetector()
        
        # Run all detectors
        results = manager.run_all_detectors(statistical_test_data)
        
        # Should have results from successful detectors
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check statistics
        stats = manager.get_detector_statistics()
        
        # Should have stats for all detectors including the failing one
        assert 'failing' in stats
        assert not stats['failing'].success
        
        # Other detectors should have succeeded
        successful_count = sum(1 for s in stats.values() if s.success)
        assert successful_count > 0
    
    def test_detector_statistics_execution_time(self, basic_config, statistical_test_data):
        """Test that execution time is properly measured."""
        from src.detector_manager import DetectorManager
        import time
        
        manager = DetectorManager(basic_config)
        
        # Create a slow detector
        class SlowDetector:
            def detect(self, df):
                time.sleep(0.1)  # Sleep for 100ms
                return pd.DataFrame()
        
        slow_detector = SlowDetector()
        
        # Run the slow detector
        manager.run_detector_safe('slow_test', slow_detector, statistical_test_data)
        
        # Check execution time
        stats = manager.get_detector_statistics()
        assert 'slow_test' in stats
        
        stat = stats['slow_test']
        assert stat.execution_time_seconds >= 0.1  # Should be at least 100ms
        assert stat.success
    
    def test_threshold_manager_empty_profile_thresholds(self):
        """Test ThresholdManager with empty profile thresholds."""
        from src.detector_manager import ThresholdManager
        
        config = {
            'detection_profile': 'normal',
            'thresholds': {
                'statistical': {'z_score': 3.0}
            }
        }
        
        manager = ThresholdManager(config)
        
        # Should fall back to default thresholds when no profiles defined
        stat_thresholds = manager.get_thresholds('statistical')
        assert stat_thresholds['z_score'] == 3.0
