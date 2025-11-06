"""
Tests for ExecutiveSummaryGenerator

Tests the executive summary generation functionality including:
- Summary statistics calculation
- Top municipalities identification
- Key findings generation in Russian
- Recommendations generation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.exporter import ExecutiveSummaryGenerator


@pytest.fixture
def sample_anomalies():
    """Create sample anomalies DataFrame for testing."""
    return pd.DataFrame({
        'anomaly_id': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'],
        'territory_id': [1, 1, 2, 2, 3, 3, 4, 5, 6, 7],
        'municipal_name': ['Москва', 'Москва', 'Уфа', 'Уфа', 'Казань', 'Казань', 'Пермь', 'Омск', 'Тюмень', 'Томск'],
        'region_name': ['Москва', 'Москва', 'Башкортостан', 'Башкортостан', 'Татарстан', 'Татарстан', 'Пермский край', 'Омская область', 'Тюменская область', 'Томская область'],
        'indicator': ['population_total', 'salary_total', 'migration_total', 'consumption_total', 'population_total', 'salary_total', 'migration_total', 'consumption_total', 'population_total', 'salary_total'],
        'anomaly_type': ['geographic_anomaly', 'cross_source_discrepancy', 'geographic_anomaly', 'logical_inconsistency', 'geographic_anomaly', 'cross_source_discrepancy', 'statistical_outlier', 'temporal_anomaly', 'data_quality_issue', 'geographic_anomaly'],
        'actual_value': [12000000, 85000, 5000, -100, 1200000, 65000, 3000, 50000, 800000, 55000],
        'expected_value': [10000000, 50000, 1000, 100, 1000000, 50000, 2000, 40000, 750000, 50000],
        'deviation': [2000000, 35000, 4000, 200, 200000, 15000, 1000, 10000, 50000, 5000],
        'deviation_pct': [20.0, 70.0, 400.0, 200.0, 20.0, 30.0, 50.0, 25.0, 6.7, 10.0],
        'severity_score': [95.0, 85.0, 92.0, 88.0, 78.0, 65.0, 55.0, 45.0, 35.0, 25.0],
        'z_score': [5.2, 4.1, 4.8, 4.3, 3.5, 2.8, 2.1, 1.5, 1.0, 0.8],
        'data_source': ['rosstat', 'sberindex', 'rosstat', 'sberindex', 'rosstat', 'sberindex', 'rosstat', 'sberindex', 'rosstat', 'sberindex'],
        'detection_method': ['regional_outlier', 'discrepancy_check', 'cluster_outlier', 'negative_value_check', 'regional_outlier', 'discrepancy_check', 'z_score', 'sudden_spike', 'missing_data', 'regional_outlier'],
        'description': ['Test anomaly'] * 10,
        'detected_at': [datetime.now()] * 10
    })


def test_generate_summary_with_data(sample_anomalies):
    """Test executive summary generation with sample data."""
    generator = ExecutiveSummaryGenerator()
    summary = generator.generate(sample_anomalies)
    
    # Check all required fields are present
    assert 'total_anomalies' in summary
    assert 'critical_count' in summary
    assert 'affected_municipalities' in summary
    assert 'top_10_municipalities' in summary
    assert 'key_findings' in summary
    assert 'recommendations' in summary
    
    # Check values
    assert summary['total_anomalies'] == 10
    assert summary['critical_count'] == 2  # severity > 90 (95.0, 92.0)
    assert summary['affected_municipalities'] == 7
    assert isinstance(summary['top_10_municipalities'], list)
    assert isinstance(summary['key_findings'], list)
    assert isinstance(summary['recommendations'], list)
    
    # Check that findings and recommendations are not empty
    assert len(summary['key_findings']) > 0
    assert len(summary['recommendations']) > 0


def test_generate_summary_empty_data():
    """Test executive summary generation with empty DataFrame."""
    generator = ExecutiveSummaryGenerator()
    empty_df = pd.DataFrame()
    summary = generator.generate(empty_df)
    
    # Check all required fields are present
    assert summary['total_anomalies'] == 0
    assert summary['critical_count'] == 0
    assert summary['affected_municipalities'] == 0
    assert summary['top_10_municipalities'] == []
    assert len(summary['key_findings']) > 0  # Should have at least one finding
    assert len(summary['recommendations']) > 0  # Should have at least one recommendation


def test_identify_top_municipalities(sample_anomalies):
    """Test identification of top municipalities by risk."""
    generator = ExecutiveSummaryGenerator()
    top_municipalities = generator._identify_top_municipalities(sample_anomalies)
    
    # Check that we get a list
    assert isinstance(top_municipalities, list)
    assert len(top_municipalities) <= 10
    
    # Check that municipalities are sorted by risk score
    if len(top_municipalities) > 1:
        for i in range(len(top_municipalities) - 1):
            assert top_municipalities[i]['risk_score'] >= top_municipalities[i + 1]['risk_score']
    
    # Check that each municipality has required fields
    if len(top_municipalities) > 0:
        muni = top_municipalities[0]
        assert 'territory_id' in muni
        assert 'municipal_name' in muni
        assert 'region_name' in muni
        assert 'risk_score' in muni
        assert 'total_severity' in muni
        assert 'anomaly_count' in muni
        assert 'critical_count' in muni
        assert 'max_severity' in muni
        assert 'avg_severity' in muni
        assert 'anomaly_types' in muni


def test_generate_key_findings(sample_anomalies):
    """Test generation of key findings in Russian."""
    generator = ExecutiveSummaryGenerator()
    findings = generator._generate_key_findings(
        sample_anomalies,
        total_anomalies=10,
        critical_count=4,
        affected_municipalities=7
    )
    
    # Check that findings is a list of strings
    assert isinstance(findings, list)
    assert len(findings) > 0
    assert all(isinstance(f, str) for f in findings)
    
    # Check that findings contain Russian text
    assert any('аномалий' in f.lower() for f in findings)


def test_generate_recommendations(sample_anomalies):
    """Test generation of recommendations in Russian."""
    generator = ExecutiveSummaryGenerator()
    recommendations = generator._generate_recommendations(
        sample_anomalies,
        critical_count=2,
        affected_municipalities=7
    )
    
    # Check that recommendations is a list of strings
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(r, str) for r in recommendations)
    
    # Check that recommendations contain Russian text
    assert any('проверить' in r.lower() or 'рассмотреть' in r.lower() or 'провести' in r.lower() for r in recommendations)


def test_top_municipalities_risk_calculation(sample_anomalies):
    """Test that risk score calculation emphasizes critical anomalies."""
    generator = ExecutiveSummaryGenerator()
    top_municipalities = generator._identify_top_municipalities(sample_anomalies)
    
    # Москва has 2 anomalies with high severity (95, 85)
    # Should be ranked high due to critical anomaly
    moscow = next((m for m in top_municipalities if m['municipal_name'] == 'Москва'), None)
    assert moscow is not None
    assert moscow['critical_count'] == 1  # One anomaly with severity > 90
    assert moscow['anomaly_count'] == 2


def test_key_findings_include_statistics(sample_anomalies):
    """Test that key findings include important statistics."""
    generator = ExecutiveSummaryGenerator()
    findings = generator._generate_key_findings(
        sample_anomalies,
        total_anomalies=10,
        critical_count=2,
        affected_municipalities=7
    )
    
    # Should mention total anomalies
    assert any('10' in f for f in findings)
    
    # Should mention affected municipalities
    assert any('7' in f for f in findings)
    
    # Should mention critical anomalies
    assert any('2' in f or 'критических' in f.lower() for f in findings)


def test_recommendations_based_on_anomaly_types(sample_anomalies):
    """Test that recommendations are tailored to anomaly types."""
    generator = ExecutiveSummaryGenerator()
    recommendations = generator._generate_recommendations(
        sample_anomalies,
        critical_count=2,
        affected_municipalities=7
    )
    
    # Should have type-specific recommendations
    # Since geographic_anomaly is most common in sample data
    assert any('географических' in r.lower() or 'порог' in r.lower() for r in recommendations)
