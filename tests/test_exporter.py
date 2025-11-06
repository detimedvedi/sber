"""
Unit tests for the exporter module
"""

import os
import pytest
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
from src.exporter import ResultsExporter


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'export': {
            'output_dir': 'output',
            'timestamp_format': '%Y%m%d_%H%M%S',
            'top_n_municipalities': 50
        }
    }


@pytest.fixture
def sample_anomalies():
    """Sample anomalies DataFrame for testing"""
    return pd.DataFrame({
        'anomaly_id': ['a1', 'a2', 'a3'],
        'territory_id': [1001, 1002, 1003],
        'municipal_name': ['Москва', 'Санкт-Петербург', 'Казань'],
        'region_name': ['Московская область', 'Ленинградская область', 'Республика Татарстан'],
        'indicator': ['consumption_продовольствие', 'market_access', 'salary_avg'],
        'anomaly_type': ['statistical_outlier', 'geographic_anomaly', 'temporal_anomaly'],
        'actual_value': [150.5, 75.2, 200.8],
        'expected_value': [100.0, 50.0, 150.0],
        'deviation': [50.5, 25.2, 50.8],
        'deviation_pct': [50.5, 50.4, 33.9],
        'severity_score': [75.5, 65.3, 55.2],
        'z_score': [3.5, 2.8, 2.1],
        'data_source': ['sberindex', 'sberindex', 'rosstat'],
        'detection_method': ['z-score', 'regional_outlier', 'sudden_spike'],
        'description': ['High consumption', 'Low market access', 'Salary spike'],
        'detected_at': [datetime.now(), datetime.now(), datetime.now()]
    })


def test_exporter_initialization(sample_config):
    """Test that exporter initializes correctly"""
    exporter = ResultsExporter(sample_config)
    assert exporter.output_dir == 'output'
    assert exporter.timestamp_format == '%Y%m%d_%H%M%S'
    assert os.path.exists(exporter.output_dir)


def test_export_master_csv(sample_config, sample_anomalies):
    """Test CSV export with sample data"""
    exporter = ResultsExporter(sample_config)
    
    # Export with custom filename
    filepath = exporter.export_master_csv(sample_anomalies, filename='test_anomalies')
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.csv')
    
    # Read back and verify content
    df_read = pd.read_csv(filepath, encoding='utf-8-sig')
    assert len(df_read) == 3
    assert 'municipal_name' in df_read.columns
    assert df_read['municipal_name'].iloc[0] == 'Москва'
    
    # Clean up
    os.remove(filepath)


def test_export_empty_dataframe(sample_config):
    """Test CSV export with empty DataFrame"""
    exporter = ResultsExporter(sample_config)
    
    empty_df = pd.DataFrame()
    filepath = exporter.export_master_csv(empty_df, filename='test_empty')
    
    # Verify file was created with headers
    assert os.path.exists(filepath)
    df_read = pd.read_csv(filepath, encoding='utf-8-sig')
    assert len(df_read) == 0
    assert 'anomaly_id' in df_read.columns
    
    # Clean up
    os.remove(filepath)


def test_russian_text_encoding(sample_config, sample_anomalies):
    """Test that Russian text is properly encoded"""
    exporter = ResultsExporter(sample_config)
    
    filepath = exporter.export_master_csv(sample_anomalies, filename='test_russian')
    
    # Read with UTF-8 encoding
    df_read = pd.read_csv(filepath, encoding='utf-8-sig')
    
    # Verify Russian characters are preserved
    assert 'Москва' in df_read['municipal_name'].values
    assert 'Санкт-Петербург' in df_read['municipal_name'].values
    assert 'Казань' in df_read['municipal_name'].values
    
    # Clean up
    os.remove(filepath)


def test_timestamp_in_filename(sample_config, sample_anomalies):
    """Test that timestamp is added to filename when not specified"""
    exporter = ResultsExporter(sample_config)
    
    filepath = exporter.export_master_csv(sample_anomalies)
    
    # Verify timestamp is in filename
    assert 'anomalies_master_' in filepath
    assert os.path.exists(filepath)
    
    # Clean up
    os.remove(filepath)


@pytest.fixture
def sample_categorized_anomalies(sample_anomalies):
    """Sample categorized anomalies for testing"""
    return {
        'statistical_outliers': sample_anomalies[sample_anomalies['anomaly_type'] == 'statistical_outlier'],
        'temporal_anomalies': sample_anomalies[sample_anomalies['anomaly_type'] == 'temporal_anomaly'],
        'geographic_anomalies': sample_anomalies[sample_anomalies['anomaly_type'] == 'geographic_anomaly'],
        'cross_source_discrepancies': pd.DataFrame(),
        'logical_inconsistencies': pd.DataFrame(),
        'data_quality_issues': pd.DataFrame()
    }


@pytest.fixture
def sample_municipality_scores():
    """Sample municipality scores for testing"""
    return pd.DataFrame({
        'rank': [1, 2, 3],
        'territory_id': [1001, 1002, 1003],
        'municipal_name': ['Москва', 'Санкт-Петербург', 'Казань'],
        'region_name': ['Московская область', 'Ленинградская область', 'Республика Татарстан'],
        'total_anomalies_count': [5, 3, 2],
        'total_severity_score': [250.5, 180.3, 120.8],
        'average_severity_score': [50.1, 60.1, 60.4],
        'max_severity': [75.5, 65.3, 55.2],
        'anomaly_types': [
            ['statistical_outlier', 'temporal_anomaly'],
            ['geographic_anomaly'],
            ['temporal_anomaly']
        ]
    })


@pytest.fixture
def sample_summary_stats():
    """Sample summary statistics for testing"""
    return {
        'total_anomalies': 3,
        'total_municipalities_affected': 3,
        'by_type': {
            'statistical_outlier': 1,
            'geographic_anomaly': 1,
            'temporal_anomaly': 1
        },
        'by_region': {
            'Московская область': 1,
            'Ленинградская область': 1,
            'Республика Татарстан': 1
        },
        'severity_stats': {
            'mean': 65.33,
            'median': 65.3,
            'min': 55.2,
            'max': 75.5,
            'std': 10.2
        },
        'severity_distribution': {
            'Low (0-25)': 0,
            'Medium (25-50)': 0,
            'High (50-75)': 3,
            'Critical (75-100)': 0
        },
        'data_source_distribution': {
            'sberindex': 2,
            'rosstat': 1
        }
    }


def test_export_summary_excel(sample_config, sample_anomalies, sample_categorized_anomalies, 
                               sample_municipality_scores, sample_summary_stats):
    """Test Excel export with multiple sheets"""
    exporter = ResultsExporter(sample_config)
    
    filepath = exporter.export_summary_excel(
        all_anomalies=sample_anomalies,
        categorized_anomalies=sample_categorized_anomalies,
        municipality_scores=sample_municipality_scores,
        summary_stats=sample_summary_stats,
        filename='test_summary'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.xlsx')
    
    # Read Excel file and verify sheets exist
    excel_file = pd.ExcelFile(filepath)
    expected_sheets = [
        'Executive_Summary',
        'Overview',
        'Statistical_Outliers',
        'Temporal_Anomalies',
        'Geographic_Anomalies',
        'Cross_Source_Discrepancies',
        'Logical_Inconsistencies',
        'Data_Quality_Issues',
        'Top_Anomalous_Municipalities',
        'Data_Dictionary'
    ]
    
    for sheet in expected_sheets:
        assert sheet in excel_file.sheet_names, f"Sheet '{sheet}' not found"
    
    # Verify Executive Summary sheet has content
    exec_summary_df = pd.read_excel(filepath, sheet_name='Executive_Summary', header=None)
    assert not exec_summary_df.empty
    # Check for key sections
    exec_summary_text = ' '.join(exec_summary_df.astype(str).values.flatten())
    assert 'EXECUTIVE SUMMARY' in exec_summary_text or 'РЕЗЮМЕ ДЛЯ РУКОВОДСТВА' in exec_summary_text
    assert 'КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ' in exec_summary_text
    assert 'ОСНОВНЫЕ ВЫВОДЫ' in exec_summary_text
    assert 'РЕКОМЕНДАЦИИ' in exec_summary_text
    assert 'ТОП-10 МУНИЦИПАЛИТЕТОВ' in exec_summary_text
    
    # Verify Overview sheet has content
    overview_df = pd.read_excel(filepath, sheet_name='Overview', header=None)
    assert not overview_df.empty
    
    # Verify Statistical_Outliers sheet has data (includes descriptive statistics rows)
    stats_outliers_df = pd.read_excel(filepath, sheet_name='Statistical_Outliers')
    # First row should be the actual anomaly data
    assert stats_outliers_df.iloc[0]['municipal_name'] == 'Москва'
    
    # Verify Top_Anomalous_Municipalities sheet
    top_munis_df = pd.read_excel(filepath, sheet_name='Top_Anomalous_Municipalities')
    assert len(top_munis_df) == 3
    assert 'municipal_name' in top_munis_df.columns
    
    # Verify Data_Dictionary sheet exists
    dict_df = pd.read_excel(filepath, sheet_name='Data_Dictionary', header=None)
    assert not dict_df.empty
    
    # Close the Excel file before cleanup
    excel_file.close()
    
    # Clean up
    os.remove(filepath)


def test_export_summary_excel_with_empty_data(sample_config):
    """Test Excel export with empty DataFrames"""
    exporter = ResultsExporter(sample_config)
    
    empty_df = pd.DataFrame()
    empty_categorized = {
        'statistical_outliers': pd.DataFrame(),
        'temporal_anomalies': pd.DataFrame(),
        'geographic_anomalies': pd.DataFrame(),
        'cross_source_discrepancies': pd.DataFrame(),
        'logical_inconsistencies': pd.DataFrame(),
        'data_quality_issues': pd.DataFrame()
    }
    empty_scores = pd.DataFrame()
    empty_stats = {
        'total_anomalies': 0,
        'total_municipalities_affected': 0,
        'by_type': {},
        'by_region': {},
        'severity_stats': {},
        'severity_distribution': {},
        'data_source_distribution': {}
    }
    
    filepath = exporter.export_summary_excel(
        all_anomalies=empty_df,
        categorized_anomalies=empty_categorized,
        municipality_scores=empty_scores,
        summary_stats=empty_stats,
        filename='test_empty_summary'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    
    # Verify all sheets exist even with empty data
    excel_file = pd.ExcelFile(filepath)
    assert 'Overview' in excel_file.sheet_names
    assert 'Data_Dictionary' in excel_file.sheet_names
    
    # Close the Excel file before cleanup
    excel_file.close()
    
    # Clean up
    os.remove(filepath)



def test_generate_visualizations(sample_config, sample_anomalies, sample_municipality_scores, sample_summary_stats):
    """Test visualization generation"""
    exporter = ResultsExporter(sample_config)
    
    generated_files = exporter.generate_visualizations(
        all_anomalies=sample_anomalies,
        municipality_scores=sample_municipality_scores,
        summary_stats=sample_summary_stats,
        filename_prefix='test_viz'
    )
    
    # Verify all expected visualizations were generated
    expected_viz_types = [
        'anomaly_type_distribution',
        'top_municipalities',
        'geographic_heatmap',
        'severity_distribution'
    ]
    
    for viz_type in expected_viz_types:
        assert viz_type in generated_files, f"Visualization '{viz_type}' not generated"
        filepath = generated_files[viz_type]
        assert os.path.exists(filepath), f"File not found: {filepath}"
        assert filepath.endswith('.png'), f"File is not PNG: {filepath}"
        
        # Clean up
        os.remove(filepath)


def test_generate_visualizations_with_empty_data(sample_config):
    """Test visualization generation with empty data"""
    exporter = ResultsExporter(sample_config)
    
    empty_df = pd.DataFrame()
    empty_scores = pd.DataFrame()
    empty_stats = {
        'by_type': {},
        'by_region': {}
    }
    
    generated_files = exporter.generate_visualizations(
        all_anomalies=empty_df,
        municipality_scores=empty_scores,
        summary_stats=empty_stats,
        filename_prefix='test_empty_viz'
    )
    
    # Should still generate files (with "No Data" messages)
    assert len(generated_files) > 0
    
    # Clean up generated files
    for filepath in generated_files.values():
        if os.path.exists(filepath):
            os.remove(filepath)



def test_generate_methodology_document(sample_config):
    """Test methodology document generation"""
    # Add thresholds to config
    config_with_thresholds = sample_config.copy()
    config_with_thresholds['thresholds'] = {
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
    
    exporter = ResultsExporter(config_with_thresholds)
    
    filepath = exporter.generate_methodology_document(
        config=config_with_thresholds,
        filename='test_methodology'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.md')
    
    # Read and verify content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key sections
    assert '# СберИндекс Anomaly Detection Methodology' in content
    assert '## 1. Overview' in content
    assert '## 2. Data Sources' in content
    assert '## 3. Detection Methods' in content
    assert '### 3.1 Statistical Outlier Detection' in content
    assert '### 3.2 Temporal Anomaly Detection' in content
    assert '### 3.3 Geographic Anomaly Detection' in content
    assert '### 3.4 Cross-Source Comparison' in content
    assert '### 3.5 Logical Consistency Checks' in content
    assert '## 4. Severity Scoring' in content
    assert '## 5. Interpretation Guidelines' in content
    assert '## 6. Configuration Summary' in content
    
    # Check for threshold values
    assert 'z_score: 3.0' in content
    assert 'spike_threshold: 100' in content
    assert 'regional_z_score: 2.0' in content
    
    # Clean up
    os.remove(filepath)


def test_generate_example_cases(sample_config, sample_anomalies):
    """Test example cases document generation"""
    exporter = ResultsExporter(sample_config)
    
    filepath = exporter.generate_example_cases(
        all_anomalies=sample_anomalies,
        filename='test_examples'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.md')
    
    # Read and verify content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key sections
    assert '# Anomaly Detection Example Cases' in content
    assert '## Statistical Outliers' in content
    assert '## Temporal Anomalies' in content
    assert '## Geographic Anomalies' in content
    assert '## Cross-Source Discrepancies' in content
    assert '## Logical Inconsistencies' in content
    assert '## Data Quality Issues' in content
    
    # Check for example case structure
    assert '### Example Cases' in content
    assert '### Potential Causes and Explanations' in content
    assert '### Investigation Recommendations' in content
    
    # Check for actual example data from sample_anomalies
    assert 'Москва' in content or 'Example 1' in content
    
    # Clean up
    os.remove(filepath)


def test_generate_example_cases_with_empty_data(sample_config):
    """Test example cases generation with empty data"""
    exporter = ResultsExporter(sample_config)
    
    empty_df = pd.DataFrame()
    
    filepath = exporter.generate_example_cases(
        all_anomalies=empty_df,
        filename='test_empty_examples'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    
    # Read and verify content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Should still have structure even with no data
    assert '# Anomaly Detection Example Cases' in content
    assert 'No anomaly data available' in content or 'No anomalies of this type were detected' in content
    
    # Clean up
    os.remove(filepath)


def test_generate_readme(sample_config, sample_summary_stats):
    """Test README document generation"""
    exporter = ResultsExporter(sample_config)
    
    # Sample output files dictionary
    output_files = {
        'master_csv': 'output/anomalies_master_20241031_120000.csv',
        'summary_excel': 'output/anomalies_summary_20241031_120000.xlsx',
        'methodology': 'output/methodology_20241031_120000.md',
        'examples': 'output/example_cases_20241031_120000.md',
        'viz_distribution': 'output/viz_20241031_120000_anomaly_type_distribution.png',
        'viz_municipalities': 'output/viz_20241031_120000_top_municipalities.png',
        'viz_heatmap': 'output/viz_20241031_120000_geographic_heatmap.png',
        'viz_severity': 'output/viz_20241031_120000_severity_distribution.png'
    }
    
    filepath = exporter.generate_readme(
        summary_stats=sample_summary_stats,
        output_files=output_files,
        filename='test_README'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    assert filepath.endswith('.md')
    
    # Read and verify content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key sections
    assert '# СберИндекс Anomaly Detection Results' in content
    assert '## Overview' in content
    assert '## Key Findings' in content
    assert '## Output Files' in content
    assert '## How to Interpret Results' in content
    assert '## Recommended Workflow' in content
    assert '## Common Questions' in content
    assert '## Technical Details' in content
    
    # Check for summary statistics
    assert 'Total Anomalies Detected:' in content
    assert 'Municipalities Affected:' in content
    
    # Check for anomaly types
    assert 'statistical_outlier' in content
    assert 'temporal_anomaly' in content
    assert 'geographic_anomaly' in content
    
    # Check for interpretation guidance
    assert 'Understanding Severity Scores' in content
    assert 'Understanding Anomaly Types' in content
    
    # Check for workflow recommendations
    assert 'Start with the Overview' in content
    assert 'Review Top Anomalous Municipalities' in content
    
    # Check for Q&A section
    assert 'Q:' in content
    assert 'A:' in content
    
    # Clean up
    os.remove(filepath)


def test_generate_readme_with_empty_stats(sample_config):
    """Test README generation with minimal data"""
    exporter = ResultsExporter(sample_config)
    
    empty_stats = {
        'total_anomalies': 0,
        'total_municipalities_affected': 0,
        'by_type': {},
        'severity_stats': {}
    }
    
    filepath = exporter.generate_readme(
        summary_stats=empty_stats,
        output_files={},
        filename='test_empty_README'
    )
    
    # Verify file was created
    assert os.path.exists(filepath)
    
    # Read and verify content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Should still have structure
    assert '# СберИндекс Anomaly Detection Results' in content
    assert '## Overview' in content
    
    # Clean up
    os.remove(filepath)


def test_all_documentation_methods_together(sample_config, sample_anomalies, sample_summary_stats):
    """Test generating all documentation at once"""
    # Add thresholds to config
    config_with_thresholds = sample_config.copy()
    config_with_thresholds['thresholds'] = {
        'statistical': {'z_score': 3.0, 'iqr_multiplier': 1.5, 'percentile_lower': 1, 'percentile_upper': 99},
        'temporal': {'spike_threshold': 100, 'drop_threshold': -50, 'volatility_multiplier': 2.0},
        'geographic': {'regional_z_score': 2.0, 'cluster_threshold': 2.5},
        'cross_source': {'correlation_threshold': 0.5, 'discrepancy_threshold': 50},
        'logical': {'check_negative_values': True, 'check_impossible_ratios': True}
    }
    
    exporter = ResultsExporter(config_with_thresholds)
    
    # Generate all documentation
    methodology_path = exporter.generate_methodology_document(
        config=config_with_thresholds,
        filename='test_all_methodology'
    )
    
    examples_path = exporter.generate_example_cases(
        all_anomalies=sample_anomalies,
        filename='test_all_examples'
    )
    
    output_files = {
        'methodology': methodology_path,
        'examples': examples_path
    }
    
    readme_path = exporter.generate_readme(
        summary_stats=sample_summary_stats,
        output_files=output_files,
        filename='test_all_README'
    )
    
    # Verify all files were created
    assert os.path.exists(methodology_path)
    assert os.path.exists(examples_path)
    assert os.path.exists(readme_path)
    
    # Verify they are all markdown files
    assert methodology_path.endswith('.md')
    assert examples_path.endswith('.md')
    assert readme_path.endswith('.md')
    
    # Clean up
    os.remove(methodology_path)
    os.remove(examples_path)
    os.remove(readme_path)
