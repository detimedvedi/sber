"""
Unit tests for the data_loader module
"""

import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import DataLoader, DataLoadError, DataValidationError


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path


@pytest.fixture
def sample_sberindex_data():
    """Create sample СберИндекс data"""
    connection_data = pd.DataFrame({
        'territory_id': [1001, 1002, 1003],
        'connection_count': [100, 200, 150],
        'connection_type': ['fiber', 'cable', 'fiber']
    })
    
    consumption_data = pd.DataFrame({
        'territory_id': [1001, 1001, 1002, 1002, 1003, 1003],
        'category': ['продовольствие', 'здоровье', 'продовольствие', 'здоровье', 'продовольствие', 'здоровье'],
        'value': [50.5, 30.2, 60.3, 35.1, 55.8, 32.5]
    })
    
    market_access_data = pd.DataFrame({
        'territory_id': [1001, 1002, 1003],
        'market_access': [75.5, 80.2, 70.8]
    })
    
    return {
        'connection': connection_data,
        'consumption': consumption_data,
        'market_access': market_access_data
    }


@pytest.fixture
def sample_rosstat_data():
    """Create sample Росстат data"""
    population_data = pd.DataFrame({
        'territory_id': [1001, 1001, 1002, 1002, 1003, 1003],
        'gender': ['male', 'female', 'male', 'female', 'male', 'female'],
        'value': [50000, 52000, 30000, 31000, 40000, 41000]
    })
    
    migration_data = pd.DataFrame({
        'territory_id': [1001, 1002, 1003],
        'value': [1000, -500, 200]
    })
    
    salary_data = pd.DataFrame({
        'territory_id': [1001, 1001, 1002, 1002, 1003, 1003],
        'okved_name': ['manufacturing', 'services', 'manufacturing', 'services', 'manufacturing', 'services'],
        'value': [50000, 45000, 48000, 42000, 52000, 46000]
    })
    
    return {
        'population': population_data,
        'migration': migration_data,
        'salary': salary_data
    }


@pytest.fixture
def sample_municipal_dict():
    """Create sample municipal dictionary"""
    return pd.DataFrame({
        'territory_id': [1001, 1002, 1003],
        'municipal_district_name_short': ['Москва', 'Санкт-Петербург', 'Казань'],
        'region_name': ['Московская область', 'Ленинградская область', 'Республика Татарстан'],
        'oktmo': ['45000000', '40000000', '92000000']
    })


def create_test_files(temp_dir, sberindex_data, rosstat_data, municipal_dict):
    """Helper function to create test parquet and Excel files"""
    # Create СберИндекс files
    sberindex_data['connection'].to_parquet(temp_dir / 'connection.parquet')
    sberindex_data['consumption'].to_parquet(temp_dir / 'consumption.parquet')
    sberindex_data['market_access'].to_parquet(temp_dir / 'market_access.parquet')
    
    # Create Росстат directory and files
    rosstat_dir = temp_dir / 'rosstat'
    rosstat_dir.mkdir()
    rosstat_data['population'].to_parquet(rosstat_dir / '2_bdmo_population.parquet')
    rosstat_data['migration'].to_parquet(rosstat_dir / '3_bdmo_migration.parquet')
    rosstat_data['salary'].to_parquet(rosstat_dir / '4_bdmo_salary.parquet')
    
    # Create municipal dictionary
    municipal_dir = temp_dir / 't_dict_municipal'
    municipal_dir.mkdir()
    municipal_dict.to_excel(municipal_dir / 't_dict_municipal_districts.xlsx', index=False)


class TestDataLoaderInitialization:
    """Test DataLoader initialization"""
    
    def test_initialization_with_default_path(self):
        """Test that DataLoader initializes with default path"""
        loader = DataLoader()
        assert loader.base_path == Path(".")
    
    def test_initialization_with_custom_path(self, temp_test_dir):
        """Test that DataLoader initializes with custom path"""
        loader = DataLoader(str(temp_test_dir))
        assert loader.base_path == temp_test_dir


class TestLoadSberindexData:
    """Test loading СберИндекс data"""
    
    def test_load_all_sberindex_files(self, temp_test_dir, sample_sberindex_data, 
                                      sample_rosstat_data, sample_municipal_dict):
        """Test loading all СберИндекс parquet files"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_sberindex_data()
        
        # Verify all files were loaded
        assert 'connection' in data
        assert 'consumption' in data
        assert 'market_access' in data
        
        # Verify data is not None
        assert data['connection'] is not None
        assert data['consumption'] is not None
        assert data['market_access'] is not None
        
        # Verify data content
        assert len(data['connection']) == 3
        assert len(data['consumption']) == 6
        assert len(data['market_access']) == 3
        
        # Verify columns
        assert 'territory_id' in data['connection'].columns
        assert 'territory_id' in data['consumption'].columns
        assert 'territory_id' in data['market_access'].columns
    
    def test_load_sberindex_with_missing_file(self, temp_test_dir, sample_sberindex_data,
                                              sample_rosstat_data, sample_municipal_dict):
        """Test loading СберИндекс data when one file is missing"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        # Remove one file
        os.remove(temp_test_dir / 'connection.parquet')
        
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_sberindex_data()
        
        # Verify missing file returns None
        assert data['connection'] is None
        
        # Verify other files still loaded
        assert data['consumption'] is not None
        assert data['market_access'] is not None
    
    def test_load_sberindex_with_all_files_missing(self, temp_test_dir):
        """Test loading СберИндекс data when all files are missing"""
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_sberindex_data()
        
        # All should be None
        assert data['connection'] is None
        assert data['consumption'] is None
        assert data['market_access'] is None


class TestLoadRosstatData:
    """Test loading Росстат data"""
    
    def test_load_all_rosstat_files(self, temp_test_dir, sample_sberindex_data,
                                    sample_rosstat_data, sample_municipal_dict):
        """Test loading all Росстат parquet files"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_rosstat_data()
        
        # Verify all files were loaded
        assert 'population' in data
        assert 'migration' in data
        assert 'salary' in data
        
        # Verify data is not None
        assert data['population'] is not None
        assert data['migration'] is not None
        assert data['salary'] is not None
        
        # Verify data content
        assert len(data['population']) == 6
        assert len(data['migration']) == 3
        assert len(data['salary']) == 6
        
        # Verify columns
        assert 'territory_id' in data['population'].columns
        assert 'territory_id' in data['migration'].columns
        assert 'territory_id' in data['salary'].columns
    
    def test_load_rosstat_with_missing_file(self, temp_test_dir, sample_sberindex_data,
                                           sample_rosstat_data, sample_municipal_dict):
        """Test loading Росстат data when one file is missing"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        # Remove one file
        os.remove(temp_test_dir / 'rosstat' / '2_bdmo_population.parquet')
        
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_rosstat_data()
        
        # Verify missing file returns None
        assert data['population'] is None
        
        # Verify other files still loaded
        assert data['migration'] is not None
        assert data['salary'] is not None
    
    def test_load_rosstat_with_missing_directory(self, temp_test_dir):
        """Test loading Росстат data when directory doesn't exist"""
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_rosstat_data()
        
        # All should be None
        assert data['population'] is None
        assert data['migration'] is None
        assert data['salary'] is None


class TestLoadMunicipalDict:
    """Test loading municipal dictionary"""
    
    def test_load_municipal_dict(self, temp_test_dir, sample_sberindex_data,
                                 sample_rosstat_data, sample_municipal_dict):
        """Test loading municipal dictionary Excel file"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_municipal_dict()
        
        # Verify data was loaded
        assert data is not None
        assert len(data) == 3
        
        # Verify columns
        assert 'territory_id' in data.columns
        assert 'municipal_district_name_short' in data.columns
        assert 'region_name' in data.columns
        assert 'oktmo' in data.columns
        
        # Verify content
        assert 'Москва' in data['municipal_district_name_short'].values
        assert 'Санкт-Петербург' in data['municipal_district_name_short'].values
    
    def test_load_municipal_dict_missing_file(self, temp_test_dir):
        """Test loading municipal dictionary when file doesn't exist"""
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_municipal_dict()
        
        # Should return None
        assert data is None
    
    def test_load_municipal_dict_missing_directory(self, temp_test_dir):
        """Test loading municipal dictionary when directory doesn't exist"""
        loader = DataLoader(str(temp_test_dir))
        data = loader.load_municipal_dict()
        
        # Should return None
        assert data is None


class TestMergeDatasets:
    """Test merging datasets"""
    
    def test_merge_all_datasets(self, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict):
        """Test merging all datasets successfully"""
        loader = DataLoader()
        
        merged_df = loader.merge_datasets(
            sberindex=sample_sberindex_data,
            rosstat=sample_rosstat_data,
            municipal=sample_municipal_dict
        )
        
        # Verify merged dataframe
        assert merged_df is not None
        assert len(merged_df) == 3
        
        # Verify municipal columns
        assert 'territory_id' in merged_df.columns
        assert 'municipal_district_name_short' in merged_df.columns
        assert 'region_name' in merged_df.columns
        
        # Verify consumption columns (pivoted)
        assert 'consumption_продовольствие' in merged_df.columns
        assert 'consumption_здоровье' in merged_df.columns
        
        # Verify market access
        assert 'market_access' in merged_df.columns
        
        # Verify population columns
        assert 'population_total' in merged_df.columns
        assert 'population_male' in merged_df.columns
        assert 'population_female' in merged_df.columns
        
        # Verify migration
        assert 'migration_total' in merged_df.columns
        
        # Verify salary columns (pivoted)
        assert 'salary_manufacturing' in merged_df.columns
        assert 'salary_services' in merged_df.columns
    
    def test_merge_without_municipal_dict(self, sample_sberindex_data, sample_rosstat_data):
        """Test merging datasets without municipal dictionary"""
        loader = DataLoader()
        
        merged_df = loader.merge_datasets(
            sberindex=sample_sberindex_data,
            rosstat=sample_rosstat_data,
            municipal=None
        )
        
        # Should still merge successfully
        assert merged_df is not None
        assert 'territory_id' in merged_df.columns
        
        # Should have data columns
        assert 'consumption_продовольствие' in merged_df.columns
        assert 'market_access' in merged_df.columns
    
    def test_merge_with_missing_sberindex_data(self, sample_rosstat_data, sample_municipal_dict):
        """Test merging when some СберИндекс data is missing"""
        loader = DataLoader()
        
        # Set consumption to None
        sberindex_partial = {
            'connection': None,
            'consumption': None,
            'market_access': sample_rosstat_data['population']  # Use any dataframe with territory_id
        }
        
        merged_df = loader.merge_datasets(
            sberindex=sberindex_partial,
            rosstat=sample_rosstat_data,
            municipal=sample_municipal_dict
        )
        
        # Should still merge
        assert merged_df is not None
        assert 'territory_id' in merged_df.columns
    
    def test_merge_with_missing_rosstat_data(self, sample_sberindex_data, sample_municipal_dict):
        """Test merging when some Росстат data is missing"""
        loader = DataLoader()
        
        # Set some rosstat data to None
        rosstat_partial = {
            'population': None,
            'migration': None,
            'salary': None
        }
        
        merged_df = loader.merge_datasets(
            sberindex=sample_sberindex_data,
            rosstat=rosstat_partial,
            municipal=sample_municipal_dict
        )
        
        # Should still merge
        assert merged_df is not None
        assert 'territory_id' in merged_df.columns
        assert 'consumption_продовольствие' in merged_df.columns
    
    def test_merge_with_no_data_raises_error(self):
        """Test that merging with no data raises error"""
        loader = DataLoader()
        
        sberindex_empty = {
            'connection': None,
            'consumption': None,
            'market_access': None
        }
        
        rosstat_empty = {
            'population': None,
            'migration': None,
            'salary': None
        }
        
        with pytest.raises(DataLoadError):
            loader.merge_datasets(
                sberindex=sberindex_empty,
                rosstat=rosstat_empty,
                municipal=None
            )
    
    def test_merge_consumption_pivot(self, sample_municipal_dict):
        """Test that consumption data is correctly pivoted"""
        loader = DataLoader()
        
        consumption_data = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1002],
            'category': ['продовольствие', 'здоровье', 'продовольствие', 'здоровье'],
            'value': [50.5, 30.2, 60.3, 35.1]
        })
        
        sberindex = {
            'connection': None,
            'consumption': consumption_data,
            'market_access': None
        }
        
        rosstat = {
            'population': None,
            'migration': None,
            'salary': None
        }
        
        merged_df = loader.merge_datasets(
            sberindex=sberindex,
            rosstat=rosstat,
            municipal=sample_municipal_dict
        )
        
        # Verify pivot worked
        assert 'consumption_продовольствие' in merged_df.columns
        assert 'consumption_здоровье' in merged_df.columns
        
        # Verify values
        row_1001 = merged_df[merged_df['territory_id'] == 1001].iloc[0]
        assert row_1001['consumption_продовольствие'] == 50.5
        assert row_1001['consumption_здоровье'] == 30.2
    
    def test_merge_salary_pivot(self, sample_municipal_dict):
        """Test that salary data is correctly pivoted"""
        loader = DataLoader()
        
        salary_data = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1002],
            'okved_name': ['manufacturing', 'services', 'manufacturing', 'services'],
            'value': [50000, 45000, 48000, 42000]
        })
        
        sberindex = {
            'connection': None,
            'consumption': None,
            'market_access': None
        }
        
        rosstat = {
            'population': None,
            'migration': None,
            'salary': salary_data
        }
        
        merged_df = loader.merge_datasets(
            sberindex=sberindex,
            rosstat=rosstat,
            municipal=sample_municipal_dict
        )
        
        # Verify pivot worked
        assert 'salary_manufacturing' in merged_df.columns
        assert 'salary_services' in merged_df.columns
        
        # Verify values
        row_1001 = merged_df[merged_df['territory_id'] == 1001].iloc[0]
        assert row_1001['salary_manufacturing'] == 50000
        assert row_1001['salary_services'] == 45000


class TestValidateData:
    """Test data validation"""
    
    def test_validate_complete_data(self, sample_municipal_dict):
        """Test validation of complete data with no issues"""
        loader = DataLoader()
        
        # Create complete dataframe
        df = sample_municipal_dict.copy()
        df['value1'] = [100, 200, 300]
        df['value2'] = [10.5, 20.5, 30.5]
        
        validation_results = loader.validate_data(df)
        
        # Verify validation results structure
        assert 'missing_values' in validation_results
        assert 'duplicate_ids' in validation_results
        assert 'data_types' in validation_results
        assert 'completeness_scores' in validation_results
        assert 'summary' in validation_results
        
        # Verify no issues found
        assert len(validation_results['missing_values']) == 0
        assert len(validation_results['duplicate_ids']) == 0
        
        # Verify summary
        assert validation_results['summary']['total_rows'] == 3
        assert validation_results['summary']['overall_completeness'] == 1.0
    
    def test_validate_data_with_missing_values(self):
        """Test validation detects missing values"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'value1': [100, None, 300],
            'value2': [10.5, 20.5, None]
        })
        
        validation_results = loader.validate_data(df)
        
        # Verify missing values detected
        assert len(validation_results['missing_values']) > 0
        assert 'value1' in validation_results['missing_values']
        assert 'value2' in validation_results['missing_values']
        assert validation_results['missing_values']['value1'] == 1
        assert validation_results['missing_values']['value2'] == 1
        
        # Verify completeness is less than 1
        assert validation_results['summary']['overall_completeness'] < 1.0
    
    def test_validate_data_with_duplicates(self):
        """Test validation detects duplicate territory_ids"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1001, 1003],
            'value1': [100, 200, 150, 300]
        })
        
        validation_results = loader.validate_data(df)
        
        # Verify duplicates detected
        assert len(validation_results['duplicate_ids']) > 0
        assert 1001 in validation_results['duplicate_ids']
    
    def test_validate_data_without_territory_id(self):
        """Test validation when territory_id column is missing"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'value1': [100, 200, 300],
            'value2': [10.5, 20.5, 30.5]
        })
        
        validation_results = loader.validate_data(df)
        
        # Should handle gracefully
        assert validation_results['duplicate_ids'] == []
        assert validation_results['completeness_scores'] == {}
    
    def test_validate_data_completeness_scores(self):
        """Test completeness score calculation"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'value1': [100, None, 300],
            'value2': [10.5, 20.5, None],
            'value3': [1, 2, 3]
        })
        
        validation_results = loader.validate_data(df)
        
        # Verify completeness scores
        assert 1001 in validation_results['completeness_scores']
        assert 1002 in validation_results['completeness_scores']
        assert 1003 in validation_results['completeness_scores']
        
        # Territory 1001 has all values (100% complete)
        assert validation_results['completeness_scores'][1001] == 1.0
        
        # Territory 1002 has 1 missing value (66.7% complete)
        assert validation_results['completeness_scores'][1002] < 1.0
        
        # Territory 1003 has 1 missing value (66.7% complete)
        assert validation_results['completeness_scores'][1003] < 1.0
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe"""
        loader = DataLoader()
        
        df = pd.DataFrame()
        
        validation_results = loader.validate_data(df)
        
        # Should handle gracefully
        assert validation_results['summary']['total_rows'] == 0
        assert validation_results['summary']['total_columns'] == 0
    
    def test_validate_data_types(self):
        """Test that data types are captured"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'value_int': [100, 200, 300],
            'value_float': [10.5, 20.5, 30.5],
            'value_str': ['a', 'b', 'c']
        })
        
        validation_results = loader.validate_data(df)
        
        # Verify data types captured
        assert 'territory_id' in validation_results['data_types']
        assert 'value_int' in validation_results['data_types']
        assert 'value_float' in validation_results['data_types']
        assert 'value_str' in validation_results['data_types']


class TestAnalyzeTemporalStructure:
    """Test temporal structure detection"""
    
    def test_detect_temporal_columns_by_name(self):
        """Test detection of temporal columns by name patterns"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'date': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'year': [2023, 2023, 2023],
            'month': [1, 1, 1],
            'value': [100, 200, 300]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify temporal columns detected
        assert metadata.has_temporal_data is True
        assert 'date' in metadata.temporal_columns
        assert 'year' in metadata.temporal_columns
        assert 'month' in metadata.temporal_columns
        assert 'value' not in metadata.temporal_columns
    
    def test_detect_temporal_columns_by_dtype(self):
        """Test detection of temporal columns by datetime dtype"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'value': [100, 200, 300]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify datetime column detected
        assert metadata.has_temporal_data is True
        assert 'timestamp' in metadata.temporal_columns
    
    def test_no_temporal_structure(self):
        """Test dataset without temporal structure"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'value1': [100, 200, 300],
            'value2': [10.5, 20.5, 30.5]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify no temporal structure detected
        assert metadata.has_temporal_data is False
        assert len(metadata.temporal_columns) == 0
        assert metadata.granularity == 'unknown'
    
    def test_count_periods_per_territory(self):
        """Test counting periods per territory"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1002, 1002, 1003],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', 
                                   '2023-01-01', '2023-02-01', '2023-01-01']),
            'value': [100, 110, 120, 200, 210, 300]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify period counts
        assert metadata.has_temporal_data is True
        assert metadata.periods_per_territory[1001] == 3
        assert metadata.periods_per_territory[1002] == 2
        assert metadata.periods_per_territory[1003] == 1
    
    def test_determine_granularity_monthly(self):
        """Test granularity detection for monthly data"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001],
            'month': [1, 2, 3],
            'year': [2023, 2023, 2023],
            'value': [100, 110, 120]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify monthly granularity detected
        assert metadata.granularity == 'monthly'
    
    def test_determine_granularity_yearly(self):
        """Test granularity detection for yearly data"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001],
            'year': [2021, 2022, 2023],
            'value': [100, 110, 120]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify yearly granularity detected
        assert metadata.granularity == 'yearly'
    
    def test_determine_granularity_quarterly(self):
        """Test granularity detection for quarterly data"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1001],
            'quarter': [1, 2, 3, 4],
            'year': [2023, 2023, 2023, 2023],
            'value': [100, 110, 120, 130]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify quarterly granularity detected
        assert metadata.granularity == 'quarterly'
    
    def test_determine_granularity_from_datetime_diff(self):
        """Test granularity inference from datetime differences"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1001],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']),
            'value': [100, 110, 120, 130]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify monthly granularity inferred from date differences
        assert metadata.granularity == 'monthly'
    
    def test_determine_granularity_daily_from_datetime(self):
        """Test daily granularity inference from datetime differences"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'value': [100, 110, 120]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify daily granularity inferred
        assert metadata.granularity == 'daily'
    
    def test_extract_date_range(self):
        """Test extraction of date range from datetime columns"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'timestamp': pd.to_datetime(['2023-01-01', '2023-06-15', '2023-12-31']),
            'value': [100, 200, 300]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify date range extracted
        assert metadata.date_range[0] == pd.Timestamp('2023-01-01')
        assert metadata.date_range[1] == pd.Timestamp('2023-12-31')
    
    def test_no_temporal_data_with_single_period(self):
        """Test detection with single period per territory (snapshot data)"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01']),
            'value': [100, 200, 300]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Should detect date column (temporal snapshot)
        assert 'date' in metadata.temporal_columns
        # Has temporal columns, so considered temporal (even if snapshot)
        assert metadata.has_temporal_data is True
        # All territories have only 1 period
        assert all(count == 1 for count in metadata.periods_per_territory.values())
    
    def test_infer_granularity_from_period_count(self):
        """Test granularity inference from period count when no datetime columns"""
        loader = DataLoader()
        
        # Create data with 12 periods per territory (likely monthly)
        df = pd.DataFrame({
            'territory_id': [1001] * 12 + [1002] * 12,
            'period': list(range(1, 13)) * 2,
            'value': list(range(100, 112)) + list(range(200, 212))
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Should infer monthly from 12 periods
        assert metadata.granularity == 'monthly'
        assert metadata.periods_per_territory[1001] == 12
        assert metadata.periods_per_territory[1002] == 12
    
    def test_russian_column_names(self):
        """Test detection of temporal columns with Russian names"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1002],
            'год': [2023, 2024, 2023],
            'месяц': [1, 1, 1],
            'value': [100, 110, 200]
        })
        
        metadata = loader.analyze_temporal_structure(df)
        
        # Verify Russian temporal columns detected
        assert metadata.has_temporal_data is True
        assert 'год' in metadata.temporal_columns
        assert 'месяц' in metadata.temporal_columns


class TestDetectDuplicates:
    """Test duplicate detection logic"""
    
    def test_no_duplicates(self):
        """Test detection when there are no duplicates"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1002, 1003],
            'value': [100, 200, 300]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify no duplicates detected
        assert report.duplicate_count == 0
        assert len(report.affected_territories) == 0
        assert report.is_temporal is False
        assert report.recommendation == 'aggregate'
    
    def test_temporal_duplicates_with_date_column(self):
        """Test detection of temporal duplicates with date column"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1002, 1002],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', 
                                   '2023-01-01', '2023-02-01']),
            'value': [100, 110, 120, 200, 210]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify temporal duplicates detected
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert 1002 in report.affected_territories
        assert report.is_temporal is True
        assert report.recommendation == 'enable_temporal_analysis'
    
    def test_temporal_duplicates_with_period_column(self):
        """Test detection of temporal duplicates with period column"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1002, 1002],
            'period': [1, 2, 3, 1, 2],
            'year': [2023, 2023, 2023, 2023, 2023],
            'value': [100, 110, 120, 200, 210]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify temporal duplicates detected
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert 1002 in report.affected_territories
        assert report.is_temporal is True
        assert report.recommendation == 'enable_temporal_analysis'
    
    def test_identical_duplicates_without_temporal_columns(self):
        """Test detection of identical duplicates without temporal structure"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1003],
            'value1': [100, 100, 200, 300],
            'value2': [10.5, 10.5, 20.5, 30.5]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify duplicates detected
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert report.is_temporal is False
        assert report.recommendation == 'aggregate'
    
    def test_different_duplicates_without_temporal_columns(self):
        """Test detection of duplicates with different values (data quality issue)"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1003],
            'value1': [100, 150, 200, 300],
            'value2': [10.5, 15.5, 20.5, 30.5]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify duplicates detected as data quality issue
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert report.is_temporal is False
        assert report.recommendation == 'investigate'
    
    def test_temporal_columns_without_variation(self):
        """Test temporal columns exist but no variation (possible data error)"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1002],
            'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']),
            'value': [100, 110, 200, 210]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify detected as data quality issue (temporal column but no variation)
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert 1002 in report.affected_territories
        assert report.is_temporal is False
        assert report.recommendation == 'investigate'
    
    def test_no_territory_id_column(self):
        """Test detection when territory_id column is missing"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'value1': [100, 200, 300],
            'value2': [10.5, 20.5, 30.5]
        })
        
        report = loader.detect_duplicates(df)
        
        # Should handle gracefully
        assert report.duplicate_count == 0
        assert len(report.affected_territories) == 0
        assert report.recommendation == 'investigate'
    
    def test_mixed_duplicates(self):
        """Test with mix of temporal and non-temporal duplicates"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1001, 1002, 1002, 1003, 1003],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01',
                                   '2023-01-01', '2023-01-01',  # Same date for 1002
                                   '2023-01-01', '2023-02-01']),
            'value': [100, 110, 120, 200, 200, 300, 310]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify duplicates detected
        assert report.duplicate_count > 0
        assert 1001 in report.affected_territories
        assert 1002 in report.affected_territories
        assert 1003 in report.affected_territories
        # Should detect temporal variation from 1001 and 1003
        assert report.is_temporal is True
    
    def test_large_number_of_duplicates(self):
        """Test with large number of duplicate territories"""
        loader = DataLoader()
        
        # Create data with many duplicates
        territory_ids = [i for i in range(1001, 1101) for _ in range(3)]  # 100 territories, 3 periods each
        dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01')] * 100
        values = list(range(300))
        
        df = pd.DataFrame({
            'territory_id': territory_ids,
            'date': dates,
            'value': values
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify all territories detected
        assert report.duplicate_count == 300  # All 300 rows are duplicates
        assert len(report.affected_territories) == 100
        assert report.is_temporal is True
        assert report.recommendation == 'enable_temporal_analysis'
    
    def test_single_duplicate_pair(self):
        """Test with just one territory having duplicates"""
        loader = DataLoader()
        
        df = pd.DataFrame({
            'territory_id': [1001, 1001, 1002, 1003],
            'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-01', '2023-01-01']),
            'value': [100, 110, 200, 300]
        })
        
        report = loader.detect_duplicates(df)
        
        # Verify single duplicate detected
        assert report.duplicate_count == 2  # Two rows for territory 1001
        assert len(report.affected_territories) == 1
        assert 1001 in report.affected_territories
        assert report.is_temporal is True
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        loader = DataLoader()
        
        df = pd.DataFrame()
        
        report = loader.detect_duplicates(df)
        
        # Should handle gracefully
        assert report.duplicate_count == 0
        assert len(report.affected_territories) == 0


class TestIntegration:
    """Integration tests for full data loading pipeline"""
    
    def test_full_pipeline(self, temp_test_dir, sample_sberindex_data, 
                          sample_rosstat_data, sample_municipal_dict):
        """Test complete data loading and merging pipeline"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        loader = DataLoader(str(temp_test_dir))
        
        # Load all data
        sberindex = loader.load_sberindex_data()
        rosstat = loader.load_rosstat_data()
        municipal = loader.load_municipal_dict()
        
        # Merge datasets
        merged_df = loader.merge_datasets(sberindex, rosstat, municipal)
        
        # Validate data
        validation_results = loader.validate_data(merged_df)
        
        # Verify pipeline completed successfully
        assert merged_df is not None
        assert len(merged_df) == 3
        assert validation_results is not None
        assert 'summary' in validation_results
    
    def test_pipeline_with_partial_data(self, temp_test_dir, sample_sberindex_data,
                                       sample_rosstat_data, sample_municipal_dict):
        """Test pipeline with some missing files"""
        create_test_files(temp_test_dir, sample_sberindex_data, sample_rosstat_data, sample_municipal_dict)
        
        # Remove some files
        os.remove(temp_test_dir / 'connection.parquet')
        os.remove(temp_test_dir / 'rosstat' / '2_bdmo_population.parquet')
        
        loader = DataLoader(str(temp_test_dir))
        
        # Load all data
        sberindex = loader.load_sberindex_data()
        rosstat = loader.load_rosstat_data()
        municipal = loader.load_municipal_dict()
        
        # Verify some data is None
        assert sberindex['connection'] is None
        assert rosstat['population'] is None
        
        # Merge should still work
        merged_df = loader.merge_datasets(sberindex, rosstat, municipal)
        
        # Validate data
        validation_results = loader.validate_data(merged_df)
        
        # Verify pipeline completed
        assert merged_df is not None
        assert validation_results is not None
