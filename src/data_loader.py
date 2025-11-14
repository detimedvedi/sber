"""
Data loading module for СберИндекс anomaly detection system.
Handles loading and preprocessing of all data sources.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.error_handler import get_error_handler

logger = logging.getLogger(__name__)


@dataclass
class TemporalMetadata:
    """
    Metadata about temporal structure in the dataset.
    
    Attributes:
        has_temporal_data: Whether the dataset contains temporal dimensions
        temporal_columns: List of column names that represent temporal data (date, period, year, month, etc.)
        granularity: Temporal granularity ('daily', 'monthly', 'quarterly', 'yearly', 'unknown')
        periods_per_territory: Dictionary mapping territory_id to number of time periods
        date_range: Tuple of (start_date, end_date) for the temporal data range
    """
    has_temporal_data: bool
    temporal_columns: List[str]
    granularity: str
    periods_per_territory: Dict[int, int]
    date_range: Tuple[Optional[datetime], Optional[datetime]]


@dataclass
class DuplicateReport:
    """
    Report on duplicate territory_ids in the dataset.
    
    Attributes:
        duplicate_count: Number of duplicate territory_id entries
        affected_territories: List of territory_ids that have duplicates
        is_temporal: Whether duplicates represent temporal data (multiple time periods)
        recommendation: Recommended action ('aggregate', 'enable_temporal_analysis', 'investigate')
    """
    duplicate_count: int
    affected_territories: List[int]
    is_temporal: bool
    recommendation: str


class DataLoadError(Exception):
    """Exception raised for errors during data loading."""
    pass


class DataValidationError(Exception):
    """Exception raised for errors during data validation."""
    pass


class DataLoader:
    """Loads and processes data from СберИндекс, Росстат, and municipal dictionary."""
    
    def __init__(self, base_path: str = "."):
        """
        Initialize DataLoader with base path.
        
        Args:
            base_path: Base directory path where data files are located
        """
        self.base_path = Path(base_path)
        self.error_handler = get_error_handler(workspace_root=self.base_path)
        logger.info(f"DataLoader initialized with base path: {self.base_path}")
    
    def _handle_load_error(self, exception: Exception, file_path: Path, data_key: str, operation: str = 'load'):
        """
        Handle data loading errors with enhanced context.
        
        Args:
            exception: The exception that occurred
            file_path: Path to the file being loaded
            data_key: Key identifying the data type
            operation: Operation being performed
        """
        self.error_handler.handle_error(
            exception=exception,
            component_name=f"DataLoader.{operation}",
            config=None,
            additional_context={
                'file_path': str(file_path),
                'data_key': data_key,
                'operation': operation,
                'file_exists': file_path.exists(),
                'file_size_bytes': file_path.stat().st_size if file_path.exists() else 0
            },
            log_level=logging.ERROR
        )
    
    def load_sberindex_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load СберИндекс data from parquet files.
        
        Returns:
            Dictionary with keys 'connection', 'consumption', 'market_access'
            containing respective DataFrames. Missing files result in None values.
        """
        logger.info("Loading СберИндекс data...")
        
        sberindex_files = {
            'connection': 'connection.parquet',
            'consumption': 'consumption.parquet',
            'market_access': 'market_access.parquet'
        }
        
        data = {}
        
        for key, filename in sberindex_files.items():
            file_path = self.base_path / filename
            try:
                df = pd.read_parquet(file_path)
                data[key] = df
                logger.info(f"Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}. Continuing with available data.")
                data[key] = None
            except Exception as e:
                self._handle_load_error(e, file_path, key, 'load_sberindex_data')
                data[key] = None
        
        return data
    
    def load_connection_data(self) -> pd.DataFrame:
        """
        Load connection graph data (4.7M connections between territories).
        
        Returns:
            DataFrame with columns:
            - territory_id_x: Source territory
            - territory_id_y: Target territory
            - distance: Distance in km
            - type: Connection type (highway, etc.)
            
            Returns empty DataFrame if file not found or invalid.
        """
        logger.info("Loading connection graph...")
        file_path = self.base_path / 'connection.parquet'
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded connection graph: {df.shape[0]:,} connections, {df.shape[1]} columns")
            
            # Validate structure
            required_cols = ['territory_id_x', 'territory_id_y', 'distance']
            missing = set(required_cols) - set(df.columns)
            if missing:
                logger.warning(f"Missing required columns in connection data: {missing}")
                return pd.DataFrame()
            
            # Log statistics
            unique_territories = set(df['territory_id_x'].unique()) | set(df['territory_id_y'].unique())
            logger.info(f"Connection graph covers {len(unique_territories)} territories")
            
            if 'type' in df.columns:
                logger.info(f"Connection types: {df['type'].value_counts().to_dict()}")
            
            return df
            
        except FileNotFoundError:
            logger.warning(f"Connection file not found: {file_path}")
            return pd.DataFrame()
        except Exception as e:
            self._handle_load_error(e, file_path, 'connection', 'load_connection_data')
            return pd.DataFrame()

    def load_rosstat_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Росстат data from parquet files.
        
        Returns:
            Dictionary with keys 'population', 'migration', 'salary'
            containing respective DataFrames. Missing files result in None values.
        """
        logger.info("Loading Росстат data...")
        
        rosstat_files = {
            'population': 'rosstat/2_bdmo_population.parquet',
            'migration': 'rosstat/3_bdmo_migration.parquet',
            'salary': 'rosstat/4_bdmo_salary.parquet'
        }
        
        data = {}
        
        for key, filename in rosstat_files.items():
            file_path = self.base_path / filename
            try:
                df = pd.read_parquet(file_path)
                data[key] = df
                logger.info(f"Loaded {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}. Continuing with available data.")
                data[key] = None
            except Exception as e:
                self._handle_load_error(e, file_path, key, 'load_rosstat_data')
                data[key] = None
        
        return data
    
    def load_municipal_dict(self) -> Optional[pd.DataFrame]:
        """
        Load municipal dictionary from Excel file.
        
        Returns:
            DataFrame with municipal district information, or None if file not found.
        """
        logger.info("Loading municipal dictionary...")
        
        file_path = self.base_path / 't_dict_municipal' / 't_dict_municipal_districts.xlsx'
        
        try:
            df = pd.read_excel(file_path)
            logger.info(f"Loaded municipal dictionary: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}. Continuing without municipal dictionary.")
            return None
        except Exception as e:
            self._handle_load_error(e, file_path, 'municipal_dict', 'load_municipal_dict')
            return None

    def merge_datasets(
        self, 
        sberindex: Dict[str, pd.DataFrame], 
        rosstat: Dict[str, pd.DataFrame],
        municipal: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge all datasets by territory_id.
        
        Args:
            sberindex: Dictionary with СберИндекс data
            rosstat: Dictionary with Росстат data
            municipal: Municipal dictionary DataFrame
        
        Returns:
            Unified DataFrame with all data merged by territory_id
        """
        logger.info("Merging datasets...")
        
        # Start with municipal dictionary if available
        if municipal is not None:
            unified_df = municipal[['territory_id', 'municipal_district_name_short', 
                                   'region_name', 'oktmo']].copy()
            logger.info(f"Starting with {len(unified_df)} municipalities from dictionary")
        else:
            # If no municipal dict, we'll build from available data
            unified_df = None
            logger.warning("No municipal dictionary available, will build from data sources")
        
        # Process consumption data (pivot categories to columns)
        # FIXED: Preserve temporal structure (date column) instead of aggregating
        if sberindex.get('consumption') is not None:
            consumption_df = sberindex['consumption']
            
            # Check if temporal data exists
            has_date = 'date' in consumption_df.columns
            
            if has_date:
                # Preserve ALL periods for temporal analysis
                logger.info(f"Preserving temporal structure: {consumption_df['date'].nunique()} periods")
                
                consumption_pivot = consumption_df.pivot(
                    index=['territory_id', 'date'],  # ✅ PRESERVE DATE
                    columns='category',
                    values='value'
                ).reset_index()
                
                # Rename columns with prefix
                consumption_pivot.columns = ['territory_id', 'date'] + [
                    f'consumption_{col}' for col in consumption_pivot.columns[2:]
                ]
                
                logger.info(f"Consumption data shape after pivot: {consumption_pivot.shape[0]} rows, {consumption_pivot.shape[1]} columns")
            else:
                # No temporal data - use simple pivot with aggregation
                consumption_pivot = consumption_df.pivot_table(
                    index='territory_id',
                    columns='category',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                consumption_pivot.columns = ['territory_id'] + [
                    f'consumption_{col}' for col in consumption_pivot.columns[1:]
                ]
            
            if unified_df is not None:
                # When merging with temporal data, we need to handle the case where
                # unified_df (from municipal dict) doesn't have a date column
                if has_date and 'date' not in unified_df.columns:
                    # Cross join: each territory in municipal dict gets all date periods
                    logger.info("Cross-joining municipal data with temporal consumption data")
                    unified_df = unified_df.merge(consumption_pivot, on='territory_id', how='right')
                else:
                    # Standard merge
                    merge_cols = ['territory_id', 'date'] if has_date else ['territory_id']
                    unified_df = unified_df.merge(consumption_pivot, on=merge_cols, how='left')
            else:
                unified_df = consumption_pivot
            
            logger.info(f"Merged consumption data: {len(consumption_pivot.columns)-2 if has_date else len(consumption_pivot.columns)-1} categories")
        
        # Process market access data
        if sberindex.get('market_access') is not None:
            market_access_df = sberindex['market_access']
            
            if unified_df is not None:
                unified_df = unified_df.merge(market_access_df, on='territory_id', how='left')
            else:
                unified_df = market_access_df
            
            logger.info("Merged market access data")
        
        # Process population data (aggregate by territory_id)
        if rosstat.get('population') is not None:
            population_df = rosstat['population']
            # Aggregate population by territory_id
            pop_agg = population_df.groupby('territory_id')['value'].sum().reset_index()
            pop_agg.columns = ['territory_id', 'population_total']
            
            # Also get gender breakdown
            pop_gender = population_df.groupby(['territory_id', 'gender'])['value'].sum().unstack(fill_value=0).reset_index()
            pop_gender.columns = ['territory_id', 'population_female', 'population_male']
            
            pop_merged = pop_agg.merge(pop_gender, on='territory_id', how='left')
            
            if unified_df is not None:
                unified_df = unified_df.merge(pop_merged, on='territory_id', how='left')
            else:
                unified_df = pop_merged
            
            logger.info("Merged population data")
        
        # Process migration data (aggregate by territory_id)
        if rosstat.get('migration') is not None:
            migration_df = rosstat['migration']
            # Aggregate migration by territory_id
            migration_agg = migration_df.groupby('territory_id')['value'].sum().reset_index()
            migration_agg.columns = ['territory_id', 'migration_total']
            
            if unified_df is not None:
                unified_df = unified_df.merge(migration_agg, on='territory_id', how='left')
            else:
                unified_df = migration_agg
            
            logger.info("Merged migration data")
        
        # Process salary data (pivot industries to columns)
        if rosstat.get('salary') is not None:
            salary_df = rosstat['salary']
            # Pivot industries to columns and aggregate by territory_id
            salary_pivot = salary_df.pivot_table(
                index='territory_id',
                columns='okved_name',
                values='value',
                aggfunc='mean'
            ).reset_index()
            # Rename columns with prefix
            salary_pivot.columns = ['territory_id'] + [
                f'salary_{col}' for col in salary_pivot.columns[1:]
            ]
            
            if unified_df is not None:
                unified_df = unified_df.merge(salary_pivot, on='territory_id', how='left')
            else:
                unified_df = salary_pivot
            
            logger.info(f"Merged salary data: {len(salary_pivot.columns)-1} industries")
        
        if unified_df is None:
            raise DataLoadError("No data available to merge")
        
        logger.info(f"Final merged dataset: {unified_df.shape[0]} rows, {unified_df.shape[1]} columns")
        
        return unified_df

    def analyze_temporal_structure(self, df: pd.DataFrame) -> TemporalMetadata:
        """
        Analyze temporal structure in the dataset.
        
        Detects temporal columns, determines granularity, and counts periods per territory.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            TemporalMetadata object with temporal structure information
        """
        logger.info("Analyzing temporal structure...")
        
        # Define temporal column patterns to search for
        temporal_patterns = {
            'date': ['date', 'datetime', 'timestamp', 'dt'],
            'period': ['period', 'period_id', 'period_name'],
            'year': ['year', 'yr', 'год'],
            'month': ['month', 'месяц'],
            'quarter': ['quarter', 'квартал', 'qtr'],
            'day': ['day', 'день'],
            'week': ['week', 'неделя']
        }
        
        # Find temporal columns
        temporal_columns = []
        for col in df.columns:
            col_lower = col.lower()
            for pattern_type, patterns in temporal_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    temporal_columns.append(col)
                    logger.debug(f"Found temporal column: {col} (type: {pattern_type})")
                    break
        
        # Also check for datetime dtypes
        for col in df.columns:
            if col not in temporal_columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    temporal_columns.append(col)
                    logger.debug(f"Found datetime column by dtype: {col}")
        
        # Check if we have temporal data
        has_temporal_data = len(temporal_columns) > 0
        
        # Count periods per territory
        periods_per_territory = {}
        territories_with_multiple_periods = 0
        if 'territory_id' in df.columns:
            periods_per_territory = df.groupby('territory_id').size().to_dict()
            
            # Log statistics
            period_counts = list(periods_per_territory.values())
            if period_counts:
                max_periods = max(period_counts)
                min_periods = min(period_counts)
                avg_periods = sum(period_counts) / len(period_counts)
                
                logger.info(f"Periods per territory - Min: {min_periods}, Max: {max_periods}, Avg: {avg_periods:.1f}")
                
                # Check if we have multiple periods (indicating temporal data)
                territories_with_multiple_periods = sum(1 for count in period_counts if count > 1)
                if territories_with_multiple_periods > 0:
                    logger.info(f"Found {territories_with_multiple_periods} territories with multiple time periods")
                else:
                    logger.info("No territories with multiple time periods detected")
                    # If we have temporal columns but no multiple periods, still consider it temporal
                    # (could be a snapshot at one point in time)
        
        # Determine granularity
        granularity = 'unknown'
        date_range = (None, None)
        
        if has_temporal_data and temporal_columns:
            # Try to determine granularity from column names
            col_names_lower = [col.lower() for col in temporal_columns]
            
            if any('day' in col or 'день' in col or 'daily' in col for col in col_names_lower):
                granularity = 'daily'
            elif any('month' in col or 'месяц' in col or 'monthly' in col for col in col_names_lower):
                granularity = 'monthly'
            elif any('quarter' in col or 'квартал' in col or 'quarterly' in col for col in col_names_lower):
                granularity = 'quarterly'
            elif any('year' in col or 'год' in col or 'yearly' in col or 'annual' in col for col in col_names_lower):
                granularity = 'yearly'
            elif any('week' in col or 'неделя' in col or 'weekly' in col for col in col_names_lower):
                granularity = 'weekly'
            
            # Try to extract date range from datetime columns
            for col in temporal_columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        min_date = df[col].min()
                        max_date = df[col].max()
                        if pd.notna(min_date) and pd.notna(max_date):
                            date_range = (min_date, max_date)
                            logger.info(f"Date range: {min_date} to {max_date}")
                            
                            # Infer granularity from date range if not already determined
                            if granularity == 'unknown':
                                # Calculate average time difference
                                date_series = df[col].dropna().sort_values()
                                if len(date_series) > 1:
                                    time_diffs = date_series.diff().dropna()
                                    if len(time_diffs) > 0:
                                        avg_diff = time_diffs.mean()
                                        
                                        if avg_diff <= pd.Timedelta(days=1):
                                            granularity = 'daily'
                                        elif avg_diff <= pd.Timedelta(days=7):
                                            granularity = 'weekly'
                                        elif avg_diff <= pd.Timedelta(days=31):
                                            granularity = 'monthly'
                                        elif avg_diff <= pd.Timedelta(days=92):
                                            granularity = 'quarterly'
                                        else:
                                            granularity = 'yearly'
                                        
                                        logger.info(f"Inferred granularity from date differences: {granularity}")
                            break
                except Exception as e:
                    logger.debug(f"Could not extract date range from column {col}: {str(e)}")
            
            # If still unknown, try to infer from period counts
            if granularity == 'unknown' and periods_per_territory and territories_with_multiple_periods > 0:
                max_periods = max(periods_per_territory.values())
                if max_periods >= 365:
                    granularity = 'daily'
                elif max_periods >= 52:
                    granularity = 'weekly'
                elif max_periods >= 12:
                    granularity = 'monthly'
                elif max_periods >= 4:
                    granularity = 'quarterly'
                else:
                    granularity = 'yearly'
                
                logger.info(f"Inferred granularity from period count ({max_periods}): {granularity}")
        
        # Create metadata object
        metadata = TemporalMetadata(
            has_temporal_data=has_temporal_data,
            temporal_columns=temporal_columns,
            granularity=granularity,
            periods_per_territory=periods_per_territory,
            date_range=date_range
        )
        
        # Log summary
        if has_temporal_data:
            logger.info(f"Temporal structure detected: {len(temporal_columns)} temporal columns, "
                       f"granularity: {granularity}")
        else:
            logger.info("No temporal structure detected in dataset")
        
        return metadata

    def detect_duplicates(self, df: pd.DataFrame) -> DuplicateReport:
        """
        Detect and analyze duplicate territory_ids in the dataset.
        
        Distinguishes between temporal duplicates (multiple time periods for same territory)
        and data quality issues (true duplicates that should be investigated).
        
        Args:
            df: DataFrame to analyze for duplicates
        
        Returns:
            DuplicateReport with analysis results and recommendations
        """
        logger.info("Detecting duplicate territory_ids...")
        
        # Check if territory_id column exists
        if 'territory_id' not in df.columns:
            logger.warning("No territory_id column found, cannot detect duplicates")
            return DuplicateReport(
                duplicate_count=0,
                affected_territories=[],
                is_temporal=False,
                recommendation='investigate'
            )
        
        # Find duplicate territory_ids
        duplicate_mask = df.duplicated(subset=['territory_id'], keep=False)
        duplicate_count = duplicate_mask.sum()
        affected_territories = df[duplicate_mask]['territory_id'].unique().tolist()
        
        if duplicate_count == 0:
            logger.info("No duplicate territory_ids found")
            return DuplicateReport(
                duplicate_count=0,
                affected_territories=[],
                is_temporal=False,
                recommendation='aggregate'
            )
        
        # Log detailed duplicate statistics
        logger.warning(
            f"Found {duplicate_count} duplicate territory_id entries affecting {len(affected_territories)} territories",
            extra={
                'duplicate_count': duplicate_count,
                'affected_territories_count': len(affected_territories),
                'affected_territories_sample': affected_territories[:10],
                'data_quality_issue': 'duplicates'
            }
        )
        
        # Analyze temporal structure to determine if duplicates are temporal
        temporal_metadata = self.analyze_temporal_structure(df)
        
        # Determine if duplicates are temporal or data errors
        is_temporal = False
        recommendation = 'investigate'
        
        if temporal_metadata.has_temporal_data and len(temporal_metadata.temporal_columns) > 0:
            # We have temporal columns, likely temporal duplicates
            is_temporal = True
            
            # Check if duplicates have different values in temporal columns
            temporal_variation_detected = False
            
            for territory_id in affected_territories[:10]:  # Sample first 10 for performance
                territory_records = df[df['territory_id'] == territory_id]
                
                # Check if temporal columns have different values
                for temp_col in temporal_metadata.temporal_columns:
                    if temp_col in territory_records.columns:
                        unique_values = territory_records[temp_col].nunique()
                        if unique_values > 1:
                            temporal_variation_detected = True
                            logger.debug(f"Territory {territory_id} has {unique_values} different values in temporal column {temp_col}")
                            break
                
                if temporal_variation_detected:
                    break
            
            if temporal_variation_detected:
                logger.info("Duplicates appear to be temporal data (different time periods)")
                recommendation = 'enable_temporal_analysis'
            else:
                # Temporal columns exist but no variation - might be true duplicates
                logger.warning("Temporal columns exist but no variation detected - possible data quality issue")
                is_temporal = False
                recommendation = 'investigate'
        else:
            # No temporal columns, check if duplicates are identical or different
            identical_duplicates = 0
            different_duplicates = 0
            
            for territory_id in affected_territories[:20]:  # Sample for performance
                territory_records = df[df['territory_id'] == territory_id]
                
                if len(territory_records) > 1:
                    # Compare all columns except territory_id
                    compare_cols = [col for col in df.columns if col != 'territory_id']
                    
                    # Check if all rows are identical
                    first_row = territory_records.iloc[0][compare_cols]
                    all_identical = True
                    
                    for idx in range(1, len(territory_records)):
                        if not territory_records.iloc[idx][compare_cols].equals(first_row):
                            all_identical = False
                            break
                    
                    if all_identical:
                        identical_duplicates += 1
                    else:
                        different_duplicates += 1
            
            logger.info(f"Duplicate analysis (sample): {identical_duplicates} identical, {different_duplicates} different")
            
            if identical_duplicates > different_duplicates:
                # Most duplicates are identical - safe to aggregate
                logger.info("Most duplicates are identical - recommend aggregation")
                recommendation = 'aggregate'
            elif different_duplicates > 0:
                # Duplicates have different values but no temporal structure
                logger.warning("Duplicates have different values without temporal structure - investigate data quality")
                recommendation = 'investigate'
            else:
                # Default to aggregation
                recommendation = 'aggregate'
        
        # Create report
        report = DuplicateReport(
            duplicate_count=duplicate_count,
            affected_territories=affected_territories,
            is_temporal=is_temporal,
            recommendation=recommendation
        )
        
        # Log recommendation
        logger.info(f"Duplicate detection complete: is_temporal={is_temporal}, recommendation='{recommendation}'")
        
        return report

    def aggregate_temporal_data(
        self, 
        df: pd.DataFrame, 
        method: str = 'latest',
        temporal_metadata: Optional[TemporalMetadata] = None
    ) -> pd.DataFrame:
        """
        Aggregate temporal data to single record per territory.
        
        When multiple time periods exist for the same territory_id, this method
        aggregates them into a single record using the specified method.
        
        Args:
            df: DataFrame with temporal data (may contain duplicate territory_ids)
            method: Aggregation method - 'latest', 'mean', or 'median'
            temporal_metadata: Optional TemporalMetadata object. If not provided,
                              will be computed automatically.
        
        Returns:
            DataFrame with one record per territory_id
        
        Raises:
            ValueError: If method is not one of 'latest', 'mean', 'median'
        """
        logger.info(f"Aggregating temporal data using method: {method}")
        
        # Validate method
        valid_methods = ['latest', 'mean', 'median']
        if method not in valid_methods:
            raise ValueError(f"Invalid aggregation method '{method}'. Must be one of {valid_methods}")
        
        # Check if territory_id column exists
        if 'territory_id' not in df.columns:
            logger.warning("No territory_id column found, returning dataframe unchanged")
            return df
        
        # Check if we have duplicates
        duplicate_count = df.duplicated(subset=['territory_id'], keep=False).sum()
        if duplicate_count == 0:
            logger.info("No duplicate territory_ids found, no aggregation needed")
            return df
        
        logger.info(f"Found {duplicate_count} duplicate records to aggregate")
        
        # Analyze temporal structure if not provided
        if temporal_metadata is None:
            temporal_metadata = self.analyze_temporal_structure(df)
        
        # Identify columns to aggregate
        # Separate columns into categories
        id_columns = ['territory_id']
        temporal_columns = temporal_metadata.temporal_columns if temporal_metadata else []
        
        # Identify categorical/text columns (should use 'first' aggregation)
        categorical_columns = []
        numeric_columns = []
        
        for col in df.columns:
            if col in id_columns or col in temporal_columns:
                continue
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            else:
                categorical_columns.append(col)
        
        logger.debug(f"Column classification: {len(numeric_columns)} numeric, "
                    f"{len(categorical_columns)} categorical, "
                    f"{len(temporal_columns)} temporal")
        
        # Prepare aggregation dictionary
        agg_dict = {}
        
        # For categorical columns, always use 'first'
        for col in categorical_columns:
            agg_dict[col] = 'first'
        
        # For numeric columns, use specified method
        if method == 'latest':
            # For 'latest', we need to sort by temporal column first
            if temporal_columns:
                # Find the best temporal column to sort by
                sort_col = None
                for temp_col in temporal_columns:
                    if temp_col in df.columns:
                        # Prefer datetime columns
                        if pd.api.types.is_datetime64_any_dtype(df[temp_col]):
                            sort_col = temp_col
                            break
                        # Otherwise use first available temporal column
                        if sort_col is None:
                            sort_col = temp_col
                
                if sort_col:
                    logger.info(f"Sorting by temporal column '{sort_col}' to get latest records")
                    # Sort by territory_id and temporal column (descending to get latest first)
                    df_sorted = df.sort_values(['territory_id', sort_col], ascending=[True, False])
                    
                    # Use 'first' aggregation after sorting (which gives us the latest)
                    for col in numeric_columns:
                        agg_dict[col] = 'first'
                    
                    # Also keep the temporal column value
                    for temp_col in temporal_columns:
                        if temp_col in df.columns:
                            agg_dict[temp_col] = 'first'
                    
                    # Group and aggregate
                    aggregated_df = df_sorted.groupby('territory_id', as_index=False).agg(agg_dict)
                    
                    logger.info(f"Aggregated {len(df)} rows to {len(aggregated_df)} rows using 'latest' method")
                    return aggregated_df
                else:
                    logger.warning("No suitable temporal column found for 'latest' method, falling back to 'first'")
                    # Fall back to using 'first' without sorting
                    for col in numeric_columns:
                        agg_dict[col] = 'first'
            else:
                logger.warning("No temporal columns detected for 'latest' method, using 'first' record")
                # No temporal columns, just use first record
                for col in numeric_columns:
                    agg_dict[col] = 'first'
        
        elif method == 'mean':
            # Use mean for numeric columns
            for col in numeric_columns:
                agg_dict[col] = 'mean'
            
            # For temporal columns, use 'first' (can't average dates)
            for temp_col in temporal_columns:
                if temp_col in df.columns:
                    agg_dict[temp_col] = 'first'
        
        elif method == 'median':
            # Use median for numeric columns
            for col in numeric_columns:
                agg_dict[col] = 'median'
            
            # For temporal columns, use 'first' (can't take median of dates)
            for temp_col in temporal_columns:
                if temp_col in df.columns:
                    agg_dict[temp_col] = 'first'
        
        # Perform aggregation
        try:
            aggregated_df = df.groupby('territory_id', as_index=False).agg(agg_dict)
            
            logger.info(f"Aggregated {len(df)} rows to {len(aggregated_df)} rows using '{method}' method")
            
            # Log some statistics about the aggregation
            original_territories = df['territory_id'].nunique()
            aggregated_territories = len(aggregated_df)
            
            if original_territories != aggregated_territories:
                logger.warning(f"Territory count mismatch: {original_territories} unique territories "
                             f"in original data, {aggregated_territories} in aggregated data")
            
            # Calculate how many records were aggregated per territory
            records_per_territory = df.groupby('territory_id').size()
            avg_records = records_per_territory.mean()
            max_records = records_per_territory.max()
            
            logger.info(f"Aggregation statistics: avg {avg_records:.1f} records per territory, "
                       f"max {max_records} records for a single territory")
            
            return aggregated_df
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}", exc_info=True)
            logger.warning("Returning original dataframe due to aggregation error")
            return df

    def create_source_mapping(self, columns: List[str]) -> Dict[str, str]:
        """
        Create explicit mapping between column names and data sources.
        
        Uses column prefixes as the primary method to determine data source,
        with fallback heuristics for ambiguous column names.
        
        Args:
            columns: List of column names to map
        
        Returns:
            Dictionary mapping column names to data sources ('sberindex', 'rosstat', 'municipal_dict', 'unknown')
        """
        logger.info(f"Creating source mapping for {len(columns)} columns...")
        
        source_mapping = {}
        ambiguous_columns = []
        
        # Define prefix-based mapping rules (primary method)
        prefix_rules = {
            'consumption_': 'sberindex',
            'connection_': 'sberindex',
            'market_access': 'sberindex',
            'salary_': 'rosstat',
            'population_': 'rosstat',
            'population': 'rosstat',  # Handle both with and without underscore
            'migration_': 'rosstat',
            'migration': 'rosstat',
        }
        
        # Define keyword-based fallback heuristics
        sberindex_keywords = ['consumption', 'connection', 'market', 'access', 'spending', 'purchase']
        rosstat_keywords = ['salary', 'population', 'migration', 'wage', 'income', 'demographic']
        municipal_keywords = ['territory_id', 'municipal', 'region', 'oktmo', 'district', 'name']
        
        for col in columns:
            col_lower = col.lower()
            source = None
            
            # Method 1: Check prefix rules (primary method)
            for prefix, src in prefix_rules.items():
                if col_lower.startswith(prefix):
                    source = src
                    logger.debug(f"Column '{col}' mapped to '{src}' by prefix '{prefix}'")
                    break
            
            # Method 2: Check for municipal dictionary columns
            if source is None:
                for keyword in municipal_keywords:
                    if keyword in col_lower:
                        source = 'municipal_dict'
                        logger.debug(f"Column '{col}' mapped to 'municipal_dict' by keyword '{keyword}'")
                        break
            
            # Method 3: Fallback heuristics using keywords
            if source is None:
                # Check СберИндекс keywords
                sberindex_matches = sum(1 for keyword in sberindex_keywords if keyword in col_lower)
                rosstat_matches = sum(1 for keyword in rosstat_keywords if keyword in col_lower)
                
                if sberindex_matches > rosstat_matches and sberindex_matches > 0:
                    source = 'sberindex'
                    logger.debug(f"Column '{col}' mapped to 'sberindex' by keyword heuristics ({sberindex_matches} matches)")
                elif rosstat_matches > sberindex_matches and rosstat_matches > 0:
                    source = 'rosstat'
                    logger.debug(f"Column '{col}' mapped to 'rosstat' by keyword heuristics ({rosstat_matches} matches)")
                elif sberindex_matches == rosstat_matches and sberindex_matches > 0:
                    # Ambiguous case - both sources have equal keyword matches
                    source = 'unknown'
                    ambiguous_columns.append(col)
                    logger.warning(f"Column '{col}' is ambiguous (equal matches: {sberindex_matches})")
            
            # Method 4: Default to unknown if no match found
            if source is None:
                source = 'unknown'
                logger.debug(f"Column '{col}' could not be mapped, marked as 'unknown'")
            
            source_mapping[col] = source
        
        # Log summary statistics
        source_counts = {}
        for source in source_mapping.values():
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info(f"Source mapping complete: {source_counts}")
        
        if ambiguous_columns:
            logger.warning(f"Found {len(ambiguous_columns)} ambiguous columns: {ambiguous_columns[:5]}"
                          f"{'...' if len(ambiguous_columns) > 5 else ''}")
        
        return source_mapping

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and completeness.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with validation results including:
            - missing_values: Count of missing values per column
            - duplicate_ids: List of duplicate territory_ids
            - data_types: Data types of each column
            - completeness_scores: Data completeness score per municipality
        """
        logger.info("Validating data quality...")
        
        validation_results = {}
        
        # Check for missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}
        validation_results['missing_values'] = missing_values
        
        if missing_values:
            # Calculate missing value statistics
            total_missing = sum(missing_values.values())
            total_cells = len(df) * len(df.columns)
            missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
            
            # Find columns with high missing percentages
            high_missing_cols = {
                col: count for col, count in missing_values.items()
                if (count / len(df) * 100) > 50
            }
            
            logger.warning(
                f"Found missing values in {len(missing_values)} columns",
                extra={
                    'columns_with_missing': len(missing_values),
                    'total_missing_values': total_missing,
                    'missing_percentage': round(missing_percentage, 2),
                    'high_missing_columns_count': len(high_missing_cols),
                    'data_quality_issue': 'missing_values'
                }
            )
            
            # Log top columns with missing values
            sorted_missing = sorted(missing_values.items(), key=lambda x: x[1], reverse=True)
            for col, count in sorted_missing[:10]:  # Log top 10
                missing_pct = (count / len(df) * 100)
                logger.warning(
                    f"  {col}: {count} missing values ({missing_pct:.1f}%)",
                    extra={
                        'column': col,
                        'missing_count': count,
                        'missing_percentage': round(missing_pct, 2),
                        'data_quality_issue': 'missing_values'
                    }
                )
            
            # Warn about columns with >50% missing
            if high_missing_cols:
                logger.warning(
                    f"Found {len(high_missing_cols)} columns with >50% missing values - consider excluding from analysis",
                    extra={
                        'high_missing_columns': list(high_missing_cols.keys())[:5],
                        'data_quality_issue': 'high_missing_values'
                    }
                )
        else:
            logger.info("No missing values found")
        
        # Check for duplicate territory_ids
        if 'territory_id' in df.columns:
            duplicate_ids = df[df.duplicated(subset=['territory_id'], keep=False)]['territory_id'].unique().tolist()
            validation_results['duplicate_ids'] = duplicate_ids
            
            if duplicate_ids:
                logger.warning(f"Found {len(duplicate_ids)} duplicate territory_ids")
            else:
                logger.info("No duplicate territory_ids found")
        else:
            validation_results['duplicate_ids'] = []
            logger.warning("No territory_id column found for duplicate check")
        
        # Get data types
        data_types = df.dtypes.astype(str).to_dict()
        validation_results['data_types'] = data_types
        
        # Calculate data completeness score for each municipality
        if 'territory_id' in df.columns:
            # Calculate completeness as percentage of non-null values per row
            completeness = df.set_index('territory_id').notna().mean(axis=1)
            validation_results['completeness_scores'] = completeness.to_dict()
            
            avg_completeness = completeness.mean()
            logger.info(f"Average data completeness: {avg_completeness:.2%}")
            
            # Log municipalities with low completeness
            low_completeness = completeness[completeness < 0.5]
            if len(low_completeness) > 0:
                logger.warning(f"Found {len(low_completeness)} municipalities with <50% data completeness")
        else:
            validation_results['completeness_scores'] = {}
        
        # Summary statistics
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        total_missing = df.isnull().sum().sum()
        overall_completeness = 1 - (total_missing / total_cells) if total_cells > 0 else 0
        
        validation_results['summary'] = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'total_cells': total_cells,
            'total_missing': int(total_missing),
            'overall_completeness': overall_completeness
        }
        
        logger.info(f"Validation complete: {total_rows} rows, {total_cols} columns, "
                   f"{overall_completeness:.2%} overall completeness")
        
        return validation_results
