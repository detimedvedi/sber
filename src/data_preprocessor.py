"""
Data Preprocessor Module for СберИндекс Anomaly Detection System

This module provides preprocessing functionality including:
- Municipality classification (capital/urban/rural)
- Robust statistics calculation (median, MAD, IQR)
- Data normalization and transformation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class RobustStats:
    """
    Container for robust statistical measures.
    
    Attributes:
        median: Median value (robust central tendency)
        mad: Median Absolute Deviation
        iqr: Interquartile Range (Q3 - Q1)
        percentiles: Dictionary of percentile values (1st, 5th, 25th, 75th, 95th, 99th)
        skewness: Distribution skewness measure
        count: Number of non-null values
    """
    median: float
    mad: float
    iqr: float
    percentiles: Dict[str, float]
    skewness: float
    count: int


@dataclass
class MissingnessReport:
    """
    Container for missingness analysis results.
    
    Attributes:
        missing_pct_per_indicator: Dictionary mapping indicator names to missing percentage (0-100)
        missing_pct_per_municipality: Dictionary mapping territory_id to missing percentage (0-100)
        indicators_with_high_missing: List of indicators with >50% missing values
        municipalities_with_high_missing: List of territory_ids with >70% missing indicators
        total_indicators: Total number of indicators analyzed
        total_municipalities: Total number of municipalities analyzed
        overall_completeness: Overall data completeness score (0-1)
    """
    missing_pct_per_indicator: Dict[str, float]
    missing_pct_per_municipality: Dict[int, float]
    indicators_with_high_missing: List[str]
    municipalities_with_high_missing: List[int]
    total_indicators: int
    total_municipalities: int
    overall_completeness: float


@dataclass
class PreprocessedData:
    """
    Container for preprocessed data and metadata.
    
    Attributes:
        df: Preprocessed DataFrame with added columns
        robust_stats: Dictionary mapping indicator names to RobustStats
        municipality_types: Dictionary mapping territory_id to municipality type
        normalized_indicators: List of indicators that were normalized
        transformed_indicators: List of indicators that were log-transformed
        winsorized_indicators: List of indicators that were winsorized
    """
    df: pd.DataFrame
    robust_stats: Dict[str, RobustStats]
    municipality_types: Dict[int, str]
    normalized_indicators: List[str]
    transformed_indicators: List[str]
    winsorized_indicators: List[str]


class MunicipalityClassifier:
    """
    Classifier for municipality types based on population and name patterns.
    
    Classifies municipalities into three categories:
    - 'capital': Regional capitals and federal cities
    - 'urban': Cities with population > threshold (default 50,000)
    - 'rural': Other municipalities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the municipality classifier.
        
        Args:
            config: Configuration dictionary with optional settings:
                - capital_cities: List of capital city names
                - urban_population_threshold: Population threshold for urban classification
        """
        self.config = config or {}
        
        # Get capital cities list from config or use defaults
        self.capital_cities = self.config.get('capital_cities', [
            'Москва', 'Санкт-Петербург', 'Севастополь',
            # Add more capital cities as needed
        ])
        
        # Get urban population threshold from config
        self.urban_population_threshold = self.config.get('urban_population_threshold', 50000)
        
        logger.info(f"MunicipalityClassifier initialized with {len(self.capital_cities)} capital cities "
                   f"and urban threshold of {self.urban_population_threshold}")
    
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify municipalities and add 'municipality_type' column.
        
        Classification logic:
        1. Check if municipality name matches capital cities list -> 'capital'
        2. Check if population > urban_population_threshold -> 'urban'
        3. Otherwise -> 'rural'
        
        Args:
            df: DataFrame with municipality data (must have 'municipal_district_name_short' 
                or similar name column, and optionally 'population_total' column)
        
        Returns:
            DataFrame with added 'municipality_type' column
        """
        logger.info("Classifying municipalities...")
        
        df = df.copy()
        
        # Initialize municipality_type column
        df['municipality_type'] = 'rural'  # Default to rural
        
        # Find name column
        name_col = None
        for col in ['municipal_district_name_short', 'municipal_name', 'name', 'district_name']:
            if col in df.columns:
                name_col = col
                break
        
        if name_col is None:
            logger.warning("No municipality name column found, cannot classify by name patterns")
        
        # Find population column
        pop_col = None
        for col in ['population_total', 'population', 'pop_total']:
            if col in df.columns:
                pop_col = col
                break
        
        if pop_col is None:
            logger.warning("No population column found, cannot classify by population threshold")
        
        # Classify by capital cities (highest priority)
        if name_col:
            for capital in self.capital_cities:
                capital_mask = df[name_col].str.contains(capital, case=False, na=False)
                df.loc[capital_mask, 'municipality_type'] = 'capital'
            
            capital_count = (df['municipality_type'] == 'capital').sum()
            logger.info(f"Classified {capital_count} municipalities as 'capital'")
        
        # Classify by population (for non-capitals)
        if pop_col:
            # Only reclassify municipalities that are still 'rural'
            urban_mask = (df['municipality_type'] == 'rural') & (df[pop_col] > self.urban_population_threshold)
            df.loc[urban_mask, 'municipality_type'] = 'urban'
            
            urban_count = (df['municipality_type'] == 'urban').sum()
            logger.info(f"Classified {urban_count} municipalities as 'urban' (population > {self.urban_population_threshold})")
        
        # Count final distribution
        rural_count = (df['municipality_type'] == 'rural').sum()
        logger.info(f"Classified {rural_count} municipalities as 'rural'")
        
        # Log summary
        type_counts = df['municipality_type'].value_counts().to_dict()
        logger.info(f"Municipality classification complete: {type_counts}")
        
        return df


class RobustStatsCalculator:
    """
    Calculator for robust statistical measures.
    
    Provides methods to calculate statistics that are resistant to outliers:
    - Median instead of mean
    - MAD (Median Absolute Deviation) instead of standard deviation
    - IQR (Interquartile Range)
    - Percentiles
    - Skewness
    """
    
    def __init__(self):
        """Initialize the robust statistics calculator."""
        logger.debug("RobustStatsCalculator initialized")
    
    def calculate_for_indicator(self, values: pd.Series) -> Optional[RobustStats]:
        """
        Calculate robust statistics for a single indicator.
        
        Args:
            values: Series of numeric values
        
        Returns:
            RobustStats object with calculated statistics, or None if insufficient data
        """
        # Remove missing values
        clean_values = values.dropna()
        
        if len(clean_values) < 3:
            logger.debug(f"Insufficient data for robust statistics: {len(clean_values)} values")
            return None
        
        # Calculate median
        median_val = clean_values.median()
        
        # Calculate MAD (Median Absolute Deviation)
        mad_val = np.median(np.abs(clean_values - median_val))
        
        # Calculate IQR
        q1 = clean_values.quantile(0.25)
        q3 = clean_values.quantile(0.75)
        iqr_val = q3 - q1
        
        # Calculate percentiles
        percentiles = {
            'p1': clean_values.quantile(0.01),
            'p5': clean_values.quantile(0.05),
            'p25': clean_values.quantile(0.25),
            'p75': clean_values.quantile(0.75),
            'p95': clean_values.quantile(0.95),
            'p99': clean_values.quantile(0.99)
        }
        
        # Calculate skewness
        skewness_val = clean_values.skew()
        
        return RobustStats(
            median=float(median_val),
            mad=float(mad_val),
            iqr=float(iqr_val),
            percentiles={k: float(v) for k, v in percentiles.items()},
            skewness=float(skewness_val),
            count=len(clean_values)
        )
    
    def calculate_for_dataframe(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> Dict[str, RobustStats]:
        """
        Calculate robust statistics for multiple indicators in a DataFrame.
        
        Args:
            df: DataFrame containing indicator columns
            indicators: List of indicator column names (if None, uses all numeric columns)
        
        Returns:
            Dictionary mapping indicator names to RobustStats objects
        """
        if indicators is None:
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns
            exclude_cols = ['territory_id', 'oktmo']
            indicators = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Calculating robust statistics for {len(indicators)} indicators...")
        
        robust_stats_dict = {}
        
        for indicator in indicators:
            if indicator not in df.columns:
                logger.warning(f"Indicator '{indicator}' not found in DataFrame")
                continue
            
            stats_obj = self.calculate_for_indicator(df[indicator])
            
            if stats_obj is not None:
                robust_stats_dict[indicator] = stats_obj
                logger.debug(f"Calculated robust stats for '{indicator}': "
                           f"median={stats_obj.median:.2f}, mad={stats_obj.mad:.2f}, "
                           f"skewness={stats_obj.skewness:.2f}")
            else:
                logger.debug(f"Skipped '{indicator}' due to insufficient data")
        
        logger.info(f"Robust statistics calculated for {len(robust_stats_dict)} indicators")
        
        return robust_stats_dict


class MissingnessAnalyzer:
    """
    Analyzer for missing data patterns and quality issues.
    
    Provides methods to:
    - Calculate missing percentage per indicator
    - Calculate missing percentage per municipality
    - Identify indicators and municipalities with high missingness
    """
    
    def __init__(self, high_missing_indicator_threshold: float = 50.0, 
                 high_missing_municipality_threshold: float = 70.0):
        """
        Initialize the missingness analyzer.
        
        Args:
            high_missing_indicator_threshold: Percentage threshold for flagging indicators (default 50%)
            high_missing_municipality_threshold: Percentage threshold for flagging municipalities (default 70%)
        """
        self.high_missing_indicator_threshold = high_missing_indicator_threshold
        self.high_missing_municipality_threshold = high_missing_municipality_threshold
        logger.debug(f"MissingnessAnalyzer initialized with thresholds: "
                    f"indicator={high_missing_indicator_threshold}%, "
                    f"municipality={high_missing_municipality_threshold}%")
    
    def analyze(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> MissingnessReport:
        """
        Perform comprehensive missingness analysis on the dataset.
        
        Args:
            df: DataFrame to analyze
            indicators: Optional list of indicator columns to analyze. If None, analyzes all numeric columns
                       excluding ID columns (territory_id, oktmo)
        
        Returns:
            MissingnessReport with detailed analysis results
        """
        logger.info("Starting missingness analysis...")
        
        # Identify indicators to analyze
        if indicators is None:
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns
            exclude_cols = ['territory_id', 'oktmo']
            indicators = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Analyzing missingness for {len(indicators)} indicators")
        
        # Calculate missing percentage per indicator
        missing_pct_per_indicator = self._calculate_missing_per_indicator(df, indicators)
        
        # Calculate missing percentage per municipality
        missing_pct_per_municipality = self._calculate_missing_per_municipality(df, indicators)
        
        # Identify indicators with high missing values
        indicators_with_high_missing = [
            indicator for indicator, pct in missing_pct_per_indicator.items()
            if pct > self.high_missing_indicator_threshold
        ]
        
        # Identify municipalities with high missing values
        municipalities_with_high_missing = [
            territory_id for territory_id, pct in missing_pct_per_municipality.items()
            if pct > self.high_missing_municipality_threshold
        ]
        
        # Calculate overall completeness
        total_cells = len(df) * len(indicators)
        if total_cells > 0:
            total_missing = sum(df[indicators].isnull().sum())
            overall_completeness = 1 - (total_missing / total_cells)
        else:
            overall_completeness = 1.0
        
        # Create report
        report = MissingnessReport(
            missing_pct_per_indicator=missing_pct_per_indicator,
            missing_pct_per_municipality=missing_pct_per_municipality,
            indicators_with_high_missing=indicators_with_high_missing,
            municipalities_with_high_missing=municipalities_with_high_missing,
            total_indicators=len(indicators),
            total_municipalities=len(df),
            overall_completeness=overall_completeness
        )
        
        # Log summary
        self._log_summary(report)
        
        return report
    
    def _calculate_missing_per_indicator(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, float]:
        """
        Calculate missing percentage for each indicator.
        
        Args:
            df: DataFrame with indicator data
            indicators: List of indicator column names
        
        Returns:
            Dictionary mapping indicator names to missing percentage (0-100)
        """
        logger.debug(f"Calculating missing percentage per indicator for {len(indicators)} indicators...")
        
        missing_pct = {}
        total_rows = len(df)
        
        if total_rows == 0:
            logger.warning("DataFrame is empty, cannot calculate missing percentages")
            return {indicator: 0.0 for indicator in indicators}
        
        for indicator in indicators:
            if indicator not in df.columns:
                logger.warning(f"Indicator '{indicator}' not found in DataFrame")
                missing_pct[indicator] = 100.0  # Consider it completely missing
                continue
            
            missing_count = df[indicator].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            missing_pct[indicator] = round(missing_percentage, 2)
            
            logger.debug(f"Indicator '{indicator}': {missing_count}/{total_rows} missing ({missing_percentage:.2f}%)")
        
        return missing_pct
    
    def _calculate_missing_per_municipality(self, df: pd.DataFrame, indicators: List[str]) -> Dict[int, float]:
        """
        Calculate missing percentage for each municipality.
        
        Args:
            df: DataFrame with municipality data (must have 'territory_id' column)
            indicators: List of indicator column names to consider
        
        Returns:
            Dictionary mapping territory_id to missing percentage (0-100)
        """
        logger.debug(f"Calculating missing percentage per municipality...")
        
        missing_pct = {}
        
        # Check if territory_id column exists
        if 'territory_id' not in df.columns:
            logger.warning("No territory_id column found, cannot calculate per-municipality missingness")
            return {}
        
        # Filter indicators to only those that exist in the DataFrame
        valid_indicators = [ind for ind in indicators if ind in df.columns]
        
        if not valid_indicators:
            logger.warning("No valid indicators found in DataFrame")
            return {}
        
        total_indicators = len(valid_indicators)
        
        # Calculate missing percentage for each municipality
        for idx, row in df.iterrows():
            territory_id = row['territory_id']
            
            # Skip if territory_id is null
            if pd.isnull(territory_id):
                continue
            
            # Count missing values for this municipality across all indicators
            missing_count = sum(pd.isnull(row[ind]) for ind in valid_indicators)
            missing_percentage = (missing_count / total_indicators) * 100
            missing_pct[int(territory_id)] = round(missing_percentage, 2)
        
        logger.debug(f"Calculated missing percentages for {len(missing_pct)} municipalities")
        
        return missing_pct
    
    def _log_summary(self, report: MissingnessReport):
        """
        Log summary of missingness analysis.
        
        Args:
            report: MissingnessReport to summarize
        """
        logger.info(f"Missingness analysis complete:")
        logger.info(f"  Total indicators analyzed: {report.total_indicators}")
        logger.info(f"  Total municipalities analyzed: {report.total_municipalities}")
        logger.info(f"  Overall data completeness: {report.overall_completeness:.2%}")
        
        # Log indicators with high missing values
        if report.indicators_with_high_missing:
            logger.warning(
                f"  Found {len(report.indicators_with_high_missing)} indicators with "
                f">{self.high_missing_indicator_threshold}% missing values",
                extra={
                    'high_missing_indicators_count': len(report.indicators_with_high_missing),
                    'high_missing_indicators_sample': report.indicators_with_high_missing[:5],
                    'data_quality_issue': 'high_missing_indicators'
                }
            )
            
            # Log top 5 indicators with highest missing percentages
            sorted_indicators = sorted(
                report.missing_pct_per_indicator.items(),
                key=lambda x: x[1],
                reverse=True
            )
            logger.warning("  Top indicators with missing values:")
            for indicator, pct in sorted_indicators[:5]:
                logger.warning(f"    {indicator}: {pct:.1f}% missing")
        else:
            logger.info(f"  No indicators with >{self.high_missing_indicator_threshold}% missing values")
        
        # Log municipalities with high missing values
        if report.municipalities_with_high_missing:
            logger.warning(
                f"  Found {len(report.municipalities_with_high_missing)} municipalities with "
                f">{self.high_missing_municipality_threshold}% missing indicators",
                extra={
                    'high_missing_municipalities_count': len(report.municipalities_with_high_missing),
                    'high_missing_municipalities_sample': report.municipalities_with_high_missing[:10],
                    'data_quality_issue': 'high_missing_municipalities'
                }
            )
            
            # Log top 10 municipalities with highest missing percentages
            sorted_municipalities = sorted(
                report.missing_pct_per_municipality.items(),
                key=lambda x: x[1],
                reverse=True
            )
            logger.warning("  Top municipalities with missing indicators:")
            for territory_id, pct in sorted_municipalities[:10]:
                logger.warning(f"    Territory {territory_id}: {pct:.1f}% missing")
        else:
            logger.info(f"  No municipalities with >{self.high_missing_municipality_threshold}% missing indicators")


class DataPreprocessor:
    """
    Main preprocessor for municipal data.
    
    Provides comprehensive preprocessing including:
    - Municipality classification
    - Robust statistics calculation
    - Data normalization (per-capita, log transformation, winsorization)
    - Missingness analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary with preprocessing settings
        """
        self.config = config
        
        # Get municipality classification config
        municipality_config = config.get('municipality_classification', {})
        self.municipality_classifier = MunicipalityClassifier(municipality_config)
        
        # Initialize robust stats calculator
        self.robust_stats_calculator = RobustStatsCalculator()
        
        # Initialize missingness analyzer
        missing_config = config.get('missing_value_handling', {})
        self.missingness_analyzer = MissingnessAnalyzer(
            high_missing_indicator_threshold=missing_config.get('indicator_threshold', 50.0),
            high_missing_municipality_threshold=missing_config.get('municipality_threshold', 70.0)
        )
        
        # Get robust statistics config
        robust_config = config.get('robust_statistics', {})
        self.use_robust_stats = robust_config.get('enabled', True)
        self.log_transform_threshold = robust_config.get('log_transform_skewness_threshold', 2.0)
        self.winsorization_limits = robust_config.get('winsorization_limits', [0.01, 0.99])
        self.apply_winsorization = robust_config.get('apply_winsorization', False)
        
        logger.info(f"DataPreprocessor initialized with robust_stats={self.use_robust_stats}, "
                   f"log_transform_threshold={self.log_transform_threshold}, "
                   f"apply_winsorization={self.apply_winsorization}")
    
    def preprocess(self, df: pd.DataFrame) -> PreprocessedData:
        """
        Perform complete preprocessing on the dataset.
        
        Steps:
        1. Classify municipalities by type
        2. Calculate robust statistics for all indicators
        3. Normalize indicators (per-capita where applicable)
        4. Apply transformations (log transform for skewed data)
        5. Apply winsorization (if enabled)
        
        Args:
            df: Raw DataFrame with municipal data
        
        Returns:
            PreprocessedData object containing processed DataFrame and metadata
        """
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Step 1: Classify municipalities
        df_processed = self.classify_municipalities(df_processed)
        
        # Extract municipality types mapping
        if 'territory_id' in df_processed.columns and 'municipality_type' in df_processed.columns:
            municipality_types = df_processed.set_index('territory_id')['municipality_type'].to_dict()
        else:
            municipality_types = {}
        
        # Step 2: Calculate robust statistics
        robust_stats = {}
        if self.use_robust_stats:
            robust_stats = self.calculate_robust_statistics(df_processed)
        
        # Step 3: Normalize indicators
        df_processed, normalized_indicators = self.normalize_indicators(df_processed)
        
        # Step 4: Apply transformations (log transform for skewed data)
        df_processed, transformed_indicators = self._apply_transformations(df_processed, robust_stats)
        
        # Step 5: Apply winsorization (if enabled)
        winsorized_indicators = []
        if self.apply_winsorization:
            df_processed, winsorized_indicators = self.winsorize_indicators(df_processed)
        
        logger.info(f"Preprocessing complete: {len(normalized_indicators)} normalized, "
                   f"{len(transformed_indicators)} transformed, "
                   f"{len(winsorized_indicators)} winsorized")
        
        return PreprocessedData(
            df=df_processed,
            robust_stats=robust_stats,
            municipality_types=municipality_types,
            normalized_indicators=normalized_indicators,
            transformed_indicators=transformed_indicators,
            winsorized_indicators=winsorized_indicators
        )
    
    def classify_municipalities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify municipalities by type (capital/urban/rural).
        
        Args:
            df: DataFrame with municipality data
        
        Returns:
            DataFrame with added 'municipality_type' column
        """
        return self.municipality_classifier.classify(df)
    
    def calculate_robust_statistics(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> Dict[str, RobustStats]:
        """
        Calculate robust statistics for indicators.
        
        Args:
            df: DataFrame with indicator data
            indicators: Optional list of specific indicators to calculate (if None, uses all numeric columns)
        
        Returns:
            Dictionary mapping indicator names to RobustStats objects
        """
        return self.robust_stats_calculator.calculate_for_dataframe(df, indicators)
    
    def analyze_missingness(self, df: pd.DataFrame, indicators: Optional[List[str]] = None) -> MissingnessReport:
        """
        Analyze missing data patterns in the dataset.
        
        Args:
            df: DataFrame to analyze
            indicators: Optional list of indicator columns to analyze (if None, uses all numeric columns)
        
        Returns:
            MissingnessReport with detailed analysis results
        """
        return self.missingness_analyzer.analyze(df, indicators)
    
    def normalize_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Normalize indicators to per-capita or percentage metrics where applicable.
        
        Creates new normalized columns with '_per_capita' suffix for indicators
        that should be normalized by population.
        
        Args:
            df: DataFrame with indicator data
        
        Returns:
            Tuple of (processed DataFrame, list of normalized indicator names)
        """
        logger.info("Normalizing indicators...")
        
        df_normalized = df.copy()
        normalized_indicators = []
        
        # Check if population column exists
        pop_col = None
        for col in ['population_total', 'population', 'pop_total']:
            if col in df.columns:
                pop_col = col
                break
        
        if pop_col is None:
            logger.warning("No population column found, skipping per-capita normalization")
            return df_normalized, normalized_indicators
        
        # Identify indicators that should be normalized
        # Typically: consumption, salary, migration indicators
        normalization_patterns = ['consumption_', 'salary_', 'migration_']
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for indicator in numeric_cols:
            # Skip if already normalized or is population itself
            if '_per_capita' in indicator or indicator == pop_col:
                continue
            
            # Check if indicator matches normalization patterns
            should_normalize = any(pattern in indicator.lower() for pattern in normalization_patterns)
            
            if should_normalize:
                # Create per-capita version
                per_capita_col = f"{indicator}_per_capita"
                
                # Calculate per-capita values (handle division by zero)
                df_normalized[per_capita_col] = df_normalized.apply(
                    lambda row: row[indicator] / row[pop_col] if pd.notna(row[pop_col]) and row[pop_col] > 0 else np.nan,
                    axis=1
                )
                
                normalized_indicators.append(per_capita_col)
                logger.debug(f"Created normalized indicator: {per_capita_col}")
        
        logger.info(f"Created {len(normalized_indicators)} per-capita normalized indicators")
        
        return df_normalized, normalized_indicators
    
    def winsorize_indicators(
        self, 
        df: pd.DataFrame, 
        indicators: Optional[List[str]] = None,
        limits: Optional[Tuple[float, float]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply winsorization to limit extreme outliers.
        
        Winsorization caps extreme values at specified percentiles to reduce
        the influence of outliers on statistical calculations.
        
        Creates new columns with '_winsorized' suffix.
        
        Args:
            df: DataFrame with indicator data
            indicators: List of indicators to winsorize (if None, uses all numeric columns)
            limits: Tuple of (lower_percentile, upper_percentile) for winsorization
                   Default is (0.01, 0.99) from config
        
        Returns:
            Tuple of (processed DataFrame, list of winsorized indicator names)
        """
        logger.info("Applying winsorization to indicators...")
        
        df_winsorized = df.copy()
        winsorized_indicators = []
        
        # Use config limits if not provided
        if limits is None:
            limits = tuple(self.winsorization_limits)
        
        lower_limit, upper_limit = limits
        
        # Get indicators to winsorize
        if indicators is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns and already processed columns
            exclude_patterns = ['territory_id', 'oktmo', '_winsorized', '_log', '_per_capita']
            indicators = [
                col for col in numeric_cols 
                if not any(pattern in col for pattern in exclude_patterns)
            ]
        
        for indicator in indicators:
            if indicator not in df.columns:
                logger.warning(f"Indicator '{indicator}' not found in DataFrame")
                continue
            
            # Get non-null values
            values = df_winsorized[indicator].dropna()
            
            if len(values) < 10:
                logger.debug(f"Skipping winsorization for '{indicator}': insufficient data ({len(values)} values)")
                continue
            
            # Calculate percentile thresholds
            lower_threshold = values.quantile(lower_limit)
            upper_threshold = values.quantile(upper_limit)
            
            # Create winsorized column
            winsorized_col = f"{indicator}_winsorized"
            
            # Apply winsorization: cap values at thresholds
            df_winsorized[winsorized_col] = df_winsorized[indicator].clip(
                lower=lower_threshold,
                upper=upper_threshold
            )
            
            winsorized_indicators.append(winsorized_col)
            
            # Count how many values were capped
            capped_count = (
                (df_winsorized[indicator] < lower_threshold) | 
                (df_winsorized[indicator] > upper_threshold)
            ).sum()
            
            logger.debug(f"Winsorized '{indicator}': capped {capped_count} values "
                        f"at [{lower_threshold:.2f}, {upper_threshold:.2f}]")
        
        logger.info(f"Applied winsorization to {len(winsorized_indicators)} indicators "
                   f"(limits: {lower_limit:.2%}, {upper_limit:.2%})")
        
        return df_winsorized, winsorized_indicators
    
    def filter_indicators_by_missingness(
        self, 
        df: pd.DataFrame, 
        indicators: Optional[List[str]] = None,
        threshold: float = 50.0
    ) -> Tuple[List[str], List[str]]:
        """
        Filter indicators based on missing value percentage.
        
        Identifies and filters out indicators with missing values exceeding the threshold.
        Logs warnings for each skipped indicator.
        
        Args:
            df: DataFrame with indicator data
            indicators: Optional list of indicators to filter (if None, uses all numeric columns)
            threshold: Missing percentage threshold (default 50.0%)
        
        Returns:
            Tuple of (valid_indicators, skipped_indicators)
        """
        logger.info(f"Filtering indicators with >{threshold}% missing values...")
        
        # Identify indicators to analyze
        if indicators is None:
            # Get all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude ID columns
            exclude_cols = ['territory_id', 'oktmo']
            indicators = [col for col in numeric_cols if col not in exclude_cols]
        
        valid_indicators = []
        skipped_indicators = []
        
        total_rows = len(df)
        
        if total_rows == 0:
            logger.warning("DataFrame is empty, cannot filter indicators")
            return indicators, []
        
        for indicator in indicators:
            if indicator not in df.columns:
                logger.warning(f"Indicator '{indicator}' not found in DataFrame, skipping")
                skipped_indicators.append(indicator)
                continue
            
            # Calculate missing percentage
            missing_count = df[indicator].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            if missing_percentage > threshold:
                # Skip this indicator
                skipped_indicators.append(indicator)
                logger.warning(
                    f"Skipping indicator '{indicator}' due to high missing values: "
                    f"{missing_count}/{total_rows} ({missing_percentage:.1f}% missing)",
                    extra={
                        'indicator': indicator,
                        'missing_count': missing_count,
                        'total_count': total_rows,
                        'missing_percentage': missing_percentage,
                        'threshold': threshold,
                        'data_quality_issue': 'high_missing_indicator'
                    }
                )
            else:
                # Keep this indicator
                valid_indicators.append(indicator)
                logger.debug(f"Indicator '{indicator}' passed filter: {missing_percentage:.1f}% missing")
        
        # Log summary
        logger.info(
            f"Indicator filtering complete: {len(valid_indicators)} valid, {len(skipped_indicators)} skipped",
            extra={
                'valid_indicators_count': len(valid_indicators),
                'skipped_indicators_count': len(skipped_indicators),
                'skipped_indicators': skipped_indicators[:10],  # Log first 10
                'threshold': threshold
            }
        )
        
        if skipped_indicators:
            logger.warning(
                f"Skipped {len(skipped_indicators)} indicators with >{threshold}% missing values. "
                f"These indicators will not be included in anomaly detection.",
                extra={
                    'skipped_indicators_count': len(skipped_indicators),
                    'skipped_indicators_list': skipped_indicators
                }
            )
        
        return valid_indicators, skipped_indicators
    
    def _apply_transformations(
        self, 
        df: pd.DataFrame, 
        robust_stats: Dict[str, RobustStats]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply transformations to skewed indicators.
        
        Applies log transformation to indicators with high skewness (> threshold).
        Creates new columns with '_log' suffix.
        
        Args:
            df: DataFrame with indicator data
            robust_stats: Dictionary of robust statistics for indicators
        
        Returns:
            Tuple of (processed DataFrame, list of transformed indicator names)
        """
        logger.info("Applying transformations to skewed indicators...")
        
        df_transformed = df.copy()
        transformed_indicators = []
        
        if not robust_stats:
            logger.info("No robust statistics available, skipping transformations")
            return df_transformed, transformed_indicators
        
        for indicator, stats in robust_stats.items():
            # Check if indicator has high skewness
            if abs(stats.skewness) > self.log_transform_threshold:
                # Apply log transformation
                log_col = f"{indicator}_log"
                
                # Log transform (add small constant to handle zeros)
                df_transformed[log_col] = df_transformed[indicator].apply(
                    lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else np.nan
                )
                
                transformed_indicators.append(log_col)
                logger.debug(f"Applied log transformation to '{indicator}' (skewness={stats.skewness:.2f})")
        
        logger.info(f"Applied log transformation to {len(transformed_indicators)} indicators")
        
        return df_transformed, transformed_indicators
