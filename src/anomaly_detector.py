"""
Anomaly Detection Module for СберИндекс Data Analysis

This module provides base and concrete classes for detecting various types of anomalies
in municipal data from СберИндекс and Росстат sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import logging

import pandas as pd
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detectors.
    
    All concrete detector classes must inherit from this class and implement
    the detect() method. Provides common functionality for severity score
    calculation and anomaly record creation.
    """
    
    def __init__(self, config: Dict[str, Any], source_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Configuration dictionary containing thresholds and parameters
            source_mapping: Optional dictionary mapping column names to data sources
        """
        self.config = config
        self.source_mapping = source_mapping or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the provided dataset.
        
        Args:
            df: DataFrame containing municipal data with indicators
            
        Returns:
            DataFrame with detected anomalies containing columns:
                - anomaly_id: Unique identifier (UUID)
                - territory_id: Municipal territory identifier
                - municipal_name: Name of municipality
                - region_name: Name of region
                - indicator: Name of the indicator with anomaly
                - anomaly_type: Type of anomaly detected
                - actual_value: Actual value of the indicator
                - expected_value: Expected value (may be None)
                - deviation: Absolute deviation from expected
                - deviation_pct: Percentage deviation
                - severity_score: Severity score (0-100)
                - z_score: Z-score if applicable (may be None)
                - data_source: Source of data ('sberindex' or 'rosstat')
                - detection_method: Method used for detection
                - description: Human-readable description
                - potential_explanation: Possible explanation for anomaly
                - detected_at: Timestamp of detection
        """
        pass
    
    def calculate_severity_score(
        self,
        deviation: float,
        z_score: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> float:
        """
        Calculate severity score for an anomaly based on deviation magnitude.
        
        The severity score is a value between 0 and 100, where:
        - 0-25: Low severity
        - 25-50: Medium severity
        - 50-75: High severity
        - 75-100: Critical severity
        
        Args:
            deviation: Absolute deviation from expected value
            z_score: Z-score of the value (optional)
            percentile: Percentile rank of the value (optional)
            
        Returns:
            Severity score between 0 and 100
        """
        # Base score on z-score if available
        if z_score is not None:
            abs_z = abs(z_score)
            if abs_z >= 5:
                return 100.0
            elif abs_z >= 4:
                return 90.0
            elif abs_z >= 3:
                return 70.0
            elif abs_z >= 2:
                return 50.0
            else:
                return min(abs_z * 25, 100.0)
        
        # Use percentile if available
        if percentile is not None:
            if percentile >= 99 or percentile <= 1:
                return 90.0
            elif percentile >= 95 or percentile <= 5:
                return 70.0
            elif percentile >= 90 or percentile <= 10:
                return 50.0
            else:
                return 30.0
        
        # Fallback to deviation-based scoring
        if abs(deviation) > 1000:
            return 100.0
        elif abs(deviation) > 500:
            return 80.0
        elif abs(deviation) > 100:
            return 60.0
        elif abs(deviation) > 50:
            return 40.0
        else:
            return 20.0
    
    def create_anomaly_record(
        self,
        territory_id: Any,
        municipal_name: str,
        region_name: str,
        indicator: str,
        anomaly_type: str,
        actual_value: float,
        expected_value: Optional[float],
        deviation: float,
        deviation_pct: float,
        severity_score: float,
        z_score: Optional[float],
        data_source: str,
        detection_method: str,
        description: str,
        potential_explanation: str = ""
    ) -> Dict[str, Any]:
        """
        Create a standardized anomaly record.
        
        Args:
            territory_id: Municipal territory identifier
            municipal_name: Name of municipality
            region_name: Name of region
            indicator: Name of the indicator with anomaly
            anomaly_type: Type of anomaly
            actual_value: Actual value of the indicator
            expected_value: Expected value (may be None)
            deviation: Absolute deviation from expected
            deviation_pct: Percentage deviation
            severity_score: Severity score (0-100)
            z_score: Z-score if applicable (may be None)
            data_source: Source of data
            detection_method: Method used for detection
            description: Human-readable description
            potential_explanation: Possible explanation for anomaly
            
        Returns:
            Dictionary containing all anomaly record fields
        """
        return {
            'anomaly_id': str(uuid.uuid4()),
            'territory_id': territory_id,
            'municipal_name': municipal_name,
            'region_name': region_name,
            'indicator': indicator,
            'anomaly_type': anomaly_type,
            'actual_value': actual_value,
            'expected_value': expected_value,
            'deviation': deviation,
            'deviation_pct': deviation_pct,
            'severity_score': severity_score,
            'z_score': z_score,
            'data_source': data_source,
            'detection_method': detection_method,
            'description': description,
            'potential_explanation': potential_explanation,
            'detected_at': datetime.now()
        }
    
    def get_data_source(self, indicator: str) -> str:
        """
        Get the data source for an indicator using explicit source mapping.
        
        Uses the source mapping created by DataLoader.create_source_mapping()
        to determine the data source. Logs warnings for ambiguous or unknown columns.
        
        Args:
            indicator: Name of the indicator
            
        Returns:
            Data source ('sberindex', 'rosstat', 'municipal_dict', or 'unknown')
        """
        if not self.source_mapping:
            # Fallback to old heuristic method if no mapping provided
            self.logger.warning(f"No source mapping available, using fallback heuristics for '{indicator}'")
            return self._fallback_determine_source(indicator)
        
        source = self.source_mapping.get(indicator, 'unknown')
        
        # Log warning for ambiguous or unknown columns
        if source == 'unknown':
            self.logger.warning(
                f"Column '{indicator}' has unknown data source. "
                f"Consider updating source mapping or column naming convention."
            )
        
        return source
    
    def _fallback_determine_source(self, indicator: str) -> str:
        """
        Fallback method to determine data source using heuristics.
        
        This is used when no explicit source mapping is provided.
        
        Args:
            indicator: Name of the indicator
            
        Returns:
            Data source ('sberindex' or 'rosstat')
        """
        # СберИндекс indicators typically contain these keywords
        sberindex_keywords = ['consumption', 'connection', 'market_access']
        
        # Росстат indicators typically contain these keywords
        rosstat_keywords = ['population', 'migration', 'salary']
        
        indicator_lower = indicator.lower()
        
        for keyword in sberindex_keywords:
            if keyword in indicator_lower:
                return 'sberindex'
        
        for keyword in rosstat_keywords:
            if keyword in indicator_lower:
                return 'rosstat'
        
        # Default to sberindex if unclear
        return 'sberindex'


class StatisticalOutlierDetector(BaseAnomalyDetector):
    """
    Detector for statistical outliers using multiple methods.
    
    Implements three detection methods:
    1. Z-score based outlier detection
    2. IQR (Interquartile Range) based outlier detection
    3. Percentile-based outlier detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the statistical outlier detector.
        
        Args:
            config: Configuration dictionary containing statistical thresholds
        """
        super().__init__(config)
        self.z_score_threshold = config.get('thresholds', {}).get('statistical', {}).get('z_score', 3.0)
        self.iqr_multiplier = config.get('thresholds', {}).get('statistical', {}).get('iqr_multiplier', 1.5)
        self.percentile_lower = config.get('thresholds', {}).get('statistical', {}).get('percentile_lower', 1)
        self.percentile_upper = config.get('thresholds', {}).get('statistical', {}).get('percentile_upper', 99)
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect statistical outliers using all available methods.
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            DataFrame with detected anomalies
        """
        self.logger.info("Starting statistical outlier detection")
        
        all_anomalies = []
        
        # Detect using z-score method
        zscore_anomalies = self.detect_zscore_outliers(df)
        all_anomalies.extend(zscore_anomalies)
        self.logger.info(f"Z-score method detected {len(zscore_anomalies)} anomalies")
        
        # Detect using IQR method
        iqr_anomalies = self.detect_iqr_outliers(df)
        all_anomalies.extend(iqr_anomalies)
        self.logger.info(f"IQR method detected {len(iqr_anomalies)} anomalies")
        
        # Detect using percentile method
        percentile_anomalies = self.detect_percentile_outliers(df)
        all_anomalies.extend(percentile_anomalies)
        self.logger.info(f"Percentile method detected {len(percentile_anomalies)} anomalies")
        
        if not all_anomalies:
            self.logger.info("No statistical outliers detected")
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        anomalies_df = pd.DataFrame(all_anomalies)
        
        # Remove duplicate anomalies (same territory_id and indicator)
        anomalies_df = anomalies_df.drop_duplicates(
            subset=['territory_id', 'indicator', 'detection_method'],
            keep='first'
        )
        
        self.logger.info(f"Total unique statistical outliers detected: {len(anomalies_df)}")
        
        return anomalies_df
    
    def detect_zscore_outliers(self, df: pd.DataFrame, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect outliers using z-score method.
        
        Identifies values that deviate more than the threshold number of standard
        deviations from the mean.
        
        Args:
            df: DataFrame containing municipal data
            threshold: Z-score threshold (uses config default if None)
            
        Returns:
            List of anomaly records
        """
        if threshold is None:
            threshold = self.z_score_threshold
        
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for indicator in indicator_cols:
            # Skip if column has too many missing values
            if df[indicator].isna().sum() / len(df) > 0.5:
                continue
            
            # Calculate z-scores maintaining original index
            values = df[indicator].dropna()
            if len(values) < 3:  # Need at least 3 values for meaningful statistics
                continue
            
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val == 0:  # All values are the same
                continue
            
            # Calculate z-scores and maintain index alignment
            z_scores_array = np.abs(stats.zscore(values))
            z_scores_series = pd.Series(z_scores_array, index=values.index)
            
            # Find outliers using boolean indexing
            outlier_mask = z_scores_series > threshold
            outlier_indices = z_scores_series[outlier_mask].index
            
            for idx in outlier_indices:
                # Safe index access - check if index exists in original dataframe
                if idx not in df.index:
                    self.logger.warning(f"Index {idx} not found in dataframe, skipping")
                    continue
                
                # Use .loc[] for safe access
                actual_value = df.loc[idx, indicator]
                z_score = z_scores_series.loc[idx]
                deviation = actual_value - mean_val
                deviation_pct = (deviation / mean_val * 100) if mean_val != 0 else 0
                
                severity_score = self.calculate_severity_score(
                    deviation=deviation,
                    z_score=z_score
                )
                
                # Determine data source from indicator name using explicit mapping
                data_source = self.get_data_source(indicator)
                
                anomaly = self.create_anomaly_record(
                    territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                    municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator=indicator,
                    anomaly_type='statistical_outlier',
                    actual_value=float(actual_value),
                    expected_value=float(mean_val),
                    deviation=float(deviation),
                    deviation_pct=float(deviation_pct),
                    severity_score=severity_score,
                    z_score=float(z_score),
                    data_source=data_source,
                    detection_method='z_score',
                    description=f"{indicator} value {actual_value:.2f} deviates {z_score:.2f} standard deviations from mean {mean_val:.2f}",
                    potential_explanation="Statistical outlier - value significantly differs from typical range"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_iqr_outliers(self, df: pd.DataFrame, multiplier: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Detect outliers using IQR (Interquartile Range) method.
        
        Identifies values that fall outside the range:
        [Q1 - multiplier * IQR, Q3 + multiplier * IQR]
        
        Args:
            df: DataFrame containing municipal data
            multiplier: IQR multiplier (uses config default if None)
            
        Returns:
            List of anomaly records
        """
        if multiplier is None:
            multiplier = self.iqr_multiplier
        
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for indicator in indicator_cols:
            # Skip if column has too many missing values
            if df[indicator].isna().sum() / len(df) > 0.5:
                continue
            
            # Maintain original indices when working with subsets
            values = df[indicator].dropna()
            if len(values) < 3:
                continue
            
            # Calculate quartiles and IQR
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # No variation in data
                continue
            
            # Calculate bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            
            # Find outliers using boolean indexing
            outlier_mask = (values < lower_bound) | (values > upper_bound)
            outlier_indices = values[outlier_mask].index
            
            for idx in outlier_indices:
                # Safe index access - check if index exists in original dataframe
                if idx not in df.index:
                    self.logger.warning(f"Index {idx} not found in dataframe, skipping")
                    continue
                
                # Use .loc[] for safe access
                actual_value = df.loc[idx, indicator]
                median_val = values.median()
                
                # Calculate deviation from nearest bound
                if actual_value < lower_bound:
                    expected_value = lower_bound
                    deviation = actual_value - lower_bound
                else:
                    expected_value = upper_bound
                    deviation = actual_value - upper_bound
                
                deviation_pct = (deviation / median_val * 100) if median_val != 0 else 0
                
                # Calculate percentile for severity scoring
                percentile = stats.percentileofscore(values, actual_value)
                
                severity_score = self.calculate_severity_score(
                    deviation=deviation,
                    percentile=percentile
                )
                
                # Determine data source from indicator name using explicit mapping
                data_source = self.get_data_source(indicator)
                
                anomaly = self.create_anomaly_record(
                    territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                    municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator=indicator,
                    anomaly_type='statistical_outlier',
                    actual_value=float(actual_value),
                    expected_value=float(expected_value),
                    deviation=float(deviation),
                    deviation_pct=float(deviation_pct),
                    severity_score=severity_score,
                    z_score=None,
                    data_source=data_source,
                    detection_method='iqr',
                    description=f"{indicator} value {actual_value:.2f} falls outside IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                    potential_explanation="IQR outlier - value outside typical interquartile range"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_percentile_outliers(
        self,
        df: pd.DataFrame,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect outliers in top/bottom percentiles.
        
        Identifies values in the extreme percentiles (e.g., top 1% or bottom 1%).
        
        Args:
            df: DataFrame containing municipal data
            lower: Lower percentile threshold (uses config default if None)
            upper: Upper percentile threshold (uses config default if None)
            
        Returns:
            List of anomaly records
        """
        if lower is None:
            lower = self.percentile_lower
        if upper is None:
            upper = self.percentile_upper
        
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for indicator in indicator_cols:
            # Skip if column has too many missing values
            if df[indicator].isna().sum() / len(df) > 0.5:
                continue
            
            # Maintain original indices when working with subsets
            values = df[indicator].dropna()
            if len(values) < 10:  # Need sufficient data for percentile analysis
                continue
            
            # Calculate percentile thresholds
            lower_threshold = np.percentile(values, lower)
            upper_threshold = np.percentile(values, upper)
            median_val = values.median()
            
            # Find outliers using boolean indexing
            outlier_mask = (values <= lower_threshold) | (values >= upper_threshold)
            outlier_indices = values[outlier_mask].index
            
            for idx in outlier_indices:
                # Safe index access - check if index exists in original dataframe
                if idx not in df.index:
                    self.logger.warning(f"Index {idx} not found in dataframe, skipping")
                    continue
                
                # Use .loc[] for safe access
                actual_value = df.loc[idx, indicator]
                percentile = stats.percentileofscore(values, actual_value)
                
                # Determine if it's a high or low outlier
                if actual_value <= lower_threshold:
                    expected_value = lower_threshold
                    outlier_direction = "bottom"
                else:
                    expected_value = upper_threshold
                    outlier_direction = "top"
                
                deviation = actual_value - median_val
                deviation_pct = (deviation / median_val * 100) if median_val != 0 else 0
                
                severity_score = self.calculate_severity_score(
                    deviation=deviation,
                    percentile=percentile
                )
                
                # Determine data source from indicator name using explicit mapping
                data_source = self.get_data_source(indicator)
                
                anomaly = self.create_anomaly_record(
                    territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                    municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator=indicator,
                    anomaly_type='statistical_outlier',
                    actual_value=float(actual_value),
                    expected_value=float(expected_value),
                    deviation=float(deviation),
                    deviation_pct=float(deviation_pct),
                    severity_score=severity_score,
                    z_score=None,
                    data_source=data_source,
                    detection_method='percentile',
                    description=f"{indicator} value {actual_value:.2f} is in {outlier_direction} {lower if outlier_direction == 'bottom' else 100-upper}% (percentile: {percentile:.1f})",
                    potential_explanation=f"Extreme value in {outlier_direction} percentile range"
                )
                
                anomalies.append(anomaly)
        
        return anomalies


class CrossSourceComparator(BaseAnomalyDetector):
    """
    Detector for discrepancies between СберИндекс and Росстат data sources.
    
    Compares comparable indicators from different sources to identify:
    1. Low correlations between related indicators
    2. Large discrepancies (>50% differences)
    3. Municipalities with systematic data quality issues
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cross-source comparator.
        
        Args:
            config: Configuration dictionary containing cross-source thresholds
        """
        super().__init__(config)
        self.correlation_threshold = config.get('thresholds', {}).get('cross_source', {}).get('correlation_threshold', 0.5)
        self.discrepancy_threshold = config.get('thresholds', {}).get('cross_source', {}).get('discrepancy_threshold', 50)
        
        # Define comparable indicator pairs (СберИндекс indicator -> Росстат indicator)
        self.comparable_pairs = self._define_comparable_pairs()
    
    def _define_comparable_pairs(self) -> Dict[str, str]:
        """
        Define pairs of comparable indicators between СберИндекс and Росстат.
        
        Returns:
            Dictionary mapping СберИндекс indicators to Росстат indicators
        """
        # This mapping should be adjusted based on actual column names in the data
        # These are examples of potentially comparable indicators
        return {
            # Economic activity indicators
            'consumption_total': 'salary_avg',
            'market_access': 'population_total',
            # Add more pairs as identified in the actual data
        }
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect cross-source discrepancies between СберИндекс and Росстат data.
        
        Args:
            df: DataFrame containing merged municipal data from both sources
            
        Returns:
            DataFrame with detected anomalies
        """
        self.logger.info("Starting cross-source comparison detection")
        
        all_anomalies = []
        
        # Calculate correlations between comparable indicators
        correlation_results = self.calculate_correlations(df)
        self.logger.info(f"Calculated correlations for {len(correlation_results)} indicator pairs")
        
        # Detect large discrepancies
        discrepancy_anomalies = self.detect_large_discrepancies(df)
        all_anomalies.extend(discrepancy_anomalies)
        self.logger.info(f"Detected {len(discrepancy_anomalies)} large discrepancies")
        
        if not all_anomalies:
            self.logger.info("No cross-source discrepancies detected")
            return pd.DataFrame()
        
        # Convert to DataFrame
        anomalies_df = pd.DataFrame(all_anomalies)
        
        # Rank by discrepancy magnitude
        anomalies_df = self.rank_by_discrepancy(anomalies_df)
        
        self.logger.info(f"Total cross-source anomalies detected: {len(anomalies_df)}")
        
        return anomalies_df
    
    def calculate_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate correlation coefficients between comparable indicators.
        
        Computes Pearson correlation between СберИндекс and Росстат indicators
        that are expected to be related.
        
        Args:
            df: DataFrame containing merged data from both sources
            
        Returns:
            Dictionary mapping indicator pairs to correlation coefficients
        """
        correlations = {}
        
        for sber_indicator, rosstat_indicator in self.comparable_pairs.items():
            # Check if both indicators exist in the dataframe
            if sber_indicator not in df.columns or rosstat_indicator not in df.columns:
                self.logger.debug(f"Skipping pair {sber_indicator} - {rosstat_indicator}: columns not found")
                continue
            
            # Get non-null values for both indicators
            valid_mask = df[sber_indicator].notna() & df[rosstat_indicator].notna()
            
            if valid_mask.sum() < 3:  # Need at least 3 pairs for meaningful correlation
                self.logger.debug(f"Skipping pair {sber_indicator} - {rosstat_indicator}: insufficient data")
                continue
            
            sber_values = df.loc[valid_mask, sber_indicator]
            rosstat_values = df.loc[valid_mask, rosstat_indicator]
            
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(sber_values, rosstat_values)
            
            pair_key = f"{sber_indicator} vs {rosstat_indicator}"
            correlations[pair_key] = correlation
            
            # Log low correlations as potential data quality issues
            if abs(correlation) < self.correlation_threshold:
                self.logger.warning(
                    f"Low correlation ({correlation:.3f}) between {sber_indicator} and {rosstat_indicator}"
                )
        
        return correlations
    
    def detect_large_discrepancies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect municipalities with large discrepancies between data sources.
        
        Identifies cases where СберИндекс indicators deviate more than the
        threshold percentage from corresponding Росстат indicators.
        
        Args:
            df: DataFrame containing merged data from both sources
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        for sber_indicator, rosstat_indicator in self.comparable_pairs.items():
            # Check if both indicators exist in the dataframe
            if sber_indicator not in df.columns or rosstat_indicator not in df.columns:
                continue
            
            # Get valid data for both indicators
            valid_mask = df[sber_indicator].notna() & df[rosstat_indicator].notna()
            valid_df = df[valid_mask].copy()
            
            if len(valid_df) == 0:
                continue
            
            # Calculate percentage differences for each municipality
            percentage_diffs = self._calculate_percentage_differences(
                valid_df[sber_indicator],
                valid_df[rosstat_indicator]
            )
            
            valid_df['percentage_diff'] = percentage_diffs
            
            # Find municipalities with discrepancies exceeding threshold
            discrepancy_mask = abs(valid_df['percentage_diff']) > self.discrepancy_threshold
            discrepancy_df = valid_df[discrepancy_mask]
            
            for idx in discrepancy_df.index:
                sber_value = df.loc[idx, sber_indicator]
                rosstat_value = df.loc[idx, rosstat_indicator]
                pct_diff = discrepancy_df.loc[idx, 'percentage_diff']
                
                # Calculate absolute deviation
                deviation = sber_value - rosstat_value
                
                # Calculate severity based on percentage difference
                severity_score = self._calculate_discrepancy_severity(abs(pct_diff))
                
                # Determine which source is higher
                if sber_value > rosstat_value:
                    direction = "higher"
                    comparison = f"СберИндекс {abs(pct_diff):.1f}% higher than Росстат"
                else:
                    direction = "lower"
                    comparison = f"СберИндекс {abs(pct_diff):.1f}% lower than Росстат"
                
                anomaly = self.create_anomaly_record(
                    territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                    municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator=f"{sber_indicator} vs {rosstat_indicator}",
                    anomaly_type='cross_source_discrepancy',
                    actual_value=float(sber_value),
                    expected_value=float(rosstat_value),
                    deviation=float(deviation),
                    deviation_pct=float(pct_diff),
                    severity_score=severity_score,
                    z_score=None,
                    data_source='cross_source',
                    detection_method='percentage_difference',
                    description=f"{comparison}: СберИндекс={sber_value:.2f}, Росстат={rosstat_value:.2f}",
                    potential_explanation=f"Significant discrepancy between data sources - {direction} value in СберИндекс may indicate data quality issue or real difference in measurement methodology"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_percentage_differences(
        self,
        sber_values: pd.Series,
        rosstat_values: pd.Series
    ) -> pd.Series:
        """
        Calculate percentage differences between two series.
        
        Uses Росстат values as the baseline for percentage calculation.
        
        Args:
            sber_values: Series of СберИндекс values
            rosstat_values: Series of Росстат values
            
        Returns:
            Series of percentage differences
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_diff = ((sber_values - rosstat_values) / rosstat_values * 100)
            
            # Handle cases where Росстат value is zero
            # Use absolute difference if baseline is zero
            zero_mask = rosstat_values == 0
            if zero_mask.any():
                pct_diff[zero_mask] = np.nan
        
        return pct_diff
    
    def _calculate_discrepancy_severity(self, abs_pct_diff: float) -> float:
        """
        Calculate severity score based on percentage difference magnitude.
        
        Args:
            abs_pct_diff: Absolute percentage difference
            
        Returns:
            Severity score between 0 and 100
        """
        if abs_pct_diff >= 200:
            return 100.0
        elif abs_pct_diff >= 150:
            return 90.0
        elif abs_pct_diff >= 100:
            return 80.0
        elif abs_pct_diff >= 75:
            return 70.0
        elif abs_pct_diff >= 50:
            return 60.0
        else:
            # Linear scaling for differences below threshold
            return min(abs_pct_diff * 1.2, 60.0)
    
    def rank_by_discrepancy(self, anomalies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort municipalities by discrepancy magnitude.
        
        Ranks anomalies by severity score and absolute percentage difference.
        
        Args:
            anomalies_df: DataFrame containing detected anomalies
            
        Returns:
            Sorted DataFrame with anomalies ranked by severity
        """
        if anomalies_df.empty:
            return anomalies_df
        
        # Sort by severity score (descending) and then by absolute deviation percentage
        anomalies_df = anomalies_df.sort_values(
            by=['severity_score', 'deviation_pct'],
            ascending=[False, False]
        ).reset_index(drop=True)
        
        # Add rank column
        anomalies_df['discrepancy_rank'] = range(1, len(anomalies_df) + 1)
        
        return anomalies_df


class TemporalAnomalyDetector(BaseAnomalyDetector):
    """
    Detector for temporal anomalies in time-series data.
    
    Identifies:
    1. Sudden spikes (>100% growth or <-50% drops)
    2. Trend reversals (significant direction changes)
    3. High volatility (excessive variation over time)
    4. Seasonal anomalies (deviations from seasonal patterns)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the temporal anomaly detector.
        
        Args:
            config: Configuration dictionary containing temporal thresholds
        """
        super().__init__(config)
        self.spike_threshold = config.get('thresholds', {}).get('temporal', {}).get('spike_threshold', 100)
        self.drop_threshold = config.get('thresholds', {}).get('temporal', {}).get('drop_threshold', -50)
        self.volatility_multiplier = config.get('thresholds', {}).get('temporal', {}).get('volatility_multiplier', 2.0)
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect temporal anomalies in the dataset.
        
        Args:
            df: DataFrame containing municipal data with temporal information
            
        Returns:
            DataFrame with detected anomalies
        """
        self.logger.info("Starting temporal anomaly detection")
        
        all_anomalies = []
        
        # Check if temporal data is available
        if not self._has_temporal_data(df):
            self.logger.warning("No temporal data available for temporal anomaly detection")
            return pd.DataFrame()
        
        # Detect sudden spikes
        spike_anomalies = self.detect_sudden_spikes(df)
        all_anomalies.extend(spike_anomalies)
        self.logger.info(f"Detected {len(spike_anomalies)} sudden spikes")
        
        # Detect trend reversals
        reversal_anomalies = self.detect_trend_reversals(df)
        all_anomalies.extend(reversal_anomalies)
        self.logger.info(f"Detected {len(reversal_anomalies)} trend reversals")
        
        # Detect high volatility
        volatility_anomalies = self.detect_high_volatility(df)
        all_anomalies.extend(volatility_anomalies)
        self.logger.info(f"Detected {len(volatility_anomalies)} high volatility cases")
        
        # Detect seasonal anomalies if seasonal data is available
        seasonal_anomalies = self.detect_seasonal_anomalies(df)
        all_anomalies.extend(seasonal_anomalies)
        self.logger.info(f"Detected {len(seasonal_anomalies)} seasonal anomalies")
        
        if not all_anomalies:
            self.logger.info("No temporal anomalies detected")
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        anomalies_df = pd.DataFrame(all_anomalies)
        
        # Remove duplicate anomalies (same territory_id and indicator)
        anomalies_df = anomalies_df.drop_duplicates(
            subset=['territory_id', 'indicator', 'detection_method'],
            keep='first'
        )
        
        self.logger.info(f"Total unique temporal anomalies detected: {len(anomalies_df)}")
        
        return anomalies_df
    
    def _has_temporal_data(self, df: pd.DataFrame) -> bool:
        """
        Check if the dataset contains temporal information.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if temporal data is available, False otherwise
        """
        # Check for common temporal column names
        temporal_columns = ['date', 'period', 'year', 'month', 'quarter', 'timestamp']
        
        for col in temporal_columns:
            if col in df.columns:
                return True
        
        # Check if there are multiple rows per territory (indicating time series)
        if 'territory_id' in df.columns:
            territory_counts = df['territory_id'].value_counts()
            if (territory_counts > 1).any():
                return True
        
        return False
    
    def detect_sudden_spikes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect sudden spikes where growth rate exceeds threshold.
        
        Identifies period-over-period changes that exceed 100% growth or
        drop below -50% in a single period.
        
        Args:
            df: DataFrame containing temporal municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo', 'year', 'month', 'quarter']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by territory if available
        if 'territory_id' in df.columns:
            grouped = df.groupby('territory_id')
        else:
            # If no territory_id, treat entire dataset as one series
            grouped = [(None, df)]
        
        for territory_id, group_df in grouped:
            # Sort by temporal column if available
            group_df = self._sort_by_temporal_column(group_df)
            
            if len(group_df) < 2:
                continue
            
            for indicator in indicator_cols:
                # Skip if column has too many missing values
                if group_df[indicator].isna().sum() / len(group_df) > 0.5:
                    continue
                
                # Calculate period-over-period growth rates
                growth_rates = self._calculate_growth_rates(group_df[indicator])
                
                if growth_rates.empty:
                    continue
                
                # Find spikes exceeding threshold
                spike_mask = (growth_rates > self.spike_threshold) | (growth_rates < self.drop_threshold)
                spike_indices = growth_rates[spike_mask].index
                
                for idx in spike_indices:
                    # Get the position in the original group
                    pos = group_df.index.get_loc(idx)
                    
                    if pos == 0:
                        continue
                    
                    current_value = group_df.loc[idx, indicator]
                    previous_idx = group_df.index[pos - 1]
                    previous_value = group_df.loc[previous_idx, indicator]
                    growth_rate = growth_rates.loc[idx]
                    
                    # Determine spike type
                    if growth_rate > self.spike_threshold:
                        spike_type = "sudden_spike"
                        description = f"{indicator} increased by {growth_rate:.1f}% from {previous_value:.2f} to {current_value:.2f}"
                    else:
                        spike_type = "sudden_drop"
                        description = f"{indicator} decreased by {abs(growth_rate):.1f}% from {previous_value:.2f} to {current_value:.2f}"
                    
                    deviation = current_value - previous_value
                    
                    # Calculate severity based on magnitude of change
                    severity_score = self._calculate_spike_severity(abs(growth_rate))
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=territory_id if territory_id is not None else idx,
                        municipal_name=group_df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in group_df.columns else 'Unknown',
                        region_name=group_df.loc[idx, 'region_name'] if 'region_name' in group_df.columns else 'Unknown',
                        indicator=indicator,
                        anomaly_type='temporal_anomaly',
                        actual_value=float(current_value),
                        expected_value=float(previous_value),
                        deviation=float(deviation),
                        deviation_pct=float(growth_rate),
                        severity_score=severity_score,
                        z_score=None,
                        data_source=data_source,
                        detection_method=spike_type,
                        description=description,
                        potential_explanation=f"Sudden change in {indicator} - may indicate data error, policy change, or significant event"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_trend_reversals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect trend reversals where indicator direction changes significantly.
        
        Identifies points where a consistent trend (increasing or decreasing)
        suddenly reverses direction.
        
        Args:
            df: DataFrame containing temporal municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo', 'year', 'month', 'quarter']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by territory if available
        if 'territory_id' in df.columns:
            grouped = df.groupby('territory_id')
        else:
            grouped = [(None, df)]
        
        for territory_id, group_df in grouped:
            # Sort by temporal column if available
            group_df = self._sort_by_temporal_column(group_df)
            
            if len(group_df) < 4:  # Need at least 4 points to detect trend reversal
                continue
            
            for indicator in indicator_cols:
                # Skip if column has too many missing values
                if group_df[indicator].isna().sum() / len(group_df) > 0.5:
                    continue
                
                # Calculate growth rates
                growth_rates = self._calculate_growth_rates(group_df[indicator])
                
                if len(growth_rates) < 3:
                    continue
                
                # Detect trend reversals by looking at sign changes in growth rates
                # A reversal occurs when the trend changes from positive to negative or vice versa
                for i in range(2, len(growth_rates)):
                    idx = growth_rates.index[i]
                    pos = group_df.index.get_loc(idx)
                    
                    # Get last 3 growth rates
                    recent_rates = growth_rates.iloc[i-2:i+1]
                    
                    # Check if there's a significant trend reversal
                    # Previous trend was positive, now negative
                    if recent_rates.iloc[0] > 10 and recent_rates.iloc[1] > 10 and recent_rates.iloc[2] < -10:
                        reversal_type = "positive_to_negative"
                        trend_change = recent_rates.iloc[2] - recent_rates.iloc[:2].mean()
                    # Previous trend was negative, now positive
                    elif recent_rates.iloc[0] < -10 and recent_rates.iloc[1] < -10 and recent_rates.iloc[2] > 10:
                        reversal_type = "negative_to_positive"
                        trend_change = recent_rates.iloc[2] - recent_rates.iloc[:2].mean()
                    else:
                        continue
                    
                    current_value = group_df.loc[idx, indicator]
                    previous_idx = group_df.index[pos - 1]
                    previous_value = group_df.loc[previous_idx, indicator]
                    
                    deviation = current_value - previous_value
                    growth_rate = recent_rates.iloc[2]
                    
                    # Calculate severity based on magnitude of reversal
                    severity_score = self._calculate_reversal_severity(abs(trend_change))
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    description = f"{indicator} trend reversed from {reversal_type.replace('_', ' ')}: current change {growth_rate:.1f}%"
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=territory_id if territory_id is not None else idx,
                        municipal_name=group_df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in group_df.columns else 'Unknown',
                        region_name=group_df.loc[idx, 'region_name'] if 'region_name' in group_df.columns else 'Unknown',
                        indicator=indicator,
                        anomaly_type='temporal_anomaly',
                        actual_value=float(current_value),
                        expected_value=float(previous_value),
                        deviation=float(deviation),
                        deviation_pct=float(growth_rate),
                        severity_score=severity_score,
                        z_score=None,
                        data_source=data_source,
                        detection_method='trend_reversal',
                        description=description,
                        potential_explanation=f"Significant trend reversal in {indicator} - may indicate structural change or data quality issue"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_high_volatility(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect municipalities with high volatility in indicators.
        
        Identifies territories where the standard deviation of changes exceeds
        2 times the median volatility across all territories.
        
        Args:
            df: DataFrame containing temporal municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo', 'year', 'month', 'quarter']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by territory if available
        if 'territory_id' not in df.columns:
            self.logger.debug("No territory_id column - cannot detect high volatility")
            return anomalies
        
        grouped = df.groupby('territory_id')
        
        for indicator in indicator_cols:
            # Calculate volatility (std of growth rates) for each territory
            volatilities = {}
            growth_rate_data = {}
            
            for territory_id, group_df in grouped:
                # Sort by temporal column if available
                group_df = self._sort_by_temporal_column(group_df)
                
                if len(group_df) < 3:
                    continue
                
                # Skip if column has too many missing values
                if group_df[indicator].isna().sum() / len(group_df) > 0.5:
                    continue
                
                # Calculate growth rates
                growth_rates = self._calculate_growth_rates(group_df[indicator])
                
                if len(growth_rates) < 2:
                    continue
                
                # Calculate volatility as standard deviation of growth rates
                volatility = growth_rates.std()
                
                if not np.isnan(volatility) and volatility > 0:
                    volatilities[territory_id] = volatility
                    growth_rate_data[territory_id] = growth_rates
            
            if len(volatilities) < 3:
                continue
            
            # Calculate median volatility across all territories
            median_volatility = np.median(list(volatilities.values()))
            
            if median_volatility == 0:
                continue
            
            # Find territories with high volatility
            threshold = median_volatility * self.volatility_multiplier
            
            for territory_id, volatility in volatilities.items():
                if volatility > threshold:
                    # Get the territory data
                    territory_df = df[df['territory_id'] == territory_id]
                    
                    # Get representative values
                    mean_value = territory_df[indicator].mean()
                    max_value = territory_df[indicator].max()
                    min_value = territory_df[indicator].min()
                    
                    deviation = volatility - median_volatility
                    deviation_pct = ((volatility - median_volatility) / median_volatility * 100)
                    
                    # Calculate severity based on how much volatility exceeds threshold
                    severity_score = self._calculate_volatility_severity(volatility / median_volatility)
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    description = f"{indicator} shows high volatility (std={volatility:.2f}) compared to median ({median_volatility:.2f}). Range: {min_value:.2f} to {max_value:.2f}"
                    
                    # Use the first row of territory data for location info
                    first_row = territory_df.iloc[0]
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=territory_id,
                        municipal_name=first_row['municipal_district_name_short'] if 'municipal_district_name_short' in territory_df.columns else 'Unknown',
                        region_name=first_row['region_name'] if 'region_name' in territory_df.columns else 'Unknown',
                        indicator=indicator,
                        anomaly_type='temporal_anomaly',
                        actual_value=float(volatility),
                        expected_value=float(median_volatility),
                        deviation=float(deviation),
                        deviation_pct=float(deviation_pct),
                        severity_score=severity_score,
                        z_score=None,
                        data_source=data_source,
                        detection_method='high_volatility',
                        description=description,
                        potential_explanation=f"Excessive variation in {indicator} over time - may indicate unstable measurements or genuine high variability"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _sort_by_temporal_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort DataFrame by temporal column if available.
        
        Args:
            df: DataFrame to sort
            
        Returns:
            Sorted DataFrame
        """
        temporal_columns = ['date', 'timestamp', 'period', 'year', 'month', 'quarter']
        
        for col in temporal_columns:
            if col in df.columns:
                return df.sort_values(by=col).reset_index(drop=True)
        
        # If no temporal column found, return as is
        return df
    
    def _calculate_growth_rates(self, series: pd.Series) -> pd.Series:
        """
        Calculate period-over-period growth rates.
        
        Args:
            series: Series of values over time
            
        Returns:
            Series of growth rates (percentage change)
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 2:
            return pd.Series(dtype=float)
        
        # Calculate percentage change
        growth_rates = clean_series.pct_change() * 100
        
        # Remove infinite values (caused by division by zero)
        growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan).dropna()
        
        return growth_rates
    
    def _calculate_spike_severity(self, abs_growth_rate: float) -> float:
        """
        Calculate severity score based on spike magnitude.
        
        Args:
            abs_growth_rate: Absolute growth rate percentage
            
        Returns:
            Severity score between 0 and 100
        """
        if abs_growth_rate >= 500:
            return 100.0
        elif abs_growth_rate >= 300:
            return 90.0
        elif abs_growth_rate >= 200:
            return 80.0
        elif abs_growth_rate >= 150:
            return 70.0
        elif abs_growth_rate >= 100:
            return 60.0
        else:
            # Linear scaling for changes below 100%
            return min(abs_growth_rate * 0.6, 60.0)
    
    def _calculate_reversal_severity(self, abs_trend_change: float) -> float:
        """
        Calculate severity score based on trend reversal magnitude.
        
        Args:
            abs_trend_change: Absolute change in trend
            
        Returns:
            Severity score between 0 and 100
        """
        if abs_trend_change >= 150:
            return 90.0
        elif abs_trend_change >= 100:
            return 75.0
        elif abs_trend_change >= 75:
            return 60.0
        elif abs_trend_change >= 50:
            return 50.0
        else:
            return min(abs_trend_change * 1.0, 50.0)
    
    def _calculate_volatility_severity(self, volatility_ratio: float) -> float:
        """
        Calculate severity score based on volatility ratio.
        
        Args:
            volatility_ratio: Ratio of actual volatility to median volatility
            
        Returns:
            Severity score between 0 and 100
        """
        if volatility_ratio >= 10:
            return 100.0
        elif volatility_ratio >= 7:
            return 85.0
        elif volatility_ratio >= 5:
            return 70.0
        elif volatility_ratio >= 3:
            return 55.0
        elif volatility_ratio >= 2:
            return 40.0
        else:
            return min(volatility_ratio * 20, 40.0)
    
    def detect_seasonal_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect seasonal anomalies by comparing values to historical patterns.
        
        Identifies values that deviate significantly from the expected seasonal
        pattern for the same period in previous years.
        
        Args:
            df: DataFrame containing temporal municipal data with seasonal information
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Check if seasonal data is available (need month or quarter information)
        if 'month' not in df.columns and 'quarter' not in df.columns:
            self.logger.debug("No seasonal information (month/quarter) available for seasonal anomaly detection")
            return anomalies
        
        # Determine seasonal column
        seasonal_col = 'month' if 'month' in df.columns else 'quarter'
        
        # Need year information for seasonal comparison
        if 'year' not in df.columns:
            self.logger.debug("No year information available for seasonal anomaly detection")
            return anomalies
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID and temporal columns
        exclude_cols = ['territory_id', 'oktmo', 'year', 'month', 'quarter']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by territory if available
        if 'territory_id' not in df.columns:
            self.logger.debug("No territory_id column - cannot detect seasonal anomalies")
            return anomalies
        
        grouped = df.groupby('territory_id')
        
        for territory_id, group_df in grouped:
            # Need at least 2 years of data for seasonal comparison
            if 'year' in group_df.columns:
                unique_years = group_df['year'].nunique()
                if unique_years < 2:
                    continue
            else:
                continue
            
            for indicator in indicator_cols:
                # Skip if column has too many missing values
                if group_df[indicator].isna().sum() / len(group_df) > 0.5:
                    continue
                
                # Calculate seasonal patterns (average for each season across years)
                seasonal_patterns = group_df.groupby(seasonal_col)[indicator].agg(['mean', 'std']).to_dict('index')
                
                # Check each observation against its seasonal pattern
                for idx, row in group_df.iterrows():
                    season = row[seasonal_col]
                    actual_value = row[indicator]
                    
                    if pd.isna(actual_value):
                        continue
                    
                    if season not in seasonal_patterns:
                        continue
                    
                    seasonal_mean = seasonal_patterns[season]['mean']
                    seasonal_std = seasonal_patterns[season]['std']
                    
                    # Skip if no variation in seasonal pattern
                    if pd.isna(seasonal_std) or seasonal_std == 0:
                        continue
                    
                    # Calculate z-score relative to seasonal pattern
                    seasonal_z_score = (actual_value - seasonal_mean) / seasonal_std
                    
                    # Flag if deviation exceeds 2 standard deviations
                    if abs(seasonal_z_score) > 2.0:
                        deviation = actual_value - seasonal_mean
                        deviation_pct = (deviation / seasonal_mean * 100) if seasonal_mean != 0 else 0
                        
                        # Calculate severity based on seasonal z-score
                        severity_score = self.calculate_severity_score(
                            deviation=deviation,
                            z_score=seasonal_z_score
                        )
                        
                        # Determine data source from indicator name
                        data_source = self._determine_data_source(indicator)
                        
                        # Create description
                        season_name = self._get_season_name(season, seasonal_col)
                        year = row['year'] if 'year' in row else 'Unknown'
                        
                        if seasonal_z_score > 0:
                            direction = "above"
                        else:
                            direction = "below"
                        
                        description = f"{indicator} in {season_name} {year} is {abs(seasonal_z_score):.2f} std deviations {direction} seasonal average ({seasonal_mean:.2f})"
                        
                        anomaly = self.create_anomaly_record(
                            territory_id=territory_id,
                            municipal_name=row['municipal_district_name_short'] if 'municipal_district_name_short' in group_df.columns else 'Unknown',
                            region_name=row['region_name'] if 'region_name' in group_df.columns else 'Unknown',
                            indicator=indicator,
                            anomaly_type='temporal_anomaly',
                            actual_value=float(actual_value),
                            expected_value=float(seasonal_mean),
                            deviation=float(deviation),
                            deviation_pct=float(deviation_pct),
                            severity_score=severity_score,
                            z_score=float(seasonal_z_score),
                            data_source=data_source,
                            detection_method='seasonal_anomaly',
                            description=description,
                            potential_explanation=f"Value deviates from typical seasonal pattern for {season_name} - may indicate unusual event or data quality issue"
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _get_season_name(self, season_value: Any, seasonal_col: str) -> str:
        """
        Get human-readable season name.
        
        Args:
            season_value: Value of the season (month number or quarter number)
            seasonal_col: Name of the seasonal column ('month' or 'quarter')
            
        Returns:
            Human-readable season name
        """
        if seasonal_col == 'month':
            month_names = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }
            return month_names.get(season_value, f'Month {season_value}')
        elif seasonal_col == 'quarter':
            return f'Q{season_value}'
        else:
            return str(season_value)



class GeographicAnomalyDetector(BaseAnomalyDetector):
    """
    Detector for geographic anomalies in municipal data.
    
    Identifies:
    1. Regional outliers (municipalities differing from regional average)
    2. Cluster outliers (municipalities differing from neighbors)
    3. Urban vs rural anomalies (separate analysis for different municipality types)
    4. Deviations from regional means for each indicator
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the geographic anomaly detector.
        
        Args:
            config: Configuration dictionary containing geographic thresholds
        """
        super().__init__(config)
        self.regional_z_score_threshold = config.get('thresholds', {}).get('geographic', {}).get('regional_z_score', 2.0)
        self.cluster_threshold = config.get('thresholds', {}).get('geographic', {}).get('cluster_threshold', 2.5)
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect geographic anomalies in the dataset.
        
        Args:
            df: DataFrame containing municipal data with geographic information
            
        Returns:
            DataFrame with detected anomalies
        """
        self.logger.info("Starting geographic anomaly detection")
        
        all_anomalies = []
        
        # Check if geographic data is available
        if not self._has_geographic_data(df):
            self.logger.warning("No geographic data available for geographic anomaly detection")
            return pd.DataFrame()
        
        # Detect regional outliers
        regional_anomalies = self.detect_regional_outliers(df)
        all_anomalies.extend(regional_anomalies)
        self.logger.info(f"Detected {len(regional_anomalies)} regional outliers")
        
        # Detect cluster outliers
        cluster_anomalies = self.detect_cluster_outliers(df)
        all_anomalies.extend(cluster_anomalies)
        self.logger.info(f"Detected {len(cluster_anomalies)} cluster outliers")
        
        # Detect urban vs rural anomalies
        urban_rural_anomalies = self.detect_urban_rural_anomalies(df)
        all_anomalies.extend(urban_rural_anomalies)
        self.logger.info(f"Detected {len(urban_rural_anomalies)} urban/rural anomalies")
        
        if not all_anomalies:
            self.logger.info("No geographic anomalies detected")
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        anomalies_df = pd.DataFrame(all_anomalies)
        
        # Remove duplicate anomalies (same territory_id and indicator)
        anomalies_df = anomalies_df.drop_duplicates(
            subset=['territory_id', 'indicator', 'detection_method'],
            keep='first'
        )
        
        self.logger.info(f"Total unique geographic anomalies detected: {len(anomalies_df)}")
        
        return anomalies_df
    
    def _has_geographic_data(self, df: pd.DataFrame) -> bool:
        """
        Check if the dataset contains geographic information.
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if geographic data is available, False otherwise
        """
        # Check for region information
        if 'region_name' not in df.columns:
            self.logger.warning("No region_name column found")
            return False
        
        # Check if we have territory identifiers
        if 'territory_id' not in df.columns:
            self.logger.warning("No territory_id column found")
            return False
        
        return True
    
    def detect_regional_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect municipalities that differ significantly from their regional average.
        
        Uses robust statistics (median/MAD) to identify municipalities where indicators deviate
        more than the threshold from the regional median. Robust z-score is calculated as:
        (value - median) / (1.4826 * MAD)
        
        Now implements type-aware comparison: groups by region AND municipality_type,
        and applies type-specific thresholds (capital: 3.5, urban: 2.5, rural: 2.0).
        
        Args:
            df: DataFrame containing municipal data with region information
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Check if region_name column exists
        if 'region_name' not in df.columns:
            self.logger.warning("No region_name column - cannot detect regional outliers")
            return anomalies
        
        # Check if municipality_type column exists (should be added by DataPreprocessor)
        if 'municipality_type' not in df.columns:
            self.logger.warning("No municipality_type column - falling back to region-only grouping")
            # Fall back to old behavior if municipality_type is not available
            return self._detect_regional_outliers_legacy(df)
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by region AND municipality_type for type-aware comparison
        grouped = df.groupby(['region_name', 'municipality_type'])
        
        for indicator in indicator_cols:
            # Skip if column has too many missing values
            if df[indicator].isna().sum() / len(df) > 0.5:
                continue
            
            # Calculate robust statistics for each (region, municipality_type) group
            # Using median and MAD instead of mean and std
            group_stats = {}
            for (region, muni_type), group_df in grouped:
                clean_values = group_df[indicator].dropna()
                if len(clean_values) == 0:
                    group_stats[(region, muni_type)] = {'median': np.nan, 'mad': np.nan, 'count': 0}
                    continue
                median_val = clean_values.median()
                mad_val = np.median(np.abs(clean_values - median_val))
                group_stats[(region, muni_type)] = {
                    'median': median_val,
                    'mad': mad_val,
                    'count': len(clean_values)
                }
            
            # Check each municipality against its group statistics
            for idx, row in df.iterrows():
                region = row['region_name']
                muni_type = row['municipality_type']
                actual_value = row[indicator]
                
                if pd.isna(actual_value):
                    continue
                
                group_key = (region, muni_type)
                if group_key not in group_stats:
                    continue
                
                group_median = group_stats[group_key]['median']
                group_mad = group_stats[group_key]['mad']
                group_count = group_stats[group_key]['count']
                
                # Skip if insufficient data in group or no variation
                if group_count < 3 or pd.isna(group_mad) or group_mad == 0:
                    continue
                
                # Calculate robust z-score: (value - median) / (1.4826 * MAD)
                # The factor 1.4826 makes MAD comparable to standard deviation for normal distributions
                robust_z_score = (actual_value - group_median) / (1.4826 * group_mad)
                
                # Get type-specific threshold
                type_threshold = self._get_threshold_for_type(muni_type)
                
                # Flag if deviation exceeds type-specific threshold
                if abs(robust_z_score) > type_threshold:
                    deviation = actual_value - group_median
                    deviation_pct = (deviation / group_median * 100) if group_median != 0 else 0
                    
                    # Calculate severity based on robust z-score
                    # Apply adjustment for same-type comparison
                    base_severity = self.calculate_severity_score(
                        deviation=deviation,
                        z_score=robust_z_score
                    )
                    
                    # Increase severity for same-type outliers (high confidence)
                    # These are municipalities that differ from their peers of the same type
                    # This indicates a real anomaly, not just natural urban-rural differences
                    # Requirement 3.5: Verify anomaly is not due to natural urban-rural differences
                    # Requirement 5.1: Apply type-based weighting
                    severity_score = min(base_severity * 1.2, 100.0)
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    # Create description
                    if robust_z_score > 0:
                        direction = "above"
                    else:
                        direction = "below"
                    
                    description = f"{indicator} value {actual_value:.2f} is {abs(robust_z_score):.2f} robust std deviations {direction} {muni_type} municipality median {group_median:.2f} in {region}"
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=row['territory_id'] if 'territory_id' in df.columns else idx,
                        municipal_name=row['municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                        region_name=region,
                        indicator=indicator,
                        anomaly_type='geographic_anomaly',
                        actual_value=float(actual_value),
                        expected_value=float(group_median),
                        deviation=float(deviation),
                        deviation_pct=float(deviation_pct),
                        severity_score=severity_score,
                        z_score=float(robust_z_score),
                        data_source=data_source,
                        detection_method='regional_outlier',
                        description=description,
                        potential_explanation=f"Municipality significantly differs from other {muni_type} municipalities in the region - may indicate unique local conditions or data quality issue"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _get_threshold_for_type(self, muni_type: str) -> float:
        """
        Get type-specific threshold for geographic anomaly detection.
        
        Capital cities and major urban centers naturally have different characteristics,
        so we apply more relaxed thresholds to reduce false positives.
        
        Args:
            muni_type: Municipality type ('capital', 'urban', 'rural', or 'unknown')
            
        Returns:
            Z-score threshold for the municipality type
        """
        thresholds = {
            'capital': 3.5,  # Very relaxed - capitals are naturally different
            'urban': 2.5,    # Relaxed - urban areas have more variation
            'rural': 2.0,    # Normal - use base threshold for rural areas
            'unknown': 2.0   # Default to normal threshold
        }
        
        threshold = thresholds.get(muni_type, 2.0)
        
        self.logger.debug(f"Using threshold {threshold} for municipality type '{muni_type}'")
        
        return threshold
    
    def _detect_regional_outliers_legacy(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Legacy method for detecting regional outliers without municipality type classification.
        
        This is used as a fallback when municipality_type column is not available.
        Groups only by region, not by municipality type.
        
        Args:
            df: DataFrame containing municipal data with region information
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by region only
        grouped = df.groupby('region_name')
        
        for indicator in indicator_cols:
            # Skip if column has too many missing values
            if df[indicator].isna().sum() / len(df) > 0.5:
                continue
            
            # Calculate regional statistics for this indicator
            regional_stats = grouped[indicator].agg(['mean', 'std', 'count']).to_dict('index')
            
            # Check each municipality against its regional statistics
            for idx, row in df.iterrows():
                region = row['region_name']
                actual_value = row[indicator]
                
                if pd.isna(actual_value):
                    continue
                
                if region not in regional_stats:
                    continue
                
                regional_mean = regional_stats[region]['mean']
                regional_std = regional_stats[region]['std']
                regional_count = regional_stats[region]['count']
                
                # Skip if insufficient data in region or no variation
                if regional_count < 3 or pd.isna(regional_std) or regional_std == 0:
                    continue
                
                # Calculate regional z-score
                regional_z_score = (actual_value - regional_mean) / regional_std
                
                # Flag if deviation exceeds threshold
                if abs(regional_z_score) > self.regional_z_score_threshold:
                    deviation = actual_value - regional_mean
                    deviation_pct = (deviation / regional_mean * 100) if regional_mean != 0 else 0
                    
                    # Calculate severity based on regional z-score
                    # Apply reduction for mixed-type comparison (legacy method)
                    base_severity = self.calculate_severity_score(
                        deviation=deviation,
                        z_score=regional_z_score
                    )
                    
                    # Reduce severity for natural differences (mixed urban/rural comparison)
                    # Legacy method doesn't separate by type, so differences may be natural
                    # Requirement 3.5: Reduce base severity for natural differences
                    severity_score = base_severity * 0.7
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    # Create description
                    if regional_z_score > 0:
                        direction = "above"
                    else:
                        direction = "below"
                    
                    description = f"{indicator} value {actual_value:.2f} is {abs(regional_z_score):.2f} std deviations {direction} regional mean {regional_mean:.2f} in {region}"
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=row['territory_id'] if 'territory_id' in df.columns else idx,
                        municipal_name=row['municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                        region_name=region,
                        indicator=indicator,
                        anomaly_type='geographic_anomaly',
                        actual_value=float(actual_value),
                        expected_value=float(regional_mean),
                        deviation=float(deviation),
                        deviation_pct=float(deviation_pct),
                        severity_score=severity_score,
                        z_score=float(regional_z_score),
                        data_source=data_source,
                        detection_method='regional_outlier',
                        description=description,
                        potential_explanation=f"Municipality significantly differs from regional average - may indicate unique local conditions or data quality issue"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_cluster_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect municipalities that differ from their neighbors.
        
        Identifies municipalities within a region that form clusters of similar
        values, then finds outliers that don't fit any cluster pattern.
        
        Uses robust statistics (median/MAD) instead of mean/std to be resistant
        to outliers when defining cluster boundaries.
        
        Args:
            df: DataFrame containing municipal data with region information
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Check if region_name column exists
        if 'region_name' not in df.columns:
            self.logger.warning("No region_name column - cannot detect cluster outliers")
            return anomalies
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Group by region
        grouped = df.groupby('region_name')
        
        for region_name, region_df in grouped:
            # Need at least 5 municipalities in a region for cluster analysis
            if len(region_df) < 5:
                continue
            
            for indicator in indicator_cols:
                # Skip if column has too many missing values in this region
                if region_df[indicator].isna().sum() / len(region_df) > 0.5:
                    continue
                
                # Get valid values for this indicator in the region
                valid_mask = region_df[indicator].notna()
                valid_df = region_df[valid_mask].copy()
                
                if len(valid_df) < 5:
                    continue
                
                values = valid_df[indicator]
                
                # Calculate robust statistics using median and MAD
                median_val = values.median()
                mad_val = np.median(np.abs(values - median_val))
                
                if mad_val == 0:  # No variation
                    continue
                
                # Define cluster boundaries using MAD-based method
                # Similar to IQR method but using MAD for robustness
                # MAD * 1.4826 approximates standard deviation for normal distributions
                # Using cluster_threshold as multiplier (similar to IQR multiplier)
                lower_cluster_bound = median_val - self.cluster_threshold * 1.4826 * mad_val
                upper_cluster_bound = median_val + self.cluster_threshold * 1.4826 * mad_val
                
                # Find outliers outside cluster bounds
                outlier_mask = (values < lower_cluster_bound) | (values > upper_cluster_bound)
                outlier_indices = values[outlier_mask].index
                
                for idx in outlier_indices:
                    actual_value = region_df.loc[idx, indicator]
                    
                    # Use median as expected value (center of cluster)
                    expected_value = median_val
                    deviation = actual_value - expected_value
                    deviation_pct = (deviation / expected_value * 100) if expected_value != 0 else 0
                    
                    # Calculate robust z-score: (value - median) / (1.4826 * MAD)
                    robust_z_score = (actual_value - median_val) / (1.4826 * mad_val)
                    
                    # Calculate how far outside the cluster bounds
                    if actual_value < lower_cluster_bound:
                        distance_from_cluster = lower_cluster_bound - actual_value
                        cluster_position = "below"
                    else:
                        distance_from_cluster = actual_value - upper_cluster_bound
                        cluster_position = "above"
                    
                    # Calculate severity based on robust z-score
                    base_severity = self.calculate_severity_score(
                        deviation=deviation,
                        z_score=robust_z_score
                    )
                    
                    # Increase severity for cluster outliers (high confidence)
                    # Municipalities that differ from their immediate neighbors are likely real anomalies
                    # Requirement 5.1: Apply type-based weighting for geographic anomalies
                    severity_score = min(base_severity * 1.15, 100.0)
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    description = f"{indicator} value {actual_value:.2f} is {cluster_position} cluster bounds [{lower_cluster_bound:.2f}, {upper_cluster_bound:.2f}] (median: {median_val:.2f}) in {region_name}"
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=region_df.loc[idx, 'territory_id'] if 'territory_id' in region_df.columns else idx,
                        municipal_name=region_df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in region_df.columns else 'Unknown',
                        region_name=region_name,
                        indicator=indicator,
                        anomaly_type='geographic_anomaly',
                        actual_value=float(actual_value),
                        expected_value=float(expected_value),
                        deviation=float(deviation),
                        deviation_pct=float(deviation_pct),
                        severity_score=severity_score,
                        z_score=float(robust_z_score),
                        data_source=data_source,
                        detection_method='cluster_outlier',
                        description=description,
                        potential_explanation=f"Municipality differs significantly from neighboring municipalities in the region - may indicate unique characteristics or measurement error"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_urban_rural_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies with separate analysis for urban vs rural municipalities.
        
        Compares municipalities to others of the same type (urban or rural) to
        account for structural differences between municipality types.
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Try to identify urban vs rural municipalities
        # This can be done through various heuristics:
        # 1. Explicit column indicating type
        # 2. Population size
        # 3. Municipality name patterns
        
        df_with_type = self._classify_urban_rural(df)
        
        if 'municipality_type' not in df_with_type.columns:
            self.logger.warning("Could not classify municipalities as urban/rural")
            return anomalies
        
        # Get numeric columns (indicators)
        numeric_cols = df_with_type.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Analyze each municipality type separately
        for muni_type in df_with_type['municipality_type'].unique():
            type_df = df_with_type[df_with_type['municipality_type'] == muni_type]
            
            # Need sufficient data for each type
            if len(type_df) < 5:
                continue
            
            for indicator in indicator_cols:
                # Skip if column has too many missing values
                if type_df[indicator].isna().sum() / len(type_df) > 0.5:
                    continue
                
                values = type_df[indicator].dropna()
                
                if len(values) < 5:
                    continue
                
                # Calculate statistics for this municipality type
                mean_val = values.mean()
                std_val = values.std()
                
                if std_val == 0:  # No variation
                    continue
                
                # Calculate z-scores within municipality type
                z_scores = (values - mean_val) / std_val
                
                # Find outliers
                outlier_mask = abs(z_scores) > self.regional_z_score_threshold
                outlier_indices = values[outlier_mask].index
                
                for idx in outlier_indices:
                    actual_value = df_with_type.loc[idx, indicator]
                    z_score = z_scores.loc[idx]
                    deviation = actual_value - mean_val
                    deviation_pct = (deviation / mean_val * 100) if mean_val != 0 else 0
                    
                    # Calculate severity based on z-score
                    base_severity = self.calculate_severity_score(
                        deviation=deviation,
                        z_score=z_score
                    )
                    
                    # Increase severity for same-type outliers (high confidence)
                    # Comparing within same municipality type eliminates natural differences
                    # Requirement 3.5: Verify anomaly is not due to natural urban-rural differences
                    # Requirement 5.1: Apply type-based weighting
                    severity_score = min(base_severity * 1.2, 100.0)
                    
                    # Determine data source from indicator name using explicit mapping
                    data_source = self.get_data_source(indicator)
                    
                    # Create description
                    if z_score > 0:
                        direction = "above"
                    else:
                        direction = "below"
                    
                    description = f"{indicator} value {actual_value:.2f} is {abs(z_score):.2f} std deviations {direction} {muni_type} municipality average {mean_val:.2f}"
                    
                    anomaly = self.create_anomaly_record(
                        territory_id=df_with_type.loc[idx, 'territory_id'] if 'territory_id' in df_with_type.columns else idx,
                        municipal_name=df_with_type.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df_with_type.columns else 'Unknown',
                        region_name=df_with_type.loc[idx, 'region_name'] if 'region_name' in df_with_type.columns else 'Unknown',
                        indicator=indicator,
                        anomaly_type='geographic_anomaly',
                        actual_value=float(actual_value),
                        expected_value=float(mean_val),
                        deviation=float(deviation),
                        deviation_pct=float(deviation_pct),
                        severity_score=severity_score,
                        z_score=float(z_score),
                        data_source=data_source,
                        detection_method=f'{muni_type}_outlier',
                        description=description,
                        potential_explanation=f"Municipality differs significantly from other {muni_type} municipalities - may indicate unique local factors"
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_urban_rural(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify municipalities as urban or rural.
        
        Uses multiple heuristics to determine municipality type:
        1. Explicit type column if available
        2. Population size (urban typically > 50,000)
        3. Name patterns (город, городской округ = urban; район, сельский = rural)
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            DataFrame with added 'municipality_type' column
        """
        df_copy = df.copy()
        
        # Check if there's already a type column
        type_columns = ['municipality_type', 'type', 'urban_rural', 'settlement_type']
        for col in type_columns:
            if col in df_copy.columns:
                df_copy['municipality_type'] = df_copy[col]
                return df_copy
        
        # Initialize type column
        df_copy['municipality_type'] = 'unknown'
        
        # Method 1: Use population if available
        if 'population_total' in df_copy.columns:
            # Urban threshold: 50,000 population
            urban_threshold = 50000
            df_copy.loc[df_copy['population_total'] >= urban_threshold, 'municipality_type'] = 'urban'
            df_copy.loc[df_copy['population_total'] < urban_threshold, 'municipality_type'] = 'rural'
        
        # Method 2: Use name patterns if name column is available
        if 'municipal_district_name_short' in df_copy.columns:
            name_col = 'municipal_district_name_short'
            
            # Urban patterns (Russian)
            urban_patterns = ['город', 'городской округ', 'г.о.', 'г. ', 'городск']
            
            # Rural patterns (Russian)
            rural_patterns = ['район', 'сельск', 'муниципальный район', 'м.р.']
            
            for idx, row in df_copy.iterrows():
                if pd.notna(row[name_col]):
                    name_lower = str(row[name_col]).lower()
                    
                    # Check urban patterns
                    for pattern in urban_patterns:
                        if pattern in name_lower:
                            df_copy.loc[idx, 'municipality_type'] = 'urban'
                            break
                    
                    # Check rural patterns (only if not already classified as urban)
                    if df_copy.loc[idx, 'municipality_type'] != 'urban':
                        for pattern in rural_patterns:
                            if pattern in name_lower:
                                df_copy.loc[idx, 'municipality_type'] = 'rural'
                                break
        
        # Log classification results
        type_counts = df_copy['municipality_type'].value_counts()
        self.logger.info(f"Municipality classification: {type_counts.to_dict()}")
        
        # If we couldn't classify any municipalities, remove the column
        if (df_copy['municipality_type'] == 'unknown').all():
            df_copy = df_copy.drop('municipality_type', axis=1)
            self.logger.warning("Could not classify any municipalities as urban/rural")
        
        return df_copy
    
    def _calculate_cluster_severity(self, normalized_distance: float) -> float:
        """
        Calculate severity score based on distance from cluster.
        
        Args:
            normalized_distance: Distance from cluster normalized by IQR
            
        Returns:
            Severity score between 0 and 100
        """
        if normalized_distance >= 5:
            return 100.0
        elif normalized_distance >= 4:
            return 85.0
        elif normalized_distance >= 3:
            return 70.0
        elif normalized_distance >= 2.5:
            return 60.0
        elif normalized_distance >= 2:
            return 50.0
        else:
            return min(normalized_distance * 25, 50.0)


class LogicalConsistencyChecker(BaseAnomalyDetector):
    """
    Detector for logical inconsistencies and data quality issues.
    
    Identifies:
    1. Negative values where only positive values are logically possible
    2. Impossible ratios (e.g., consumption exceeding capacity)
    3. Contradictory indicators (e.g., high consumption with low connection rates)
    4. Duplicate or inconsistent municipality identifiers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the logical consistency checker.
        
        Args:
            config: Configuration dictionary containing logical check settings
        """
        super().__init__(config)
        self.check_negative_values = config.get('thresholds', {}).get('logical', {}).get('check_negative_values', True)
        self.check_impossible_ratios = config.get('thresholds', {}).get('logical', {}).get('check_impossible_ratios', True)
        
        # Get threshold for flagging municipalities with high missing indicators
        self.high_missing_municipality_threshold = config.get('missing_value_handling', {}).get('municipality_threshold', 70.0)
        
        # Define indicators that must be positive
        self.positive_only_indicators = self._define_positive_only_indicators()
        
        # Define impossible ratio checks
        self.ratio_checks = self._define_ratio_checks()
        
        # Define contradictory indicator pairs
        self.contradictory_pairs = self._define_contradictory_pairs()
    
    def _define_positive_only_indicators(self) -> List[str]:
        """
        Define indicators that must have positive values.
        
        Returns:
            List of indicator name patterns that should only be positive
        """
        return [
            'population',
            'consumption',
            'salary',
            'market_access',
            'connection'
        ]
    
    def _define_ratio_checks(self) -> List[Dict[str, Any]]:
        """
        Define ratio checks for logically impossible relationships.
        
        Returns:
            List of ratio check definitions with numerator, denominator, and max ratio
        """
        return [
            {
                'name': 'consumption_to_population',
                'numerator_pattern': 'consumption',
                'denominator': 'population_total',
                'max_ratio': 1000000,  # Maximum reasonable consumption per capita
                'description': 'consumption per capita'
            },
            {
                'name': 'salary_to_consumption',
                'numerator': 'consumption_total',
                'denominator_pattern': 'salary',
                'max_ratio': 100,  # Consumption shouldn't be 100x salary
                'description': 'consumption to salary ratio'
            }
        ]
    
    def _define_contradictory_pairs(self) -> List[Dict[str, Any]]:
        """
        Define pairs of indicators that should be correlated.
        
        Returns:
            List of contradictory indicator pair definitions
        """
        return [
            {
                'name': 'high_consumption_low_population',
                'high_indicator_pattern': 'consumption',
                'low_indicator': 'population_total',
                'description': 'high consumption with low population'
            },
            {
                'name': 'high_consumption_low_salary',
                'high_indicator_pattern': 'consumption',
                'low_indicator_pattern': 'salary',
                'description': 'high consumption with low salary'
            }
        ]
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect logical inconsistencies in the dataset.
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            DataFrame with detected anomalies
        """
        self.logger.info("Starting logical consistency checking")
        
        all_anomalies = []
        
        # Detect negative values where they shouldn't exist
        if self.check_negative_values:
            negative_anomalies = self.detect_negative_values(df)
            all_anomalies.extend(negative_anomalies)
            self.logger.info(f"Detected {len(negative_anomalies)} negative value anomalies")
        
        # Detect impossible ratios
        if self.check_impossible_ratios:
            ratio_anomalies = self.detect_impossible_ratios(df)
            all_anomalies.extend(ratio_anomalies)
            self.logger.info(f"Detected {len(ratio_anomalies)} impossible ratio anomalies")
        
        # Detect contradictory indicators
        contradictory_anomalies = self.detect_contradictory_indicators(df)
        all_anomalies.extend(contradictory_anomalies)
        self.logger.info(f"Detected {len(contradictory_anomalies)} contradictory indicator anomalies")
        
        # Detect unusual missing data patterns
        missing_data_anomalies = self.detect_unusual_missing_patterns(df)
        all_anomalies.extend(missing_data_anomalies)
        self.logger.info(f"Detected {len(missing_data_anomalies)} unusual missing data pattern anomalies")
        
        # Flag municipalities with >70% missing indicators
        high_missing_anomalies = self.flag_high_missing_municipalities(df)
        all_anomalies.extend(high_missing_anomalies)
        self.logger.info(f"Detected {len(high_missing_anomalies)} municipalities with high missing indicators")
        
        # Detect duplicate or inconsistent identifiers
        duplicate_anomalies = self.detect_duplicate_identifiers(df)
        all_anomalies.extend(duplicate_anomalies)
        self.logger.info(f"Detected {len(duplicate_anomalies)} duplicate identifier anomalies")
        
        if not all_anomalies:
            self.logger.info("No logical inconsistencies detected")
            return pd.DataFrame()
        
        # Convert to DataFrame and remove duplicates
        anomalies_df = pd.DataFrame(all_anomalies)
        
        # Remove duplicate anomalies (same territory_id and indicator)
        anomalies_df = anomalies_df.drop_duplicates(
            subset=['territory_id', 'indicator', 'detection_method'],
            keep='first'
        )
        
        self.logger.info(f"Total unique logical inconsistencies detected: {len(anomalies_df)}")
        
        return anomalies_df
    
    def detect_negative_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect negative values where only positive values are logically possible.
        
        Identifies indicators that should always be positive (like population,
        consumption, salary) but have negative values.
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to indicators that should be positive
        positive_indicators = []
        for col in numeric_cols:
            col_lower = col.lower()
            for pattern in self.positive_only_indicators:
                if pattern in col_lower:
                    positive_indicators.append(col)
                    break
        
        # Check each positive-only indicator for negative values
        for indicator in positive_indicators:
            # Find rows with negative values
            negative_mask = df[indicator] < 0
            negative_indices = df[negative_mask].index
            
            for idx in negative_indices:
                actual_value = df.loc[idx, indicator]
                
                # Expected value should be at least 0
                expected_value = 0.0
                deviation = actual_value - expected_value
                
                # Negative values in these indicators are always severe
                severity_score = 90.0
                
                # Determine data source from indicator name using explicit mapping
                data_source = self.get_data_source(indicator)
                
                description = f"{indicator} has negative value {actual_value:.2f}, which is logically impossible"
                
                anomaly = self.create_anomaly_record(
                    territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                    municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator=indicator,
                    anomaly_type='logical_inconsistency',
                    actual_value=float(actual_value),
                    expected_value=expected_value,
                    deviation=float(deviation),
                    deviation_pct=0.0,  # Percentage doesn't make sense for negative values
                    severity_score=severity_score,
                    z_score=None,
                    data_source=data_source,
                    detection_method='negative_value',
                    description=description,
                    potential_explanation="Data entry error or calculation error - this indicator cannot be negative"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_impossible_ratios(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect impossible ratios between related indicators.
        
        Identifies cases where the ratio between two related indicators exceeds
        logically possible bounds (e.g., consumption per capita exceeding reasonable limits).
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        for ratio_check in self.ratio_checks:
            # Find columns matching the patterns
            numerator_cols = self._find_matching_columns(
                df, 
                ratio_check.get('numerator_pattern'), 
                ratio_check.get('numerator')
            )
            
            denominator_cols = self._find_matching_columns(
                df,
                ratio_check.get('denominator_pattern'),
                ratio_check.get('denominator')
            )
            
            if not numerator_cols or not denominator_cols:
                continue
            
            # Check each combination of numerator and denominator
            for num_col in numerator_cols:
                for denom_col in denominator_cols:
                    # Skip if either column doesn't exist
                    if num_col not in df.columns or denom_col not in df.columns:
                        continue
                    
                    # Calculate ratios where both values are available and denominator is not zero
                    valid_mask = (
                        df[num_col].notna() & 
                        df[denom_col].notna() & 
                        (df[denom_col] != 0) &
                        (df[denom_col] > 0)  # Avoid negative denominators
                    )
                    
                    if not valid_mask.any():
                        continue
                    
                    valid_df = df[valid_mask].copy()
                    valid_df['ratio'] = valid_df[num_col] / valid_df[denom_col]
                    
                    # Find impossible ratios
                    max_ratio = ratio_check['max_ratio']
                    impossible_mask = valid_df['ratio'] > max_ratio
                    impossible_indices = valid_df[impossible_mask].index
                    
                    for idx in impossible_indices:
                        numerator_value = df.loc[idx, num_col]
                        denominator_value = df.loc[idx, denom_col]
                        ratio_value = valid_df.loc[idx, 'ratio']
                        
                        # Calculate how much the ratio exceeds the maximum
                        deviation = ratio_value - max_ratio
                        deviation_pct = (deviation / max_ratio * 100)
                        
                        # Calculate severity based on how much the ratio is exceeded
                        severity_score = self._calculate_ratio_severity(ratio_value / max_ratio)
                        
                        # Determine data source
                        data_source = 'cross_source'  # Ratios involve multiple sources
                        
                        description = f"Impossible ratio: {num_col}/{denom_col} = {ratio_value:.2f} exceeds maximum {max_ratio:.2f} ({ratio_check['description']})"
                        
                        anomaly = self.create_anomaly_record(
                            territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                            municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                            region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                            indicator=f"{num_col}/{denom_col}",
                            anomaly_type='logical_inconsistency',
                            actual_value=float(ratio_value),
                            expected_value=float(max_ratio),
                            deviation=float(deviation),
                            deviation_pct=float(deviation_pct),
                            severity_score=severity_score,
                            z_score=None,
                            data_source=data_source,
                            detection_method='impossible_ratio',
                            description=description,
                            potential_explanation=f"Ratio between {num_col} ({numerator_value:.2f}) and {denom_col} ({denominator_value:.2f}) is logically impossible - likely data quality issue"
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies
    
    def detect_contradictory_indicators(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect contradictory indicators that conflict with each other.
        
        Identifies cases where related indicators have values that contradict
        each other (e.g., high consumption with very low population or salary).
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        for pair in self.contradictory_pairs:
            # Find columns matching the patterns
            high_indicator_cols = self._find_matching_columns(
                df,
                pair.get('high_indicator_pattern'),
                pair.get('high_indicator')
            )
            
            low_indicator_cols = self._find_matching_columns(
                df,
                pair.get('low_indicator_pattern'),
                pair.get('low_indicator')
            )
            
            if not high_indicator_cols or not low_indicator_cols:
                continue
            
            # Check each combination
            for high_col in high_indicator_cols:
                for low_col in low_indicator_cols:
                    # Skip if either column doesn't exist
                    if high_col not in df.columns or low_col not in df.columns:
                        continue
                    
                    # Get valid data
                    valid_mask = df[high_col].notna() & df[low_col].notna()
                    valid_df = df[valid_mask].copy()
                    
                    if len(valid_df) < 10:  # Need sufficient data
                        continue
                    
                    # Calculate percentiles for both indicators
                    high_percentiles = valid_df[high_col].rank(pct=True) * 100
                    low_percentiles = valid_df[low_col].rank(pct=True) * 100
                    
                    # Find contradictions: high value in one indicator, low in another
                    # High consumption (>75th percentile) with low population/salary (<25th percentile)
                    contradiction_mask = (high_percentiles > 75) & (low_percentiles < 25)
                    contradiction_indices = valid_df[contradiction_mask].index
                    
                    for idx in contradiction_indices:
                        high_value = df.loc[idx, high_col]
                        low_value = df.loc[idx, low_col]
                        high_pct = high_percentiles.loc[idx]
                        low_pct = low_percentiles.loc[idx]
                        
                        # Calculate severity based on the contradiction magnitude
                        # More extreme percentiles = higher severity
                        percentile_gap = high_pct - low_pct
                        severity_score = self._calculate_contradiction_severity(percentile_gap)
                        
                        # Determine data source
                        data_source = 'cross_source'
                        
                        description = f"Contradictory indicators: {high_col} is high (percentile {high_pct:.1f}) but {low_col} is low (percentile {low_pct:.1f})"
                        
                        anomaly = self.create_anomaly_record(
                            territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                            municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                            region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                            indicator=f"{high_col} vs {low_col}",
                            anomaly_type='logical_inconsistency',
                            actual_value=float(high_value),
                            expected_value=float(low_value),
                            deviation=float(high_value - low_value),
                            deviation_pct=float(percentile_gap),
                            severity_score=severity_score,
                            z_score=None,
                            data_source=data_source,
                            detection_method='contradictory_indicators',
                            description=description,
                            potential_explanation=f"{pair['description']} - these indicators should typically be correlated, contradiction may indicate data quality issue"
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies
    
    def detect_unusual_missing_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect municipalities with unusual missing data patterns.
        
        Identifies municipalities where the pattern of missing data differs
        significantly from typical patterns (e.g., missing many more indicators
        than average, or missing specific combinations of indicators).
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(indicator_cols) == 0:
            return anomalies
        
        # Calculate missing data count for each municipality
        missing_counts = df[indicator_cols].isna().sum(axis=1)
        
        # Calculate missing data percentage for each municipality
        missing_percentages = (missing_counts / len(indicator_cols)) * 100
        
        # Calculate statistics for missing data across all municipalities
        mean_missing_pct = missing_percentages.mean()
        std_missing_pct = missing_percentages.std()
        
        if std_missing_pct == 0:  # All municipalities have same missing pattern
            return anomalies
        
        # Find municipalities with unusual missing patterns
        # Unusual = more than 2 standard deviations above mean
        threshold = mean_missing_pct + 2 * std_missing_pct
        
        unusual_mask = missing_percentages > threshold
        unusual_indices = missing_percentages[unusual_mask].index
        
        for idx in unusual_indices:
            missing_count = missing_counts.loc[idx]
            missing_pct = missing_percentages.loc[idx]
            
            # Calculate severity based on how much missing data exceeds typical
            z_score = (missing_pct - mean_missing_pct) / std_missing_pct if std_missing_pct > 0 else 0
            severity_score = self._calculate_missing_data_severity(missing_pct, z_score)
            
            # Get list of missing indicators for this municipality
            missing_indicators = [col for col in indicator_cols if pd.isna(df.loc[idx, col])]
            
            description = f"Municipality has {missing_count} missing indicators ({missing_pct:.1f}% missing) compared to average {mean_missing_pct:.1f}%"
            
            anomaly = self.create_anomaly_record(
                territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                indicator='missing_data_pattern',
                anomaly_type='logical_inconsistency',
                actual_value=float(missing_pct),
                expected_value=float(mean_missing_pct),
                deviation=float(missing_pct - mean_missing_pct),
                deviation_pct=float(((missing_pct - mean_missing_pct) / mean_missing_pct * 100) if mean_missing_pct > 0 else 0),
                severity_score=severity_score,
                z_score=float(z_score),
                data_source='metadata',
                detection_method='unusual_missing_pattern',
                description=description,
                potential_explanation=f"Unusual amount of missing data - {missing_count} indicators missing. This may indicate incomplete data collection or data quality issues. Missing indicators: {', '.join(missing_indicators[:5])}{'...' if len(missing_indicators) > 5 else ''}"
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def flag_high_missing_municipalities(self, df: pd.DataFrame, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Flag municipalities with >70% missing indicators.
        
        Identifies municipalities where more than the threshold percentage of indicators
        are missing, indicating severe data quality issues that require investigation.
        
        Args:
            df: DataFrame containing municipal data
            threshold: Percentage threshold for flagging (uses config default if None)
            
        Returns:
            List of anomaly records for municipalities with high missing indicators
        """
        if threshold is None:
            threshold = self.high_missing_municipality_threshold
        
        anomalies = []
        
        # Get numeric columns (indicators)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(indicator_cols) == 0:
            self.logger.warning("No indicator columns found for municipality flagging")
            return anomalies
        
        # Calculate missing data count for each municipality
        missing_counts = df[indicator_cols].isna().sum(axis=1)
        
        # Calculate missing data percentage for each municipality
        missing_percentages = (missing_counts / len(indicator_cols)) * 100
        
        # Find municipalities exceeding the threshold
        high_missing_mask = missing_percentages > threshold
        high_missing_indices = missing_percentages[high_missing_mask].index
        
        self.logger.info(f"Found {len(high_missing_indices)} municipalities with >{threshold}% missing indicators")
        
        for idx in high_missing_indices:
            missing_count = missing_counts.loc[idx]
            missing_pct = missing_percentages.loc[idx]
            
            # Calculate severity based on how much the threshold is exceeded
            # Higher missing percentage = higher severity
            if missing_pct >= 90:
                severity_score = 95.0
            elif missing_pct >= 80:
                severity_score = 85.0
            elif missing_pct >= 70:
                severity_score = 75.0
            else:
                severity_score = 65.0
            
            # Get list of missing indicators for this municipality
            missing_indicators = [col for col in indicator_cols if pd.isna(df.loc[idx, col])]
            
            # Get list of available indicators
            available_indicators = [col for col in indicator_cols if pd.notna(df.loc[idx, col])]
            
            description = f"Municipality flagged for data quality: {missing_count} of {len(indicator_cols)} indicators missing ({missing_pct:.1f}%)"
            
            # Create detailed explanation
            explanation_parts = [
                f"Severe data quality issue - {missing_pct:.1f}% of indicators are missing.",
                f"Only {len(available_indicators)} of {len(indicator_cols)} indicators have values.",
                "This municipality should be excluded from analysis or investigated for data collection issues."
            ]
            
            if len(missing_indicators) <= 10:
                explanation_parts.append(f"Missing indicators: {', '.join(missing_indicators)}")
            else:
                explanation_parts.append(f"Missing indicators include: {', '.join(missing_indicators[:10])}... and {len(missing_indicators) - 10} more")
            
            if len(available_indicators) > 0 and len(available_indicators) <= 5:
                explanation_parts.append(f"Available indicators: {', '.join(available_indicators)}")
            
            potential_explanation = " ".join(explanation_parts)
            
            anomaly = self.create_anomaly_record(
                territory_id=df.loc[idx, 'territory_id'] if 'territory_id' in df.columns else idx,
                municipal_name=df.loc[idx, 'municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                region_name=df.loc[idx, 'region_name'] if 'region_name' in df.columns else 'Unknown',
                indicator='high_missing_indicators',
                anomaly_type='logical_inconsistency',
                actual_value=float(missing_pct),
                expected_value=float(threshold),
                deviation=float(missing_pct - threshold),
                deviation_pct=float(((missing_pct - threshold) / threshold * 100) if threshold > 0 else 0),
                severity_score=severity_score,
                z_score=None,
                data_source='metadata',
                detection_method='high_missing_municipality',
                description=description,
                potential_explanation=potential_explanation
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def detect_duplicate_identifiers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect duplicate or inconsistent municipality identifiers.
        
        Identifies cases where:
        1. Multiple rows have the same territory_id (duplicates)
        2. Same municipality name with different IDs (inconsistency)
        3. Same ID with different municipality names (inconsistency)
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of anomaly records
        """
        anomalies = []
        
        # Check 1: Duplicate territory_ids
        if 'territory_id' in df.columns:
            duplicate_ids = df[df.duplicated(subset=['territory_id'], keep=False)]['territory_id'].unique()
            
            for territory_id in duplicate_ids:
                duplicate_rows = df[df['territory_id'] == territory_id]
                
                # Get the first row for reference
                first_row = duplicate_rows.iloc[0]
                
                severity_score = 85.0  # Duplicates are serious data quality issues
                
                description = f"Duplicate territory_id {territory_id} found in {len(duplicate_rows)} rows"
                
                anomaly = self.create_anomaly_record(
                    territory_id=territory_id,
                    municipal_name=first_row['municipal_district_name_short'] if 'municipal_district_name_short' in df.columns else 'Unknown',
                    region_name=first_row['region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator='territory_id',
                    anomaly_type='logical_inconsistency',
                    actual_value=float(len(duplicate_rows)),
                    expected_value=1.0,
                    deviation=float(len(duplicate_rows) - 1),
                    deviation_pct=float((len(duplicate_rows) - 1) * 100),
                    severity_score=severity_score,
                    z_score=None,
                    data_source='metadata',
                    detection_method='duplicate_identifier',
                    description=description,
                    potential_explanation="Duplicate territory identifiers - may indicate data loading error or temporal data without proper time dimension"
                )
                
                anomalies.append(anomaly)
        
        # Check 2: Same municipality name with different IDs
        if 'municipal_district_name_short' in df.columns and 'territory_id' in df.columns:
            name_id_groups = df.groupby('municipal_district_name_short')['territory_id'].nunique()
            inconsistent_names = name_id_groups[name_id_groups > 1].index
            
            for name in inconsistent_names:
                name_rows = df[df['municipal_district_name_short'] == name]
                unique_ids = name_rows['territory_id'].unique()
                
                # Get the first row for reference
                first_row = name_rows.iloc[0]
                
                severity_score = 75.0
                
                description = f"Municipality name '{name}' has {len(unique_ids)} different territory_ids: {list(unique_ids)}"
                
                anomaly = self.create_anomaly_record(
                    territory_id=first_row['territory_id'],
                    municipal_name=name,
                    region_name=first_row['region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator='municipal_name_consistency',
                    anomaly_type='logical_inconsistency',
                    actual_value=float(len(unique_ids)),
                    expected_value=1.0,
                    deviation=float(len(unique_ids) - 1),
                    deviation_pct=float((len(unique_ids) - 1) * 100),
                    severity_score=severity_score,
                    z_score=None,
                    data_source='metadata',
                    detection_method='inconsistent_identifier',
                    description=description,
                    potential_explanation="Same municipality name with different IDs - may indicate naming inconsistencies or different municipalities with similar names"
                )
                
                anomalies.append(anomaly)
        
        # Check 3: Same ID with different municipality names
        if 'territory_id' in df.columns and 'municipal_district_name_short' in df.columns:
            id_name_groups = df.groupby('territory_id')['municipal_district_name_short'].nunique()
            inconsistent_ids = id_name_groups[id_name_groups > 1].index
            
            for territory_id in inconsistent_ids:
                id_rows = df[df['territory_id'] == territory_id]
                unique_names = id_rows['municipal_district_name_short'].unique()
                
                # Get the first row for reference
                first_row = id_rows.iloc[0]
                
                severity_score = 80.0
                
                description = f"Territory ID {territory_id} has {len(unique_names)} different names: {list(unique_names)}"
                
                anomaly = self.create_anomaly_record(
                    territory_id=territory_id,
                    municipal_name=first_row['municipal_district_name_short'],
                    region_name=first_row['region_name'] if 'region_name' in df.columns else 'Unknown',
                    indicator='territory_id_consistency',
                    anomaly_type='logical_inconsistency',
                    actual_value=float(len(unique_names)),
                    expected_value=1.0,
                    deviation=float(len(unique_names) - 1),
                    deviation_pct=float((len(unique_names) - 1) * 100),
                    severity_score=severity_score,
                    z_score=None,
                    data_source='metadata',
                    detection_method='inconsistent_identifier',
                    description=description,
                    potential_explanation="Same territory ID with different names - indicates data quality issue in municipality naming or ID assignment"
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _find_matching_columns(
        self,
        df: pd.DataFrame,
        pattern: Optional[str] = None,
        exact: Optional[str] = None
    ) -> List[str]:
        """
        Find columns matching a pattern or exact name.
        
        Args:
            df: DataFrame to search
            pattern: Pattern to match (substring search)
            exact: Exact column name to match
            
        Returns:
            List of matching column names
        """
        if exact:
            return [exact] if exact in df.columns else []
        
        if pattern:
            return [col for col in df.columns if pattern.lower() in col.lower()]
        
        return []
    
    def _calculate_ratio_severity(self, ratio_multiplier: float) -> float:
        """
        Calculate severity score based on how much a ratio exceeds its maximum.
        
        Args:
            ratio_multiplier: Actual ratio divided by maximum allowed ratio
            
        Returns:
            Severity score between 0 and 100
        """
        if ratio_multiplier >= 100:
            return 100.0
        elif ratio_multiplier >= 50:
            return 95.0
        elif ratio_multiplier >= 10:
            return 85.0
        elif ratio_multiplier >= 5:
            return 75.0
        elif ratio_multiplier >= 2:
            return 65.0
        else:
            return min(ratio_multiplier * 30, 65.0)
    
    def _calculate_contradiction_severity(self, percentile_gap: float) -> float:
        """
        Calculate severity score based on contradiction magnitude.
        
        Args:
            percentile_gap: Gap between high and low indicator percentiles
            
        Returns:
            Severity score between 0 and 100
        """
        if percentile_gap >= 90:
            return 85.0
        elif percentile_gap >= 80:
            return 75.0
        elif percentile_gap >= 70:
            return 65.0
        elif percentile_gap >= 60:
            return 55.0
        else:
            return min(percentile_gap * 0.9, 55.0)
    
    def _calculate_missing_data_severity(self, missing_pct: float, z_score: float) -> float:
        """
        Calculate severity score based on missing data percentage.
        
        Args:
            missing_pct: Percentage of missing indicators
            z_score: Z-score of missing data percentage
            
        Returns:
            Severity score between 0 and 100
        """
        # Base severity on percentage of missing data
        if missing_pct >= 80:
            return 95.0
        elif missing_pct >= 60:
            return 85.0
        elif missing_pct >= 40:
            return 70.0
        elif missing_pct >= 20:
            return 55.0
        else:
            # Also consider z-score for lower percentages
            if z_score >= 3:
                return 60.0
            elif z_score >= 2:
                return 45.0
            else:
                return min(missing_pct * 2, 45.0)
