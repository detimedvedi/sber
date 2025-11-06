"""
Results Aggregation Module for СберИндекс Anomaly Detection System

This module provides functionality to aggregate, rank, and categorize anomalies
detected by various detector classes.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """
    Aggregates and processes results from all anomaly detectors.
    
    Provides methods to:
    - Combine anomalies from multiple detectors
    - Calculate municipality-level anomaly scores
    - Rank anomalies by severity
    - Categorize anomalies by type
    - Calculate priority scores for anomalies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ResultsAggregator.
        
        Args:
            config: Configuration dictionary containing priority weights
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config or {}
        
        # Load priority weights from config
        priority_config = self.config.get('priority_weights', {})
        self.type_weights = priority_config.get('anomaly_types', {
            'logical_inconsistency': 1.5,
            'cross_source_discrepancy': 1.2,
            'temporal_anomaly': 1.1,
            'statistical_outlier': 1.0,
            'geographic_anomaly': 0.8
        })
        self.indicator_weights = priority_config.get('indicators', {
            'population': 1.3,
            'consumption_total': 1.2,
            'salary': 1.1,
            'default': 1.0
        })
    
    def combine_anomalies(self, anomalies: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Combine results from all detectors into a single DataFrame.
        
        Merges anomaly DataFrames from different detectors, removes duplicates,
        and ensures consistent schema across all anomalies. Sorts by priority_score
        if available, otherwise falls back to severity_score.
        
        Args:
            anomalies: List of DataFrames containing anomalies from different detectors
            
        Returns:
            Combined DataFrame with all anomalies, sorted by priority_score (descending)
            or severity_score if priority_score is not present
        """
        self.logger.info(f"Combining anomalies from {len(anomalies)} detectors")
        
        # Filter out empty DataFrames
        non_empty_anomalies = [df for df in anomalies if df is not None and not df.empty]
        
        if not non_empty_anomalies:
            self.logger.warning("No anomalies to combine")
            return pd.DataFrame()
        
        # Log counts from each detector
        for i, df in enumerate(non_empty_anomalies):
            self.logger.info(f"Detector {i+1}: {len(df)} anomalies")
        
        # Concatenate all anomaly DataFrames
        combined_df = pd.concat(non_empty_anomalies, ignore_index=True)
        
        self.logger.info(f"Combined total: {len(combined_df)} anomalies before deduplication")
        
        # Remove exact duplicates (same anomaly detected by multiple methods)
        # Keep the one with highest priority score (if available) or severity score
        sort_column = 'priority_score' if 'priority_score' in combined_df.columns else 'severity_score'
        combined_df = combined_df.sort_values(sort_column, ascending=False)
        combined_df = combined_df.drop_duplicates(
            subset=['territory_id', 'indicator', 'anomaly_type'],
            keep='first'
        )
        
        self.logger.info(f"After deduplication: {len(combined_df)} unique anomalies")
        
        # Sort by priority score (if available) or severity score (descending)
        sort_column = 'priority_score' if 'priority_score' in combined_df.columns else 'severity_score'
        combined_df = combined_df.sort_values(sort_column, ascending=False).reset_index(drop=True)
        self.logger.info(f"Sorted anomalies by {sort_column}")
        
        return combined_df
    
    def calculate_priority_score(self, anomaly: Dict[str, Any]) -> float:
        """
        Calculate priority score for an anomaly based on type and indicator weights.
        
        Priority score is calculated as:
        priority_score = base_severity * type_weight * indicator_weight
        
        Type weights (from config):
        - logical_inconsistency: 1.5 (highest priority)
        - cross_source_discrepancy: 1.2
        - temporal_anomaly: 1.1
        - statistical_outlier: 1.0 (baseline)
        - geographic_anomaly: 0.8 (lowest priority)
        
        Indicator weights (from config):
        - population_*: 1.3 (critical indicator)
        - consumption_total: 1.2 (important indicator)
        - salary_*: 1.1 (important indicator)
        - other: 1.0 (default)
        
        Args:
            anomaly: Dictionary containing anomaly information with keys:
                - severity_score: Base severity score (0-100)
                - anomaly_type: Type of anomaly
                - indicator: Name of the indicator
        
        Returns:
            Priority score (weighted severity score)
        """
        # Get base severity score
        base_severity = anomaly.get('severity_score', 0.0)
        
        # Get anomaly type weight
        anomaly_type = anomaly.get('anomaly_type', '')
        type_weight = self.type_weights.get(anomaly_type, 1.0)
        
        # Get indicator weight
        indicator = anomaly.get('indicator', '')
        indicator_weight = self._get_indicator_weight(indicator)
        
        # Calculate priority score
        priority_score = base_severity * type_weight * indicator_weight
        
        return priority_score
    
    def _get_indicator_weight(self, indicator: str) -> float:
        """
        Determine the weight for a specific indicator.
        
        Checks indicator name against configured patterns:
        - population_* indicators get population weight
        - consumption_total gets consumption_total weight
        - salary_* indicators get salary weight
        - all others get default weight
        
        Args:
            indicator: Name of the indicator
        
        Returns:
            Weight value for the indicator
        """
        indicator_lower = indicator.lower()
        
        # Check for population indicators
        if 'population' in indicator_lower:
            return self.indicator_weights.get('population', 1.3)
        
        # Check for consumption_total
        if indicator_lower == 'consumption_total':
            return self.indicator_weights.get('consumption_total', 1.2)
        
        # Check for salary indicators
        if 'salary' in indicator_lower:
            return self.indicator_weights.get('salary', 1.1)
        
        # Default weight
        return self.indicator_weights.get('default', 1.0)
    
    def add_priority_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add priority scores to all anomalies in the DataFrame.
        
        Calculates priority score for each anomaly and adds it as a new column.
        Also re-sorts the DataFrame by priority score in descending order.
        
        Args:
            df: DataFrame containing anomalies
        
        Returns:
            DataFrame with added 'priority_score' column, sorted by priority
        """
        self.logger.info("Calculating priority scores for all anomalies")
        
        if df.empty:
            self.logger.warning("No anomalies to calculate priority scores for")
            return df
        
        # Calculate priority score for each row
        df['priority_score'] = df.apply(
            lambda row: self.calculate_priority_score(row.to_dict()),
            axis=1
        )
        
        # Sort by priority score (descending)
        df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
        
        # Log statistics
        self.logger.info(f"Priority scores calculated for {len(df)} anomalies")
        self.logger.info(f"Priority score range: {df['priority_score'].min():.2f} - "
                        f"{df['priority_score'].max():.2f}")
        self.logger.info(f"Mean priority score: {df['priority_score'].mean():.2f}")
        
        # Log top priority anomaly
        if len(df) > 0:
            top_anomaly = df.iloc[0]
            self.logger.info(f"Highest priority anomaly: {top_anomaly['municipal_name']} - "
                           f"{top_anomaly['indicator']} ({top_anomaly['anomaly_type']}) - "
                           f"Priority: {top_anomaly['priority_score']:.2f}")
        
        return df
    
    def calculate_municipality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate total anomaly scores for each municipality.
        
        Aggregates anomalies by municipality to compute:
        - Total number of anomalies per municipality
        - Total severity score (sum of all severity scores)
        - List of anomaly types present
        - Average severity score
        
        Args:
            df: DataFrame containing all detected anomalies
            
        Returns:
            DataFrame with municipality-level scores containing columns:
                - territory_id: Municipal territory identifier
                - municipal_name: Name of municipality
                - region_name: Name of region
                - total_anomalies_count: Total number of anomalies
                - total_severity_score: Sum of all severity scores
                - average_severity_score: Average severity score
                - anomaly_types: List of anomaly types present
                - max_severity: Maximum severity score among all anomalies
        """
        self.logger.info("Calculating municipality-level anomaly scores")
        
        if df.empty:
            self.logger.warning("No anomalies to calculate scores from")
            return pd.DataFrame()
        
        # Group by municipality
        grouped = df.groupby('territory_id')
        
        # Calculate aggregated metrics
        municipality_scores = grouped.agg({
            'anomaly_id': 'count',  # Total count of anomalies
            'severity_score': ['sum', 'mean', 'max'],  # Severity metrics
            'municipal_name': 'first',  # Municipality name (should be same for all)
            'region_name': 'first',  # Region name (should be same for all)
            'anomaly_type': lambda x: list(x.unique())  # List of unique anomaly types
        }).reset_index()
        
        # Flatten multi-level column names
        municipality_scores.columns = [
            'territory_id',
            'total_anomalies_count',
            'total_severity_score',
            'average_severity_score',
            'max_severity',
            'municipal_name',
            'region_name',
            'anomaly_types'
        ]
        
        # Sort by total severity score (descending)
        municipality_scores = municipality_scores.sort_values(
            'total_severity_score',
            ascending=False
        ).reset_index(drop=True)
        
        # Add rank
        municipality_scores['rank'] = range(1, len(municipality_scores) + 1)
        
        self.logger.info(f"Calculated scores for {len(municipality_scores)} municipalities")
        self.logger.info(f"Top municipality: {municipality_scores.iloc[0]['municipal_name']} "
                        f"with {municipality_scores.iloc[0]['total_anomalies_count']} anomalies "
                        f"and total severity score {municipality_scores.iloc[0]['total_severity_score']:.2f}")
        
        return municipality_scores
    
    def rank_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort anomalies by priority score and add ranking information.
        
        Ranks all anomalies by priority score (if available) or severity score,
        and adds additional ranking columns for different perspectives 
        (by type, by region, etc.).
        
        Args:
            df: DataFrame containing all detected anomalies
            
        Returns:
            DataFrame with added ranking columns:
                - overall_rank: Overall rank by priority score (or severity if no priority)
                - rank_in_type: Rank within anomaly type
                - rank_in_region: Rank within region
        """
        self.logger.info("Ranking anomalies by priority")
        
        if df.empty:
            self.logger.warning("No anomalies to rank")
            return df
        
        # Create a copy to avoid modifying original
        ranked_df = df.copy()
        
        # Determine which score to use for ranking
        ranking_column = 'priority_score' if 'priority_score' in ranked_df.columns else 'severity_score'
        
        # Overall rank by priority/severity score
        ranked_df = ranked_df.sort_values(ranking_column, ascending=False).reset_index(drop=True)
        ranked_df['overall_rank'] = range(1, len(ranked_df) + 1)
        
        # Rank within anomaly type
        ranked_df['rank_in_type'] = ranked_df.groupby('anomaly_type')[ranking_column].rank(
            method='dense',
            ascending=False
        ).astype(int)
        
        # Rank within region (if region information is available)
        if 'region_name' in ranked_df.columns:
            ranked_df['rank_in_region'] = ranked_df.groupby('region_name')[ranking_column].rank(
                method='dense',
                ascending=False
            ).astype(int)
        else:
            ranked_df['rank_in_region'] = None
        
        self.logger.info(f"Ranked {len(ranked_df)} anomalies using {ranking_column}")
        self.logger.info(f"{ranking_column.replace('_', ' ').title()} range: "
                        f"{ranked_df[ranking_column].min():.2f} - "
                        f"{ranked_df[ranking_column].max():.2f}")
        
        return ranked_df
    
    def categorize_anomalies(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group anomalies by type into separate DataFrames.
        
        Splits the combined anomalies DataFrame into separate DataFrames
        for each anomaly type, making it easier to analyze and export
        specific types of anomalies. Each category is sorted by priority_score
        if available, otherwise by severity_score.
        
        Args:
            df: DataFrame containing all detected anomalies
            
        Returns:
            Dictionary mapping anomaly type names to DataFrames (sorted by priority):
                - 'statistical_outliers': Statistical outlier anomalies
                - 'temporal_anomalies': Temporal anomaly detections
                - 'geographic_anomalies': Geographic anomaly detections
                - 'cross_source_discrepancies': Cross-source comparison anomalies
                - 'logical_inconsistencies': Logical consistency issues
                - 'data_quality_issues': General data quality problems
        """
        self.logger.info("Categorizing anomalies by type")
        
        if df.empty:
            self.logger.warning("No anomalies to categorize")
            return {}
        
        # Define category mappings
        # Map specific anomaly_type values to broader categories
        category_mapping = {
            'statistical_outlier': 'statistical_outliers',
            'temporal_anomaly': 'temporal_anomalies',
            'geographic_anomaly': 'geographic_anomalies',
            'cross_source_discrepancy': 'cross_source_discrepancies',
            'logical_inconsistency': 'logical_inconsistencies',
            'data_quality_issue': 'data_quality_issues'
        }
        
        # Create categorized dictionary
        categorized = {}
        
        # Get unique anomaly types in the data
        unique_types = df['anomaly_type'].unique()
        
        for anomaly_type in unique_types:
            # Get the category name (use mapping or default to the type itself)
            category_name = category_mapping.get(anomaly_type, anomaly_type)
            
            # Filter anomalies of this type
            type_df = df[df['anomaly_type'] == anomaly_type].copy()
            
            # Sort by priority score (if available) or severity within category
            sort_column = 'priority_score' if 'priority_score' in type_df.columns else 'severity_score'
            type_df = type_df.sort_values(sort_column, ascending=False).reset_index(drop=True)
            
            # Add to categorized dictionary
            categorized[category_name] = type_df
            
            self.logger.info(f"Category '{category_name}': {len(type_df)} anomalies")
        
        # Ensure all expected categories exist (even if empty)
        expected_categories = [
            'statistical_outliers',
            'temporal_anomalies',
            'geographic_anomalies',
            'cross_source_discrepancies',
            'logical_inconsistencies',
            'data_quality_issues'
        ]
        
        for category in expected_categories:
            if category not in categorized:
                categorized[category] = pd.DataFrame()
                self.logger.info(f"Category '{category}': 0 anomalies (empty)")
        
        return categorized
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for all anomalies.
        
        Provides high-level statistics about the detected anomalies including
        counts by type, severity distribution, geographic distribution, etc.
        
        Args:
            df: DataFrame containing all detected anomalies
            
        Returns:
            Dictionary containing summary statistics
        """
        self.logger.info("Calculating summary statistics")
        
        if df.empty:
            return {
                'total_anomalies': 0,
                'total_municipalities_affected': 0,
                'by_type': {},
                'by_region': {},
                'severity_stats': {},
                'data_source_distribution': {}
            }
        
        summary = {}
        
        # Total counts
        summary['total_anomalies'] = len(df)
        summary['total_municipalities_affected'] = df['territory_id'].nunique()
        
        # Counts by anomaly type
        summary['by_type'] = df['anomaly_type'].value_counts().to_dict()
        
        # Counts by region (if available)
        if 'region_name' in df.columns:
            summary['by_region'] = df['region_name'].value_counts().head(10).to_dict()
        else:
            summary['by_region'] = {}
        
        # Severity statistics
        summary['severity_stats'] = {
            'mean': float(df['severity_score'].mean()),
            'median': float(df['severity_score'].median()),
            'min': float(df['severity_score'].min()),
            'max': float(df['severity_score'].max()),
            'std': float(df['severity_score'].std())
        }
        
        # Severity distribution by category
        severity_categories = pd.cut(
            df['severity_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low (0-25)', 'Medium (25-50)', 'High (50-75)', 'Critical (75-100)']
        )
        summary['severity_distribution'] = severity_categories.value_counts().to_dict()
        
        # Data source distribution
        if 'data_source' in df.columns:
            summary['data_source_distribution'] = df['data_source'].value_counts().to_dict()
        else:
            summary['data_source_distribution'] = {}
        
        # Detection method distribution
        if 'detection_method' in df.columns:
            summary['detection_method_distribution'] = df['detection_method'].value_counts().to_dict()
        else:
            summary['detection_method_distribution'] = {}
        
        self.logger.info(f"Summary: {summary['total_anomalies']} total anomalies "
                        f"affecting {summary['total_municipalities_affected']} municipalities")
        
        return summary
    
    def group_related_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group related anomalies by territory_id and identify patterns.
        
        Analyzes anomalies for each municipality to identify:
        - Multiple indicators affected (systemic issue)
        - Same indicator detected by multiple detectors (high confidence)
        - Related indicators affected (e.g., all salary_* indicators)
        
        Adds grouping information to the DataFrame:
        - anomaly_group_id: UUID for grouped anomalies
        - pattern_type: Type of pattern detected
        - pattern_description: Human-readable description of the pattern
        - related_anomaly_count: Number of related anomalies in the group
        
        Args:
            df: DataFrame containing all detected anomalies
            
        Returns:
            DataFrame with added grouping columns
        """
        import uuid
        
        self.logger.info("Grouping related anomalies by territory")
        
        if df.empty:
            self.logger.warning("No anomalies to group")
            return df
        
        # Create a copy to avoid modifying original
        grouped_df = df.copy()
        
        # Initialize new columns
        grouped_df['anomaly_group_id'] = None
        grouped_df['pattern_type'] = None
        grouped_df['pattern_description'] = None
        grouped_df['related_anomaly_count'] = 0
        grouped_df['root_cause'] = None
        
        # Group by territory_id
        territory_groups = grouped_df.groupby('territory_id')
        
        patterns_found = {
            'multiple_indicators': 0,
            'same_indicator_multiple_detectors': 0,
            'related_indicators': 0,
            'single_anomaly': 0
        }
        
        for territory_id, territory_anomalies in territory_groups:
            # Generate a unique group ID for this territory
            group_id = str(uuid.uuid4())
            
            # Get indices for this territory
            indices = territory_anomalies.index
            
            # Count of anomalies for this territory
            anomaly_count = len(territory_anomalies)
            
            # Set the group ID and count for all anomalies in this territory
            grouped_df.loc[indices, 'anomaly_group_id'] = group_id
            grouped_df.loc[indices, 'related_anomaly_count'] = anomaly_count
            
            # Identify root cause for this territory
            root_cause = self.identify_root_cause(territory_anomalies)
            grouped_df.loc[indices, 'root_cause'] = root_cause
            
            # Analyze patterns
            if anomaly_count == 1:
                # Single anomaly - no pattern
                grouped_df.loc[indices, 'pattern_type'] = 'single_anomaly'
                grouped_df.loc[indices, 'pattern_description'] = 'Единичная аномалия'
                patterns_found['single_anomaly'] += 1
                continue
            
            # Check for same indicator detected by multiple detectors
            indicator_detector_groups = territory_anomalies.groupby('indicator')
            same_indicator_multiple = []
            
            for indicator, indicator_group in indicator_detector_groups:
                if len(indicator_group) > 1:
                    # Same indicator detected by multiple detectors
                    detector_types = indicator_group['anomaly_type'].unique()
                    if len(detector_types) > 1:
                        same_indicator_multiple.append(indicator)
            
            if same_indicator_multiple:
                # Pattern: Same indicator across multiple detectors (high confidence)
                grouped_df.loc[indices, 'pattern_type'] = 'same_indicator_multiple_detectors'
                grouped_df.loc[indices, 'pattern_description'] = (
                    f"Показатель(и) {', '.join(same_indicator_multiple[:3])} "
                    f"обнаружены несколькими детекторами (высокая достоверность)"
                )
                patterns_found['same_indicator_multiple_detectors'] += 1
                continue
            
            # Check for related indicators (e.g., all salary_*, all population_*)
            indicators = territory_anomalies['indicator'].unique()
            indicator_prefixes = {}
            
            for indicator in indicators:
                # Extract prefix (e.g., 'salary' from 'salary_Финансы')
                if '_' in indicator:
                    prefix = indicator.split('_')[0]
                    if prefix not in indicator_prefixes:
                        indicator_prefixes[prefix] = []
                    indicator_prefixes[prefix].append(indicator)
            
            # Check if multiple indicators share the same prefix
            related_prefix = None
            for prefix, prefix_indicators in indicator_prefixes.items():
                if len(prefix_indicators) >= 2:
                    related_prefix = prefix
                    break
            
            if related_prefix:
                # Pattern: Related indicators affected
                grouped_df.loc[indices, 'pattern_type'] = 'related_indicators'
                grouped_df.loc[indices, 'pattern_description'] = (
                    f"Множественные показатели категории '{related_prefix}' "
                    f"({len(indicator_prefixes[related_prefix])} показателей)"
                )
                patterns_found['related_indicators'] += 1
                continue
            
            # Default: Multiple different indicators (systemic issue)
            unique_indicators = len(indicators)
            grouped_df.loc[indices, 'pattern_type'] = 'multiple_indicators'
            grouped_df.loc[indices, 'pattern_description'] = (
                f"Системная проблема: {unique_indicators} различных показателей"
            )
            patterns_found['multiple_indicators'] += 1
        
        # Log pattern statistics
        self.logger.info(f"Grouped {len(df)} anomalies into {territory_groups.ngroups} territories")
        self.logger.info(f"Pattern distribution:")
        for pattern_type, count in patterns_found.items():
            if count > 0:
                self.logger.info(f"  - {pattern_type}: {count} territories")
        
        # Log examples of high-confidence patterns
        high_confidence = grouped_df[
            grouped_df['pattern_type'] == 'same_indicator_multiple_detectors'
        ]
        if not high_confidence.empty:
            example = high_confidence.iloc[0]
            self.logger.info(
                f"Example high-confidence anomaly: {example['municipal_name']} - "
                f"{example['indicator']} detected by multiple methods"
            )
        
        return grouped_df
    
    def identify_root_cause(self, territory_anomalies: pd.DataFrame) -> str:
        """
        Analyze anomalies for one territory and suggest root cause.
        
        Examines patterns in anomalies to identify likely root causes:
        - Missing data (>70% of indicators missing)
        - Duplicate records (multiple temporal records)
        - Systematic cross-source discrepancies
        - Extreme values across multiple indicators
        - Data quality issues
        
        Args:
            territory_anomalies: DataFrame containing all anomalies for a single territory
            
        Returns:
            Russian-language description of the identified root cause
        """
        if territory_anomalies.empty:
            return "Неизвестная причина"
        
        anomaly_count = len(territory_anomalies)
        anomaly_types = territory_anomalies['anomaly_type'].value_counts().to_dict()
        unique_indicators = territory_anomalies['indicator'].nunique()
        
        # Pattern 1: Missing data (logical inconsistencies with missing values)
        logical_inconsistencies = anomaly_types.get('logical_inconsistency', 0)
        if logical_inconsistencies > 0:
            # Check if descriptions mention missing data
            missing_data_keywords = ['отсутств', 'пропущ', 'missing', 'null', 'nan']
            missing_data_count = 0
            
            if 'description' in territory_anomalies.columns:
                for desc in territory_anomalies['description']:
                    if isinstance(desc, str):
                        desc_lower = desc.lower()
                        if any(keyword in desc_lower for keyword in missing_data_keywords):
                            missing_data_count += 1
            
            # If majority of logical inconsistencies are about missing data
            if missing_data_count >= logical_inconsistencies * 0.5:
                missing_pct = (missing_data_count / anomaly_count) * 100
                return f"Данные отсутствуют или неполные ({missing_pct:.0f}% показателей)"
        
        # Pattern 2: Duplicate records (temporal anomalies or data quality issues)
        temporal_anomalies = anomaly_types.get('temporal_anomaly', 0)
        data_quality_issues = anomaly_types.get('data_quality_issue', 0)
        
        if temporal_anomalies > 0 or data_quality_issues > 0:
            # Check if descriptions mention duplicates
            duplicate_keywords = ['дубликат', 'повтор', 'duplicate', 'множественн']
            duplicate_count = 0
            
            if 'description' in territory_anomalies.columns:
                for desc in territory_anomalies['description']:
                    if isinstance(desc, str):
                        desc_lower = desc.lower()
                        if any(keyword in desc_lower for keyword in duplicate_keywords):
                            duplicate_count += 1
            
            if duplicate_count > 0:
                return f"Дубликаты записей обнаружены ({duplicate_count} случаев)"
        
        # Pattern 3: Systematic cross-source discrepancies
        cross_source_discrepancies = anomaly_types.get('cross_source_discrepancy', 0)
        
        if cross_source_discrepancies >= 3:
            # Multiple cross-source discrepancies indicate systematic issue
            discrepancy_pct = (cross_source_discrepancies / anomaly_count) * 100
            return (
                f"Систематическое расхождение между источниками данных "
                f"({cross_source_discrepancies} показателей, {discrepancy_pct:.0f}%)"
            )
        
        # Pattern 4: Multiple high-severity anomalies across different types
        high_severity_count = len(territory_anomalies[territory_anomalies['severity_score'] >= 75])
        
        if high_severity_count >= 3 and len(anomaly_types) > 1:
            anomaly_type_count = len(anomaly_types)
            return (
                f"Множественные критические аномалии "
                f"({high_severity_count} аномалий высокой серьезности, "
                f"{anomaly_type_count} типов)"
            )
        
        # Pattern 5: Extreme values (high severity statistical outliers)
        statistical_outliers = anomaly_types.get('statistical_outlier', 0)
        
        if statistical_outliers > 0:
            # Check average severity of statistical outliers
            stat_outlier_anomalies = territory_anomalies[
                territory_anomalies['anomaly_type'] == 'statistical_outlier'
            ]
            if not stat_outlier_anomalies.empty:
                avg_severity = stat_outlier_anomalies['severity_score'].mean()
                
                if avg_severity >= 80:
                    return (
                        f"Экстремальные значения показателей "
                        f"({statistical_outliers} показателей со средней серьезностью {avg_severity:.0f})"
                    )
        
        # Pattern 6: Geographic anomalies (municipality differs from region)
        geographic_anomalies = anomaly_types.get('geographic_anomaly', 0)
        
        if geographic_anomalies >= anomaly_count * 0.6:
            # Majority are geographic anomalies
            return (
                f"Значительное отличие от региональных показателей "
                f"({geographic_anomalies} из {anomaly_count} аномалий)"
            )
        
        # Pattern 7: Single dominant anomaly type
        if len(anomaly_types) == 1:
            dominant_type = list(anomaly_types.keys())[0]
            type_names = {
                'logical_inconsistency': 'логические несоответствия',
                'cross_source_discrepancy': 'расхождения между источниками',
                'temporal_anomaly': 'временные аномалии',
                'statistical_outlier': 'статистические выбросы',
                'geographic_anomaly': 'географические аномалии',
                'data_quality_issue': 'проблемы качества данных'
            }
            type_name = type_names.get(dominant_type, dominant_type)
            return f"Проблема типа: {type_name} ({anomaly_count} случаев)"
        
        # Pattern 8: Multiple indicators of same category
        if unique_indicators >= 3:
            # Check if indicators share common prefixes
            indicators = territory_anomalies['indicator'].unique()
            prefixes = {}
            
            for indicator in indicators:
                if '_' in indicator:
                    prefix = indicator.split('_')[0]
                    if prefix not in prefixes:
                        prefixes[prefix] = 0
                    prefixes[prefix] += 1
            
            # Find dominant prefix
            if prefixes:
                dominant_prefix = max(prefixes, key=prefixes.get)
                prefix_count = prefixes[dominant_prefix]
                
                if prefix_count >= 3:
                    prefix_names = {
                        'salary': 'зарплаты',
                        'population': 'населения',
                        'consumption': 'потребления',
                        'connection': 'подключений',
                        'market': 'рыночного доступа'
                    }
                    prefix_name = prefix_names.get(dominant_prefix, dominant_prefix)
                    return (
                        f"Проблемы с данными категории '{prefix_name}' "
                        f"({prefix_count} показателей)"
                    )
        
        # Default: Unknown cause with general statistics
        return (
            f"Комплексная проблема: {anomaly_count} аномалий, "
            f"{unique_indicators} показателей, {len(anomaly_types)} типов"
        )
    
    def get_top_anomalous_municipalities(
        self,
        municipality_scores: pd.DataFrame,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Get the top N most anomalous municipalities.
        
        Returns the municipalities with the highest total anomaly scores,
        useful for focusing attention on the most problematic areas.
        
        Args:
            municipality_scores: DataFrame with municipality-level scores
            top_n: Number of top municipalities to return (default: 50)
            
        Returns:
            DataFrame with top N municipalities sorted by total severity score
        """
        self.logger.info(f"Getting top {top_n} anomalous municipalities")
        
        if municipality_scores.empty:
            self.logger.warning("No municipality scores available")
            return pd.DataFrame()
        
        # Get top N municipalities
        top_municipalities = municipality_scores.head(top_n).copy()
        
        self.logger.info(f"Returning top {len(top_municipalities)} municipalities")
        
        return top_municipalities
    
    def calculate_detection_metrics(
        self,
        anomalies_df: pd.DataFrame,
        total_municipalities: int
    ) -> Dict[str, Any]:
        """
        Calculate detection metrics for validation and reporting.
        
        Provides comprehensive metrics about the anomaly detection results:
        - Count anomalies by type and severity
        - Calculate percentage of municipalities affected
        - Calculate anomaly rate per 1000 municipalities
        
        These metrics help assess the effectiveness of the detection system
        and identify potential issues with threshold settings.
        
        Args:
            anomalies_df: DataFrame containing all detected anomalies
            total_municipalities: Total number of municipalities in the dataset
            
        Returns:
            Dictionary containing detection metrics:
                - total_anomalies: Total number of anomalies detected
                - anomalies_by_type: Dictionary with counts per anomaly type
                - anomalies_by_severity: Dictionary with counts per severity category
                - municipalities_affected: Number of unique municipalities with anomalies
                - municipalities_affected_pct: Percentage of municipalities affected
                - anomaly_rate_per_1000: Anomaly rate per 1000 municipalities
                - anomaly_rate_per_1000_by_type: Anomaly rate per 1000 by type
                - avg_anomalies_per_municipality: Average anomalies per affected municipality
                - severity_distribution: Detailed severity distribution
        """
        self.logger.info("Calculating detection metrics")
        
        if anomalies_df.empty:
            self.logger.warning("No anomalies to calculate metrics from")
            return {
                'total_anomalies': 0,
                'anomalies_by_type': {},
                'anomalies_by_severity': {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'municipalities_affected': 0,
                'municipalities_affected_pct': 0.0,
                'anomaly_rate_per_1000': 0.0,
                'anomaly_rate_per_1000_by_type': {},
                'avg_anomalies_per_municipality': 0.0,
                'severity_distribution': {}
            }
        
        # Total anomalies
        total_anomalies = len(anomalies_df)
        
        # Count anomalies by type
        anomalies_by_type = {}
        if 'anomaly_type' in anomalies_df.columns:
            type_counts = anomalies_df['anomaly_type'].value_counts()
            anomalies_by_type = type_counts.to_dict()
            
            self.logger.info(f"Anomalies by type: {anomalies_by_type}")
        else:
            self.logger.warning("Column 'anomaly_type' not found in anomalies DataFrame")
        
        # Count anomalies by severity category
        anomalies_by_severity = {
            'critical': 0,  # 90-100
            'high': 0,      # 70-90
            'medium': 0,    # 50-70
            'low': 0        # 0-50
        }
        
        if 'severity_score' in anomalies_df.columns:
            anomalies_by_severity['critical'] = len(anomalies_df[anomalies_df['severity_score'] >= 90])
            anomalies_by_severity['high'] = len(anomalies_df[
                (anomalies_df['severity_score'] >= 70) & (anomalies_df['severity_score'] < 90)
            ])
            anomalies_by_severity['medium'] = len(anomalies_df[
                (anomalies_df['severity_score'] >= 50) & (anomalies_df['severity_score'] < 70)
            ])
            anomalies_by_severity['low'] = len(anomalies_df[anomalies_df['severity_score'] < 50])
            
            self.logger.info(f"Anomalies by severity: {anomalies_by_severity}")
        else:
            self.logger.warning("Column 'severity_score' not found in anomalies DataFrame")
        
        # Calculate municipalities affected
        municipalities_affected = 0
        if 'territory_id' in anomalies_df.columns:
            municipalities_affected = anomalies_df['territory_id'].nunique()
            self.logger.info(f"Municipalities affected: {municipalities_affected}")
        else:
            self.logger.warning("Column 'territory_id' not found in anomalies DataFrame")
        
        # Calculate percentage of municipalities affected
        municipalities_affected_pct = 0.0
        if total_municipalities > 0:
            municipalities_affected_pct = (municipalities_affected / total_municipalities) * 100
            self.logger.info(f"Percentage of municipalities affected: {municipalities_affected_pct:.2f}%")
        else:
            self.logger.warning("Total municipalities is 0, cannot calculate percentage")
        
        # Calculate anomaly rate per 1000 municipalities
        anomaly_rate_per_1000 = 0.0
        if total_municipalities > 0:
            anomaly_rate_per_1000 = (total_anomalies / total_municipalities) * 1000
            self.logger.info(f"Anomaly rate per 1000 municipalities: {anomaly_rate_per_1000:.2f}")
        else:
            self.logger.warning("Total municipalities is 0, cannot calculate anomaly rate")
        
        # Calculate anomaly rate per 1000 municipalities by type
        anomaly_rate_per_1000_by_type = {}
        if total_municipalities > 0 and anomalies_by_type:
            for anomaly_type, count in anomalies_by_type.items():
                rate = (count / total_municipalities) * 1000
                anomaly_rate_per_1000_by_type[anomaly_type] = round(rate, 2)
            
            self.logger.info(f"Anomaly rate per 1000 by type: {anomaly_rate_per_1000_by_type}")
        
        # Calculate average anomalies per affected municipality
        avg_anomalies_per_municipality = 0.0
        if municipalities_affected > 0:
            avg_anomalies_per_municipality = total_anomalies / municipalities_affected
            self.logger.info(f"Average anomalies per affected municipality: {avg_anomalies_per_municipality:.2f}")
        
        # Detailed severity distribution (percentages)
        severity_distribution = {}
        if 'severity_score' in anomalies_df.columns:
            for category, count in anomalies_by_severity.items():
                pct = (count / total_anomalies * 100) if total_anomalies > 0 else 0.0
                severity_distribution[category] = {
                    'count': count,
                    'percentage': round(pct, 2)
                }
        
        # Compile metrics
        metrics = {
            'total_anomalies': total_anomalies,
            'anomalies_by_type': anomalies_by_type,
            'anomalies_by_severity': anomalies_by_severity,
            'municipalities_affected': municipalities_affected,
            'municipalities_affected_pct': round(municipalities_affected_pct, 2),
            'anomaly_rate_per_1000': round(anomaly_rate_per_1000, 2),
            'anomaly_rate_per_1000_by_type': anomaly_rate_per_1000_by_type,
            'avg_anomalies_per_municipality': round(avg_anomalies_per_municipality, 2),
            'severity_distribution': severity_distribution
        }
        
        # Log summary
        self.logger.info(
            "Detection metrics calculated",
            extra={
                'total_anomalies': total_anomalies,
                'municipalities_affected': municipalities_affected,
                'municipalities_affected_pct': round(municipalities_affected_pct, 2),
                'anomaly_rate_per_1000': round(anomaly_rate_per_1000, 2),
                'critical_anomalies': anomalies_by_severity.get('critical', 0)
            }
        )
        
        return metrics
    
    def check_anomaly_count_warnings(
        self,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check if anomaly counts are within expected ranges and generate warnings.
        
        Analyzes detection metrics to identify potential issues with threshold settings:
        - Too many anomalies (possible false positives)
        - Too few anomalies (possible missed detections)
        - Unusual distribution by type or severity
        
        Provides actionable recommendations for threshold adjustments.
        
        Args:
            metrics: Detection metrics from calculate_detection_metrics()
            config: Optional configuration with expected ranges
            
        Returns:
            List of warning dictionaries, each containing:
                - warning_type: Type of warning (e.g., 'high_anomaly_count')
                - severity: Warning severity ('info', 'warning', 'critical')
                - message: Human-readable warning message
                - recommendation: Suggested action to address the issue
                - affected_metric: The metric that triggered the warning
                - current_value: Current value of the metric
                - expected_range: Expected range for the metric
        """
        self.logger.info("Checking anomaly count warnings")
        
        warnings = []
        
        # Get expected ranges from config or use defaults
        if config is None:
            config = self.config
        
        expected_ranges = config.get('expected_anomaly_ranges', {
            'total_anomalies': {
                'min': 10,
                'max': 5000,
                'optimal_min': 100,
                'optimal_max': 2000
            },
            'municipalities_affected_pct': {
                'min': 1.0,
                'max': 90.0,
                'optimal_min': 5.0,
                'optimal_max': 50.0
            },
            'anomaly_rate_per_1000': {
                'min': 10,
                'max': 5000,
                'optimal_min': 50,
                'optimal_max': 1500
            },
            'critical_anomalies_pct': {
                'min': 1.0,
                'max': 50.0,
                'optimal_min': 5.0,
                'optimal_max': 20.0
            },
            'avg_anomalies_per_municipality': {
                'min': 1.0,
                'max': 20.0,
                'optimal_min': 2.0,
                'optimal_max': 8.0
            }
        })
        
        # Check total anomalies
        total_anomalies = metrics.get('total_anomalies', 0)
        total_range = expected_ranges.get('total_anomalies', {})
        
        if total_anomalies < total_range.get('min', 10):
            warnings.append({
                'warning_type': 'low_anomaly_count',
                'severity': 'warning',
                'message': f"Очень мало аномалий обнаружено: {total_anomalies}",
                'recommendation': "Рассмотрите снижение порогов детекции (уменьшите z_score, iqr_multiplier)",
                'affected_metric': 'total_anomalies',
                'current_value': total_anomalies,
                'expected_range': f"{total_range.get('min', 10)}-{total_range.get('max', 5000)}"
            })
            self.logger.warning(
                f"Low anomaly count detected: {total_anomalies} (expected: {total_range.get('min', 10)}-{total_range.get('max', 5000)})"
            )
        
        elif total_anomalies > total_range.get('max', 5000):
            warnings.append({
                'warning_type': 'high_anomaly_count',
                'severity': 'warning',
                'message': f"Слишком много аномалий обнаружено: {total_anomalies}",
                'recommendation': "Рассмотрите повышение порогов детекции (увеличьте z_score, iqr_multiplier)",
                'affected_metric': 'total_anomalies',
                'current_value': total_anomalies,
                'expected_range': f"{total_range.get('min', 10)}-{total_range.get('max', 5000)}"
            })
            self.logger.warning(
                f"High anomaly count detected: {total_anomalies} (expected: {total_range.get('min', 10)}-{total_range.get('max', 5000)})"
            )
        
        elif total_anomalies > total_range.get('optimal_max', 2000):
            warnings.append({
                'warning_type': 'suboptimal_high_anomaly_count',
                'severity': 'info',
                'message': f"Количество аномалий выше оптимального: {total_anomalies}",
                'recommendation': "Возможны ложные срабатывания. Рассмотрите небольшое повышение порогов",
                'affected_metric': 'total_anomalies',
                'current_value': total_anomalies,
                'expected_range': f"{total_range.get('optimal_min', 100)}-{total_range.get('optimal_max', 2000)} (оптимально)"
            })
            self.logger.info(
                f"Anomaly count above optimal range: {total_anomalies} (optimal: {total_range.get('optimal_min', 100)}-{total_range.get('optimal_max', 2000)})"
            )
        
        # Check municipalities affected percentage
        municipalities_affected_pct = metrics.get('municipalities_affected_pct', 0.0)
        muni_pct_range = expected_ranges.get('municipalities_affected_pct', {})
        
        if municipalities_affected_pct > muni_pct_range.get('max', 90.0):
            warnings.append({
                'warning_type': 'high_municipalities_affected',
                'severity': 'critical',
                'message': f"Слишком много муниципалитетов с аномалиями: {municipalities_affected_pct:.1f}%",
                'recommendation': "Критически высокий процент. Повысьте пороги детекции значительно",
                'affected_metric': 'municipalities_affected_pct',
                'current_value': municipalities_affected_pct,
                'expected_range': f"{muni_pct_range.get('min', 1.0)}-{muni_pct_range.get('max', 90.0)}%"
            })
            self.logger.warning(
                f"High percentage of municipalities affected: {municipalities_affected_pct:.1f}% (expected: <{muni_pct_range.get('max', 90.0)}%)"
            )
        
        elif municipalities_affected_pct > muni_pct_range.get('optimal_max', 50.0):
            warnings.append({
                'warning_type': 'suboptimal_high_municipalities_affected',
                'severity': 'warning',
                'message': f"Высокий процент муниципалитетов с аномалиями: {municipalities_affected_pct:.1f}%",
                'recommendation': "Возможны ложные срабатывания. Рассмотрите повышение порогов",
                'affected_metric': 'municipalities_affected_pct',
                'current_value': municipalities_affected_pct,
                'expected_range': f"{muni_pct_range.get('optimal_min', 5.0)}-{muni_pct_range.get('optimal_max', 50.0)}% (оптимально)"
            })
            self.logger.info(
                f"Municipalities affected above optimal: {municipalities_affected_pct:.1f}% (optimal: <{muni_pct_range.get('optimal_max', 50.0)}%)"
            )
        
        # Check anomaly rate per 1000
        anomaly_rate = metrics.get('anomaly_rate_per_1000', 0.0)
        rate_range = expected_ranges.get('anomaly_rate_per_1000', {})
        
        if anomaly_rate > rate_range.get('max', 5000):
            warnings.append({
                'warning_type': 'high_anomaly_rate',
                'severity': 'warning',
                'message': f"Очень высокая частота аномалий: {anomaly_rate:.1f} на 1000 муниципалитетов",
                'recommendation': "Повысьте пороги детекции для снижения частоты",
                'affected_metric': 'anomaly_rate_per_1000',
                'current_value': anomaly_rate,
                'expected_range': f"{rate_range.get('min', 10)}-{rate_range.get('max', 5000)}"
            })
            self.logger.warning(
                f"High anomaly rate: {anomaly_rate:.1f} per 1000 (expected: <{rate_range.get('max', 5000)})"
            )
        
        # Check critical anomalies percentage
        total_anomalies = metrics.get('total_anomalies', 0)
        if total_anomalies > 0:
            critical_count = metrics.get('anomalies_by_severity', {}).get('critical', 0)
            critical_pct = (critical_count / total_anomalies) * 100
            critical_range = expected_ranges.get('critical_anomalies_pct', {})
            
            if critical_pct > critical_range.get('max', 50.0):
                warnings.append({
                    'warning_type': 'high_critical_percentage',
                    'severity': 'warning',
                    'message': f"Слишком много критических аномалий: {critical_pct:.1f}%",
                    'recommendation': "Пересмотрите критерии серьезности или повысьте пороги",
                    'affected_metric': 'critical_anomalies_pct',
                    'current_value': critical_pct,
                    'expected_range': f"{critical_range.get('min', 1.0)}-{critical_range.get('max', 50.0)}%"
                })
                self.logger.warning(
                    f"High percentage of critical anomalies: {critical_pct:.1f}% (expected: <{critical_range.get('max', 50.0)}%)"
                )
            
            elif critical_pct < critical_range.get('min', 1.0):
                warnings.append({
                    'warning_type': 'low_critical_percentage',
                    'severity': 'info',
                    'message': f"Мало критических аномалий: {critical_pct:.1f}%",
                    'recommendation': "Возможно, критерии серьезности слишком строгие",
                    'affected_metric': 'critical_anomalies_pct',
                    'current_value': critical_pct,
                    'expected_range': f"{critical_range.get('min', 1.0)}-{critical_range.get('max', 50.0)}%"
                })
                self.logger.info(
                    f"Low percentage of critical anomalies: {critical_pct:.1f}% (expected: >{critical_range.get('min', 1.0)}%)"
                )
        
        # Check average anomalies per municipality
        avg_anomalies = metrics.get('avg_anomalies_per_municipality', 0.0)
        avg_range = expected_ranges.get('avg_anomalies_per_municipality', {})
        
        if avg_anomalies > avg_range.get('max', 20.0):
            warnings.append({
                'warning_type': 'high_avg_anomalies_per_municipality',
                'severity': 'warning',
                'message': f"Слишком много аномалий на муниципалитет: {avg_anomalies:.1f}",
                'recommendation': "Возможны системные проблемы с данными или порогами",
                'affected_metric': 'avg_anomalies_per_municipality',
                'current_value': avg_anomalies,
                'expected_range': f"{avg_range.get('min', 1.0)}-{avg_range.get('max', 20.0)}"
            })
            self.logger.warning(
                f"High average anomalies per municipality: {avg_anomalies:.1f} (expected: <{avg_range.get('max', 20.0)})"
            )
        
        # Check for anomaly type imbalances
        anomalies_by_type = metrics.get('anomalies_by_type', {})
        if anomalies_by_type and total_anomalies > 0:
            for anomaly_type, count in anomalies_by_type.items():
                type_pct = (count / total_anomalies) * 100
                
                # Warn if one type dominates (>70%)
                if type_pct > 70.0:
                    warnings.append({
                        'warning_type': 'dominant_anomaly_type',
                        'severity': 'info',
                        'message': f"Тип '{anomaly_type}' доминирует: {type_pct:.1f}% всех аномалий",
                        'recommendation': f"Проверьте пороги для детектора '{anomaly_type}'",
                        'affected_metric': f'anomaly_type_{anomaly_type}_pct',
                        'current_value': type_pct,
                        'expected_range': "<70%"
                    })
                    self.logger.info(
                        f"Dominant anomaly type '{anomaly_type}': {type_pct:.1f}% of all anomalies"
                    )
        
        # Log summary
        if warnings:
            self.logger.warning(
                f"Generated {len(warnings)} anomaly count warnings",
                extra={
                    'warning_count': len(warnings),
                    'critical_warnings': len([w for w in warnings if w['severity'] == 'critical']),
                    'warnings': len([w for w in warnings if w['severity'] == 'warning']),
                    'info_warnings': len([w for w in warnings if w['severity'] == 'info'])
                }
            )
        else:
            self.logger.info("No anomaly count warnings - metrics are within expected ranges")
        
        return warnings
    
    def calculate_data_quality_metrics(
        self,
        df: pd.DataFrame,
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate data quality metrics for validation and reporting.
        
        Provides comprehensive metrics about data quality:
        - Data completeness score (percentage of non-missing values)
        - Consistency score (based on logical consistency checks)
        - Missing value statistics
        - Duplicate statistics
        
        These metrics help assess the quality of the input data and identify
        potential data quality issues that may affect anomaly detection.
        
        Args:
            df: DataFrame containing the input data
            validation_results: Optional validation results from DataLoader.validate_data()
                              If not provided, will calculate basic metrics from df
            
        Returns:
            Dictionary containing data quality metrics:
                - data_completeness_score: Overall completeness (0-1)
                - completeness_by_indicator: Completeness per indicator
                - consistency_score: Overall consistency (0-1)
                - missing_value_stats: Statistics about missing values
                - duplicate_stats: Statistics about duplicates
                - quality_grade: Overall quality grade (A-F)
                - quality_issues: List of identified quality issues
        """
        self.logger.info("Calculating data quality metrics")
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided for quality metrics")
            return {
                'data_completeness_score': 0.0,
                'completeness_by_indicator': {},
                'consistency_score': 0.0,
                'missing_value_stats': {},
                'duplicate_stats': {},
                'quality_grade': 'F',
                'quality_issues': ['No data available']
            }
        
        quality_metrics = {}
        quality_issues = []
        
        # 1. Calculate data completeness score
        total_cells = df.shape[0] * df.shape[1]
        non_missing_cells = df.notna().sum().sum()
        data_completeness_score = non_missing_cells / total_cells if total_cells > 0 else 0.0
        
        quality_metrics['data_completeness_score'] = round(data_completeness_score, 4)
        
        self.logger.info(f"Overall data completeness: {data_completeness_score:.2%}")
        
        # Flag if completeness is low
        if data_completeness_score < 0.7:
            quality_issues.append(f"Low data completeness: {data_completeness_score:.1%}")
        
        # 2. Calculate completeness by indicator
        # Identify indicator columns (exclude metadata columns)
        metadata_columns = ['territory_id', 'municipal_name', 'municipal_district_name_short', 
                           'region_name', 'oktmo', 'municipality_type']
        indicator_columns = [col for col in df.columns if col not in metadata_columns]
        
        completeness_by_indicator = {}
        low_completeness_indicators = []
        
        for col in indicator_columns:
            if col in df.columns:
                completeness = df[col].notna().sum() / len(df) if len(df) > 0 else 0.0
                completeness_by_indicator[col] = round(completeness, 4)
                
                # Flag indicators with low completeness
                if completeness < 0.5:
                    low_completeness_indicators.append(col)
        
        quality_metrics['completeness_by_indicator'] = completeness_by_indicator
        
        if low_completeness_indicators:
            quality_issues.append(
                f"{len(low_completeness_indicators)} indicators with <50% completeness"
            )
            self.logger.warning(
                f"Found {len(low_completeness_indicators)} indicators with <50% completeness",
                extra={
                    'low_completeness_indicators': low_completeness_indicators[:5],
                    'data_quality_issue': 'low_completeness'
                }
            )
        
        # 3. Calculate missing value statistics
        missing_value_stats = {}
        
        if validation_results and 'missing_values' in validation_results:
            # Use validation results if available
            missing_values = validation_results['missing_values']
            total_missing = sum(missing_values.values())
            
            missing_value_stats = {
                'total_missing_values': total_missing,
                'columns_with_missing': len(missing_values),
                'missing_percentage': round((total_missing / total_cells * 100) if total_cells > 0 else 0, 2),
                'top_missing_columns': dict(sorted(missing_values.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        else:
            # Calculate from DataFrame
            missing_counts = df.isnull().sum()
            missing_values = {col: int(count) for col, count in missing_counts.items() if count > 0}
            total_missing = sum(missing_values.values())
            
            missing_value_stats = {
                'total_missing_values': total_missing,
                'columns_with_missing': len(missing_values),
                'missing_percentage': round((total_missing / total_cells * 100) if total_cells > 0 else 0, 2),
                'top_missing_columns': dict(sorted(missing_values.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        
        quality_metrics['missing_value_stats'] = missing_value_stats
        
        self.logger.info(
            f"Missing values: {missing_value_stats['total_missing_values']} "
            f"({missing_value_stats['missing_percentage']}%)"
        )
        
        # 4. Calculate duplicate statistics
        duplicate_stats = {}
        
        if 'territory_id' in df.columns:
            duplicate_count = df.duplicated(subset=['territory_id'], keep=False).sum()
            affected_territories = df[df.duplicated(subset=['territory_id'], keep=False)]['territory_id'].nunique()
            
            duplicate_stats = {
                'duplicate_records': int(duplicate_count),
                'affected_territories': int(affected_territories),
                'duplicate_percentage': round((duplicate_count / len(df) * 100) if len(df) > 0 else 0, 2)
            }
            
            if duplicate_count > 0:
                quality_issues.append(
                    f"{duplicate_count} duplicate records affecting {affected_territories} territories"
                )
                self.logger.warning(
                    f"Found {duplicate_count} duplicate records",
                    extra={
                        'duplicate_count': duplicate_count,
                        'affected_territories': affected_territories,
                        'data_quality_issue': 'duplicates'
                    }
                )
        else:
            duplicate_stats = {
                'duplicate_records': 0,
                'affected_territories': 0,
                'duplicate_percentage': 0.0
            }
            quality_issues.append("No territory_id column for duplicate detection")
        
        quality_metrics['duplicate_stats'] = duplicate_stats
        
        # 5. Calculate consistency score
        # Consistency is based on:
        # - Absence of duplicates (weight: 0.3)
        # - High completeness (weight: 0.4)
        # - Logical consistency (weight: 0.3)
        
        # Duplicate score (1.0 if no duplicates, decreases with more duplicates)
        duplicate_score = 1.0
        if 'territory_id' in df.columns and len(df) > 0:
            duplicate_rate = duplicate_stats['duplicate_records'] / len(df)
            duplicate_score = max(0.0, 1.0 - duplicate_rate)
        
        # Completeness score (already calculated)
        completeness_score = data_completeness_score
        
        # Logical consistency score
        # Check for impossible values (negative values where they shouldn't be)
        logical_consistency_score = 1.0
        logical_issues = 0
        
        for col in indicator_columns:
            if col in df.columns:
                # Check for negative values in indicators that should be positive
                # (population, salary, consumption, etc.)
                if any(keyword in col.lower() for keyword in ['population', 'salary', 'consumption', 'connection']):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        logical_issues += negative_count
                        self.logger.warning(
                            f"Found {negative_count} negative values in {col}",
                            extra={
                                'column': col,
                                'negative_count': int(negative_count),
                                'data_quality_issue': 'negative_values'
                            }
                        )
        
        # Calculate logical consistency score
        if len(df) > 0:
            logical_issue_rate = logical_issues / (len(df) * len(indicator_columns))
            logical_consistency_score = max(0.0, 1.0 - logical_issue_rate * 10)  # Scale up the impact
        
        if logical_issues > 0:
            quality_issues.append(f"{logical_issues} logical inconsistencies (negative values)")
        
        # Calculate overall consistency score (weighted average)
        consistency_score = (
            duplicate_score * 0.3 +
            completeness_score * 0.4 +
            logical_consistency_score * 0.3
        )
        
        quality_metrics['consistency_score'] = round(consistency_score, 4)
        
        self.logger.info(
            f"Consistency score: {consistency_score:.2%} "
            f"(duplicate: {duplicate_score:.2%}, completeness: {completeness_score:.2%}, "
            f"logical: {logical_consistency_score:.2%})"
        )
        
        # 6. Determine overall quality grade
        # Grade based on combined completeness and consistency scores
        overall_score = (data_completeness_score + consistency_score) / 2
        
        if overall_score >= 0.95:
            quality_grade = 'A'
        elif overall_score >= 0.85:
            quality_grade = 'B'
        elif overall_score >= 0.75:
            quality_grade = 'C'
        elif overall_score >= 0.65:
            quality_grade = 'D'
        elif overall_score >= 0.50:
            quality_grade = 'E'
        else:
            quality_grade = 'F'
        
        quality_metrics['quality_grade'] = quality_grade
        quality_metrics['quality_issues'] = quality_issues
        
        self.logger.info(
            f"Overall data quality grade: {quality_grade} (score: {overall_score:.2%})",
            extra={
                'quality_grade': quality_grade,
                'overall_score': round(overall_score, 4),
                'completeness_score': round(data_completeness_score, 4),
                'consistency_score': round(consistency_score, 4),
                'quality_issues_count': len(quality_issues)
            }
        )
        
        # 7. Add detailed breakdown
        quality_metrics['score_breakdown'] = {
            'overall_score': round(overall_score, 4),
            'completeness_component': round(data_completeness_score, 4),
            'consistency_component': round(consistency_score, 4),
            'duplicate_score': round(duplicate_score, 4),
            'logical_consistency_score': round(logical_consistency_score, 4)
        }
        
        return quality_metrics
