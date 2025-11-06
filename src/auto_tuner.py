"""
Auto-Tuner Module for СберИндекс Anomaly Detection System

This module provides automatic threshold optimization to minimize false positives
and improve detection quality. Supports multiple optimization strategies and
periodic re-tuning based on historical detection results.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

import pandas as pd
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class ThresholdOptimizationResult:
    """Result of threshold optimization for a single detector."""
    detector_name: str
    original_thresholds: Dict[str, float]
    optimized_thresholds: Dict[str, float]
    estimated_fpr_before: float
    estimated_fpr_after: float
    anomaly_count_before: int
    anomaly_count_after: int
    optimization_strategy: str
    optimization_time: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0


@dataclass
class TuningHistory:
    """Historical record of tuning operations."""
    tuning_id: str
    timestamp: datetime
    results: List[ThresholdOptimizationResult]
    total_anomalies_before: int
    total_anomalies_after: int
    avg_fpr_before: float
    avg_fpr_after: float


class AutoTuner:
    """
    Automatic threshold tuner for anomaly detection system.
    
    Analyzes historical detection results and optimizes thresholds to:
    - Minimize false positive rate
    - Maintain detection of true anomalies
    - Balance sensitivity across different detector types
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the auto-tuner with configuration.
        
        Args:
            config: Configuration dictionary containing auto-tuning parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AutoTuner")
        
        # Get auto-tuning configuration
        auto_tuning_config = config.get('auto_tuning', {})
        
        # Target false positive rate (default: 5%)
        self.target_fpr = auto_tuning_config.get('target_false_positive_rate', 0.05)
        
        # Minimum and maximum anomalies per detector
        self.min_anomalies = auto_tuning_config.get('min_anomalies_per_detector', 10)
        self.max_anomalies = auto_tuning_config.get('max_anomalies_per_detector', 1000)
        
        # Re-tuning interval in days
        self.retuning_interval_days = auto_tuning_config.get('retuning_interval_days', 30)
        
        # Optimization strategy
        self.default_strategy = auto_tuning_config.get('optimization_strategy', 'adaptive')
        
        # Tuning history
        self.tuning_history: List[TuningHistory] = []
        
        # Historical anomaly results cache
        self.historical_results: Optional[pd.DataFrame] = None
        
        # Load historical tuning data if available
        self._load_tuning_history()
        
        self.logger.info(
            f"AutoTuner initialized: target_fpr={self.target_fpr}, "
            f"strategy={self.default_strategy}"
        )
    
    def optimize_thresholds(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, Dict[str, float]],
        strategy: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize detection thresholds based on data distribution.
        
        Args:
            df: DataFrame containing municipal data
            current_thresholds: Current threshold configuration
            strategy: Optimization strategy ('conservative', 'balanced', 'adaptive')
            
        Returns:
            Dictionary of optimized thresholds for each detector
        """
        if strategy is None:
            strategy = self.default_strategy
        
        self.logger.info(f"Starting threshold optimization with strategy: {strategy}")
        
        optimized_thresholds = {}
        optimization_results = []
        
        # Optimize statistical detector thresholds
        if 'statistical' in current_thresholds:
            result = self._optimize_statistical_thresholds(
                df, current_thresholds['statistical'], strategy
            )
            optimized_thresholds['statistical'] = result.optimized_thresholds
            optimization_results.append(result)
            self.logger.info(
                f"Statistical thresholds optimized: "
                f"FPR {result.estimated_fpr_before:.3f} -> {result.estimated_fpr_after:.3f}"
            )
        
        # Optimize geographic detector thresholds
        if 'geographic' in current_thresholds:
            result = self._optimize_geographic_thresholds(
                df, current_thresholds['geographic'], strategy
            )
            optimized_thresholds['geographic'] = result.optimized_thresholds
            optimization_results.append(result)
            self.logger.info(
                f"Geographic thresholds optimized: "
                f"FPR {result.estimated_fpr_before:.3f} -> {result.estimated_fpr_after:.3f}"
            )
        
        # Optimize temporal detector thresholds
        if 'temporal' in current_thresholds:
            result = self._optimize_temporal_thresholds(
                df, current_thresholds['temporal'], strategy
            )
            optimized_thresholds['temporal'] = result.optimized_thresholds
            optimization_results.append(result)
            self.logger.info(
                f"Temporal thresholds optimized: "
                f"FPR {result.estimated_fpr_before:.3f} -> {result.estimated_fpr_after:.3f}"
            )
        
        # Optimize cross-source detector thresholds
        if 'cross_source' in current_thresholds:
            result = self._optimize_cross_source_thresholds(
                df, current_thresholds['cross_source'], strategy
            )
            optimized_thresholds['cross_source'] = result.optimized_thresholds
            optimization_results.append(result)
            self.logger.info(
                f"Cross-source thresholds optimized: "
                f"FPR {result.estimated_fpr_before:.3f} -> {result.estimated_fpr_after:.3f}"
            )
        
        # Store tuning history
        self._save_tuning_results(optimization_results)
        
        self.logger.info("Threshold optimization completed")
        
        return optimized_thresholds
    
    def calculate_fpr_from_historical_results(
        self,
        historical_anomalies: pd.DataFrame,
        total_municipalities: int
    ) -> Dict[str, float]:
        """
        Calculate false positive rate for each detector from historical results.
        
        This method analyzes historical anomaly detection results to estimate
        the actual false positive rate. It assumes that anomalies affecting
        a large percentage of municipalities are likely false positives.
        
        Args:
            historical_anomalies: DataFrame with historical anomaly detections
                                 Must contain columns: detector_name, territory_id, severity_score
            total_municipalities: Total number of municipalities in dataset
            
        Returns:
            Dictionary mapping detector names to estimated FPR values
        """
        self.logger.info("Calculating FPR from historical detection results")
        
        if historical_anomalies.empty:
            self.logger.warning("No historical anomalies provided, cannot calculate FPR")
            return {}
        
        fpr_by_detector = {}
        
        # Group by detector
        for detector_name, group in historical_anomalies.groupby('detector_name'):
            # Count unique municipalities flagged
            flagged_municipalities = group['territory_id'].nunique()
            
            # Calculate raw detection rate
            detection_rate = flagged_municipalities / total_municipalities
            
            # Estimate FPR based on detection patterns
            # High detection rates (>20%) likely indicate high FPR
            # Low severity scores also indicate potential false positives
            
            # Calculate percentage of low-severity anomalies (severity < 60)
            low_severity_pct = (group['severity_score'] < 60).sum() / len(group) if len(group) > 0 else 0
            
            # Estimate FPR using heuristics:
            # - If >50% of municipalities flagged -> likely high FPR
            # - If >50% are low severity -> likely false positives
            if detection_rate > 0.5:
                # Very high detection rate - likely many false positives
                estimated_fpr = min(detection_rate * 0.8, 0.9)
            elif detection_rate > 0.2:
                # Moderate detection rate - some false positives
                estimated_fpr = detection_rate * 0.6
            else:
                # Low detection rate - fewer false positives
                estimated_fpr = detection_rate * 0.3
            
            # Adjust based on severity distribution
            estimated_fpr = estimated_fpr * (0.5 + low_severity_pct * 0.5)
            
            fpr_by_detector[detector_name] = estimated_fpr
            
            self.logger.info(
                f"{detector_name}: detection_rate={detection_rate:.3f}, "
                f"low_severity_pct={low_severity_pct:.3f}, "
                f"estimated_fpr={estimated_fpr:.3f}"
            )
        
        return fpr_by_detector
    
    def calculate_fpr_by_threshold_sweep(
        self,
        df: pd.DataFrame,
        detector_name: str,
        threshold_param: str,
        threshold_range: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate FPR across a range of threshold values using threshold sweep.
        
        This method simulates detection at different threshold levels to find
        the optimal threshold that minimizes FPR while maintaining detection capability.
        
        Args:
            df: DataFrame containing municipal data
            detector_name: Name of detector to analyze
            threshold_param: Name of threshold parameter to sweep
            threshold_range: Array of threshold values to test
            
        Returns:
            Tuple of (threshold_values, fpr_values)
        """
        self.logger.debug(
            f"Performing threshold sweep for {detector_name}.{threshold_param}"
        )
        
        fpr_values = []
        
        # Get numeric columns for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not indicator_cols:
            self.logger.warning("No indicator columns found for threshold sweep")
            return threshold_range, np.zeros_like(threshold_range)
        
        # For each threshold value, estimate detection rate
        for threshold in threshold_range:
            if detector_name == 'statistical':
                # Calculate z-scores and count outliers
                total_outliers = 0
                total_values = 0
                
                for col in indicator_cols[:10]:  # Sample first 10 indicators
                    values = df[col].dropna()
                    if len(values) < 10:
                        continue
                    
                    z_scores = np.abs(stats.zscore(values))
                    outliers = (z_scores > threshold).sum()
                    
                    total_outliers += outliers
                    total_values += len(values)
                
                # FPR is the proportion of values flagged as outliers
                fpr = total_outliers / total_values if total_values > 0 else 0
            
            elif detector_name == 'geographic':
                # Estimate based on regional variation
                if 'region_name' not in df.columns:
                    fpr = 0.05  # Default
                else:
                    total_outliers = 0
                    total_municipalities = 0
                    
                    for col in indicator_cols[:5]:  # Sample first 5 indicators
                        values = df[col].dropna()
                        if len(values) < 10:
                            continue
                        
                        # Group by region and calculate outliers
                        for region, group_df in df.groupby('region_name'):
                            region_values = group_df[col].dropna()
                            if len(region_values) < 3:
                                continue
                            
                            median_val = region_values.median()
                            mad_val = np.median(np.abs(region_values - median_val))
                            
                            if mad_val > 0:
                                robust_z = np.abs((region_values - median_val) / (1.4826 * mad_val))
                                outliers = (robust_z > threshold).sum()
                                
                                total_outliers += outliers
                                total_municipalities += len(region_values)
                    
                    fpr = total_outliers / total_municipalities if total_municipalities > 0 else 0
            
            else:
                # For other detectors, use theoretical estimate
                fpr = 2 * (1 - stats.norm.cdf(threshold))
            
            fpr_values.append(fpr)
        
        return threshold_range, np.array(fpr_values)
    
    def identify_optimal_threshold(
        self,
        threshold_range: np.ndarray,
        fpr_values: np.ndarray,
        target_fpr: Optional[float] = None
    ) -> float:
        """
        Identify optimal threshold value that achieves target FPR.
        
        Args:
            threshold_range: Array of threshold values tested
            fpr_values: Array of corresponding FPR values
            target_fpr: Target false positive rate (uses self.target_fpr if None)
            
        Returns:
            Optimal threshold value
        """
        if target_fpr is None:
            target_fpr = self.target_fpr
        
        # Find threshold closest to target FPR
        idx = np.argmin(np.abs(fpr_values - target_fpr))
        optimal_threshold = threshold_range[idx]
        achieved_fpr = fpr_values[idx]
        
        self.logger.info(
            f"Optimal threshold: {optimal_threshold:.3f} "
            f"(target_fpr={target_fpr:.3f}, achieved_fpr={achieved_fpr:.3f})"
        )
        
        return optimal_threshold
    
    def load_historical_results(self, results_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical anomaly detection results from file.
        
        Args:
            results_file: Path to historical results CSV file
                         If None, looks for latest results in output directory
            
        Returns:
            DataFrame with historical anomaly results
        """
        if results_file is None:
            # Look for latest results file in output directory
            output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
            
            if not output_dir.exists():
                self.logger.warning(f"Output directory {output_dir} does not exist")
                return pd.DataFrame()
            
            # Find most recent anomalies file
            anomaly_files = sorted(output_dir.glob('anomalies_*.csv'), reverse=True)
            
            if not anomaly_files:
                self.logger.warning("No historical anomaly files found")
                return pd.DataFrame()
            
            results_file = anomaly_files[0]
            self.logger.info(f"Loading historical results from {results_file}")
        
        try:
            df = pd.read_csv(results_file, encoding='utf-8')
            
            # Validate required columns
            required_cols = ['detector_name', 'territory_id', 'severity_score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Historical results missing columns: {missing_cols}")
                return pd.DataFrame()
            
            self.historical_results = df
            self.logger.info(f"Loaded {len(df)} historical anomaly records")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to load historical results: {e}")
            return pd.DataFrame()
    
    def analyze_historical_fpr(
        self,
        df: pd.DataFrame,
        results_file: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze historical detection results to calculate FPR for each detector.
        
        This method loads historical anomaly results and calculates:
        - False positive rate estimate
        - Detection rate (% of municipalities flagged)
        - Severity distribution
        - Recommendations for threshold adjustment
        
        Args:
            df: Current DataFrame with municipal data
            results_file: Optional path to historical results file
            
        Returns:
            Dictionary with FPR analysis for each detector
        """
        self.logger.info("Analyzing historical FPR")
        
        # Load historical results
        historical_df = self.load_historical_results(results_file)
        
        if historical_df.empty:
            self.logger.warning("No historical results available for FPR analysis")
            return {}
        
        total_municipalities = len(df)
        
        # Calculate FPR for each detector
        fpr_by_detector = self.calculate_fpr_from_historical_results(
            historical_df, total_municipalities
        )
        
        # Build detailed analysis
        analysis = {}
        
        for detector_name, group in historical_df.groupby('detector_name'):
            fpr = fpr_by_detector.get(detector_name, 0.0)
            
            # Calculate statistics
            flagged_municipalities = group['territory_id'].nunique()
            detection_rate = flagged_municipalities / total_municipalities
            avg_severity = group['severity_score'].mean()
            
            # Severity distribution
            severity_dist = {
                'critical': (group['severity_score'] >= 90).sum(),
                'high': ((group['severity_score'] >= 70) & (group['severity_score'] < 90)).sum(),
                'medium': ((group['severity_score'] >= 50) & (group['severity_score'] < 70)).sum(),
                'low': (group['severity_score'] < 50).sum()
            }
            
            # Generate recommendation
            if fpr > self.target_fpr * 2:
                recommendation = "Increase thresholds significantly to reduce false positives"
                adjustment = "increase_high"
            elif fpr > self.target_fpr * 1.2:
                recommendation = "Increase thresholds moderately"
                adjustment = "increase_moderate"
            elif fpr < self.target_fpr * 0.5:
                recommendation = "Consider decreasing thresholds to improve detection"
                adjustment = "decrease"
            else:
                recommendation = "Thresholds are well-calibrated"
                adjustment = "maintain"
            
            analysis[detector_name] = {
                'estimated_fpr': fpr,
                'detection_rate': detection_rate,
                'flagged_municipalities': flagged_municipalities,
                'total_anomalies': len(group),
                'avg_severity': avg_severity,
                'severity_distribution': severity_dist,
                'recommendation': recommendation,
                'adjustment': adjustment,
                'meets_target': fpr <= self.target_fpr * 1.2
            }
            
            self.logger.info(
                f"{detector_name}: FPR={fpr:.3f}, "
                f"detection_rate={detection_rate:.3f}, "
                f"recommendation={adjustment}"
            )
        
        return analysis
    
    def _optimize_statistical_thresholds(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, float],
        strategy: str
    ) -> ThresholdOptimizationResult:
        """
        Optimize thresholds for statistical outlier detector.
        
        Args:
            df: DataFrame containing municipal data
            current_thresholds: Current statistical thresholds
            strategy: Optimization strategy
            
        Returns:
            ThresholdOptimizationResult with optimized values
        """
        self.logger.debug("Optimizing statistical detector thresholds")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate data characteristics
        skewness_values = []
        outlier_percentages = []
        
        for col in indicator_cols:
            values = df[col].dropna()
            if len(values) < 10:
                continue
            
            # Calculate skewness
            skew = stats.skew(values)
            skewness_values.append(abs(skew))
            
            # Calculate percentage of potential outliers with current threshold
            z_scores = np.abs(stats.zscore(values))
            outlier_pct = (z_scores > current_thresholds.get('z_score', 3.0)).sum() / len(values)
            outlier_percentages.append(outlier_pct)
        
        avg_skewness = np.mean(skewness_values) if skewness_values else 0
        avg_outlier_pct = np.mean(outlier_percentages) if outlier_percentages else 0
        
        # Calculate actual FPR using threshold sweep
        current_z_threshold = current_thresholds.get('z_score', 3.0)
        
        # Perform threshold sweep to find optimal value
        if strategy == 'adaptive' and len(indicator_cols) > 0:
            # Use threshold sweep for adaptive strategy
            threshold_range = np.linspace(2.0, 4.5, 26)
            _, fpr_values = self.calculate_fpr_by_threshold_sweep(
                df, 'statistical', 'z_score', threshold_range
            )
            
            # Find current FPR from sweep
            current_idx = np.argmin(np.abs(threshold_range - current_z_threshold))
            estimated_fpr_before = fpr_values[current_idx]
            
            # Find optimal threshold
            optimal_z_score = self.identify_optimal_threshold(
                threshold_range, fpr_values, self.target_fpr
            )
        else:
            # Use theoretical estimate for other strategies
            estimated_fpr_before = 2 * (1 - stats.norm.cdf(current_z_threshold))
            optimal_z_score = None
        
        # Optimize based on strategy
        optimized = current_thresholds.copy()
        
        if strategy == 'conservative':
            # Increase thresholds to reduce false positives
            optimized['z_score'] = min(current_z_threshold * 1.2, 4.0)
            optimized['iqr_multiplier'] = min(
                current_thresholds.get('iqr_multiplier', 1.5) * 1.3, 2.5
            )
        
        elif strategy == 'balanced':
            # Adjust based on data characteristics
            if avg_skewness > 1.5:
                # Highly skewed data - use more robust methods
                optimized['z_score'] = min(current_z_threshold * 1.1, 3.5)
                optimized['iqr_multiplier'] = current_thresholds.get('iqr_multiplier', 1.5) * 1.2
            else:
                # Normal-ish data - moderate adjustment
                optimized['z_score'] = current_z_threshold * 1.05
                optimized['iqr_multiplier'] = current_thresholds.get('iqr_multiplier', 1.5) * 1.1
        
        elif strategy == 'adaptive':
            # Use optimal threshold from sweep if available
            if optimal_z_score is not None:
                optimized['z_score'] = optimal_z_score
                # Adjust IQR multiplier proportionally
                ratio = optimal_z_score / current_z_threshold
                optimized['iqr_multiplier'] = min(
                    current_thresholds.get('iqr_multiplier', 1.5) * ratio, 3.0
                )
            else:
                # Fallback to heuristic adjustment
                if avg_outlier_pct > self.target_fpr * 2:
                    # Too many outliers - increase threshold
                    adjustment_factor = min(avg_outlier_pct / (self.target_fpr * 2), 1.5)
                    optimized['z_score'] = min(current_z_threshold * adjustment_factor, 4.5)
                    optimized['iqr_multiplier'] = min(
                        current_thresholds.get('iqr_multiplier', 1.5) * adjustment_factor, 3.0
                    )
                elif avg_outlier_pct < self.target_fpr * 0.5:
                    # Too few outliers - decrease threshold slightly
                    adjustment_factor = max(avg_outlier_pct / (self.target_fpr * 0.5), 0.9)
                    optimized['z_score'] = max(current_z_threshold * adjustment_factor, 2.0)
                    optimized['iqr_multiplier'] = max(
                        current_thresholds.get('iqr_multiplier', 1.5) * adjustment_factor, 1.0
                    )
        
        # Calculate new FPR with optimized threshold
        if strategy == 'adaptive' and optimal_z_score is not None:
            # Use actual FPR from sweep
            optimal_idx = np.argmin(np.abs(threshold_range - optimized['z_score']))
            estimated_fpr_after = fpr_values[optimal_idx]
        else:
            # Use theoretical estimate
            estimated_fpr_after = 2 * (1 - stats.norm.cdf(optimized['z_score']))
        
        # Estimate anomaly counts
        total_values = sum(len(df[col].dropna()) for col in indicator_cols)
        anomaly_count_before = int(total_values * estimated_fpr_before)
        anomaly_count_after = int(total_values * estimated_fpr_after)
        
        return ThresholdOptimizationResult(
            detector_name='statistical',
            original_thresholds=current_thresholds,
            optimized_thresholds=optimized,
            estimated_fpr_before=estimated_fpr_before,
            estimated_fpr_after=estimated_fpr_after,
            anomaly_count_before=anomaly_count_before,
            anomaly_count_after=anomaly_count_after,
            optimization_strategy=strategy,
            confidence_score=0.8 if len(indicator_cols) > 10 else 0.6
        )
    
    def _optimize_geographic_thresholds(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, float],
        strategy: str
    ) -> ThresholdOptimizationResult:
        """
        Optimize thresholds for geographic anomaly detector.
        
        Args:
            df: DataFrame containing municipal data
            current_thresholds: Current geographic thresholds
            strategy: Optimization strategy
            
        Returns:
            ThresholdOptimizationResult with optimized values
        """
        self.logger.debug("Optimizing geographic detector thresholds")
        
        # Analyze regional variation
        if 'region_name' not in df.columns:
            # Cannot optimize without regional data
            return ThresholdOptimizationResult(
                detector_name='geographic',
                original_thresholds=current_thresholds,
                optimized_thresholds=current_thresholds,
                estimated_fpr_before=0.05,
                estimated_fpr_after=0.05,
                anomaly_count_before=0,
                anomaly_count_after=0,
                optimization_strategy=strategy,
                confidence_score=0.0
            )
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate inter-regional variation
        regional_variations = []
        for col in indicator_cols[:10]:  # Sample first 10 indicators
            values = df[col].dropna()
            if len(values) < 10:
                continue
            
            # Calculate coefficient of variation by region
            regional_cv = df.groupby('region_name')[col].apply(
                lambda x: x.std() / x.mean() if x.mean() != 0 and len(x) > 2 else 0
            )
            regional_variations.append(regional_cv.mean())
        
        avg_regional_cv = np.mean(regional_variations) if regional_variations else 0.5
        
        # Current threshold
        current_z_threshold = current_thresholds.get('regional_z_score', 2.0)
        estimated_fpr_before = 2 * (1 - stats.norm.cdf(current_z_threshold))
        
        # Optimize based on strategy and regional variation
        optimized = current_thresholds.copy()
        
        if strategy == 'conservative':
            # Increase thresholds to account for natural regional variation
            optimized['regional_z_score'] = min(current_z_threshold * 1.3, 3.5)
            optimized['cluster_threshold'] = min(
                current_thresholds.get('cluster_threshold', 2.5) * 1.2, 3.5
            )
        
        elif strategy == 'balanced':
            # Adjust based on regional variation
            if avg_regional_cv > 0.5:
                # High regional variation - relax thresholds
                optimized['regional_z_score'] = min(current_z_threshold * 1.2, 3.0)
                optimized['cluster_threshold'] = min(
                    current_thresholds.get('cluster_threshold', 2.5) * 1.15, 3.0
                )
            else:
                # Low regional variation - moderate adjustment
                optimized['regional_z_score'] = current_z_threshold * 1.1
                optimized['cluster_threshold'] = current_thresholds.get('cluster_threshold', 2.5) * 1.05
        
        elif strategy == 'adaptive':
            # Adapt to regional characteristics
            if avg_regional_cv > 0.7:
                # Very high variation - significantly relax
                optimized['regional_z_score'] = min(current_z_threshold * 1.4, 4.0)
                optimized['cluster_threshold'] = min(
                    current_thresholds.get('cluster_threshold', 2.5) * 1.3, 3.5
                )
            elif avg_regional_cv < 0.3:
                # Low variation - can be more strict
                optimized['regional_z_score'] = max(current_z_threshold * 0.95, 1.5)
                optimized['cluster_threshold'] = max(
                    current_thresholds.get('cluster_threshold', 2.5) * 0.95, 2.0
                )
            else:
                # Moderate variation - slight adjustment
                optimized['regional_z_score'] = current_z_threshold * 1.1
                optimized['cluster_threshold'] = current_thresholds.get('cluster_threshold', 2.5) * 1.1
        
        # Estimate new FPR
        estimated_fpr_after = 2 * (1 - stats.norm.cdf(optimized['regional_z_score']))
        
        # Estimate anomaly counts (rough approximation)
        n_municipalities = len(df)
        n_indicators = len(indicator_cols)
        anomaly_count_before = int(n_municipalities * n_indicators * estimated_fpr_before * 0.3)
        anomaly_count_after = int(n_municipalities * n_indicators * estimated_fpr_after * 0.3)
        
        return ThresholdOptimizationResult(
            detector_name='geographic',
            original_thresholds=current_thresholds,
            optimized_thresholds=optimized,
            estimated_fpr_before=estimated_fpr_before,
            estimated_fpr_after=estimated_fpr_after,
            anomaly_count_before=anomaly_count_before,
            anomaly_count_after=anomaly_count_after,
            optimization_strategy=strategy,
            confidence_score=0.7 if len(regional_variations) > 5 else 0.5
        )
    
    def _optimize_temporal_thresholds(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, float],
        strategy: str
    ) -> ThresholdOptimizationResult:
        """
        Optimize thresholds for temporal anomaly detector.
        
        Args:
            df: DataFrame containing municipal data
            current_thresholds: Current temporal thresholds
            strategy: Optimization strategy
            
        Returns:
            ThresholdOptimizationResult with optimized values
        """
        self.logger.debug("Optimizing temporal detector thresholds")
        
        # Temporal optimization requires time-series data
        # For now, use conservative adjustments based on strategy
        
        optimized = current_thresholds.copy()
        
        if strategy == 'conservative':
            # Increase thresholds to reduce false positives
            optimized['spike_threshold'] = current_thresholds.get('spike_threshold', 100) * 1.3
            optimized['drop_threshold'] = current_thresholds.get('drop_threshold', -50) * 1.3
            optimized['volatility_multiplier'] = min(
                current_thresholds.get('volatility_multiplier', 2.0) * 1.2, 3.0
            )
        
        elif strategy == 'balanced':
            # Moderate adjustment
            optimized['spike_threshold'] = current_thresholds.get('spike_threshold', 100) * 1.15
            optimized['drop_threshold'] = current_thresholds.get('drop_threshold', -50) * 1.15
            optimized['volatility_multiplier'] = current_thresholds.get('volatility_multiplier', 2.0) * 1.1
        
        elif strategy == 'adaptive':
            # Slight adjustment for adaptive strategy
            optimized['spike_threshold'] = current_thresholds.get('spike_threshold', 100) * 1.1
            optimized['drop_threshold'] = current_thresholds.get('drop_threshold', -50) * 1.1
            optimized['volatility_multiplier'] = current_thresholds.get('volatility_multiplier', 2.0) * 1.05
        
        # Estimate FPR (rough approximation for temporal anomalies)
        estimated_fpr_before = 0.02
        estimated_fpr_after = estimated_fpr_before * 0.8  # Assume 20% reduction
        
        return ThresholdOptimizationResult(
            detector_name='temporal',
            original_thresholds=current_thresholds,
            optimized_thresholds=optimized,
            estimated_fpr_before=estimated_fpr_before,
            estimated_fpr_after=estimated_fpr_after,
            anomaly_count_before=0,
            anomaly_count_after=0,
            optimization_strategy=strategy,
            confidence_score=0.5  # Lower confidence without time-series analysis
        )
    
    def _optimize_cross_source_thresholds(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, float],
        strategy: str
    ) -> ThresholdOptimizationResult:
        """
        Optimize thresholds for cross-source comparator.
        
        Args:
            df: DataFrame containing municipal data
            current_thresholds: Current cross-source thresholds
            strategy: Optimization strategy
            
        Returns:
            ThresholdOptimizationResult with optimized values
        """
        self.logger.debug("Optimizing cross-source detector thresholds")
        
        optimized = current_thresholds.copy()
        
        if strategy == 'conservative':
            # Increase thresholds to reduce false positives
            optimized['discrepancy_threshold'] = min(
                current_thresholds.get('discrepancy_threshold', 50) * 1.3, 80
            )
            optimized['correlation_threshold'] = max(
                current_thresholds.get('correlation_threshold', 0.5) * 0.85, 0.3
            )
        
        elif strategy == 'balanced':
            # Moderate adjustment
            optimized['discrepancy_threshold'] = current_thresholds.get('discrepancy_threshold', 50) * 1.15
            optimized['correlation_threshold'] = current_thresholds.get('correlation_threshold', 0.5) * 0.92
        
        elif strategy == 'adaptive':
            # Slight adjustment
            optimized['discrepancy_threshold'] = current_thresholds.get('discrepancy_threshold', 50) * 1.1
            optimized['correlation_threshold'] = current_thresholds.get('correlation_threshold', 0.5) * 0.95
        
        # Estimate FPR (rough approximation)
        estimated_fpr_before = 0.03
        estimated_fpr_after = estimated_fpr_before * 0.75  # Assume 25% reduction
        
        return ThresholdOptimizationResult(
            detector_name='cross_source',
            original_thresholds=current_thresholds,
            optimized_thresholds=optimized,
            estimated_fpr_before=estimated_fpr_before,
            estimated_fpr_after=estimated_fpr_after,
            anomaly_count_before=0,
            anomaly_count_after=0,
            optimization_strategy=strategy,
            confidence_score=0.6
        )
    
    def should_retune(self, force_check: bool = False) -> Tuple[bool, str]:
        """
        Check if re-tuning is needed based on time since last tuning.
        
        Args:
            force_check: If True, performs additional checks beyond time-based criteria
        
        Returns:
            Tuple of (should_retune, reason) where reason explains why re-tuning is recommended
        """
        if not self.tuning_history:
            # Never tuned before
            return True, "No previous tuning history found"
        
        last_tuning = self.tuning_history[-1]
        days_since_tuning = (datetime.now() - last_tuning.timestamp).days
        
        # Check time-based criterion
        if days_since_tuning >= self.retuning_interval_days:
            reason = (
                f"{days_since_tuning} days since last tuning "
                f"(threshold: {self.retuning_interval_days} days)"
            )
            self.logger.info(f"Re-tuning recommended: {reason}")
            return True, reason
        
        # Additional checks if force_check is enabled
        if force_check:
            # Check if FPR has degraded significantly
            if last_tuning.avg_fpr_after > self.target_fpr * 1.5:
                reason = (
                    f"FPR ({last_tuning.avg_fpr_after:.3f}) exceeds target "
                    f"({self.target_fpr:.3f}) by >50%"
                )
                self.logger.info(f"Re-tuning recommended: {reason}")
                return True, reason
            
            # Check if anomaly counts are outside acceptable range
            if len(self.tuning_history) >= 2:
                prev_tuning = self.tuning_history[-2]
                anomaly_change_pct = abs(
                    (last_tuning.total_anomalies_after - prev_tuning.total_anomalies_after) 
                    / max(prev_tuning.total_anomalies_after, 1)
                )
                
                if anomaly_change_pct > 0.5:  # 50% change
                    reason = (
                        f"Anomaly count changed by {anomaly_change_pct*100:.1f}% "
                        f"since previous tuning"
                    )
                    self.logger.info(f"Re-tuning recommended: {reason}")
                    return True, reason
        
        # No re-tuning needed
        days_remaining = self.retuning_interval_days - days_since_tuning
        reason = f"Re-tuning not needed ({days_remaining} days until next scheduled tuning)"
        self.logger.debug(reason)
        return False, reason
    
    def schedule_periodic_retuning(
        self,
        df: pd.DataFrame,
        current_thresholds: Dict[str, Dict[str, float]],
        strategy: Optional[str] = None,
        force: bool = False
    ) -> Tuple[bool, Dict[str, Dict[str, float]], str]:
        """
        Execute periodic re-tuning if needed based on schedule and data conditions.
        
        This method checks if re-tuning is needed and executes optimization if required.
        It's designed to be called periodically (e.g., at the start of each analysis run).
        
        Args:
            df: DataFrame containing current municipal data
            current_thresholds: Current threshold configuration
            strategy: Optimization strategy to use (uses default if None)
            force: If True, forces re-tuning regardless of schedule
            
        Returns:
            Tuple of (was_retuned, thresholds, message) where:
            - was_retuned: True if re-tuning was performed
            - thresholds: Updated thresholds (same as input if not retuned)
            - message: Description of action taken
        """
        self.logger.info("Checking if periodic re-tuning is needed")
        
        # Check if re-tuning is needed
        if force:
            should_tune = True
            reason = "Forced re-tuning requested"
            self.logger.info(reason)
        else:
            should_tune, reason = self.should_retune(force_check=True)
        
        if not should_tune:
            return False, current_thresholds, reason
        
        # Perform re-tuning
        self.logger.info(f"Starting periodic re-tuning: {reason}")
        
        try:
            optimized_thresholds = self.optimize_thresholds(
                df=df,
                current_thresholds=current_thresholds,
                strategy=strategy
            )
            
            message = f"Re-tuning completed successfully: {reason}"
            self.logger.info(message)
            
            return True, optimized_thresholds, message
        
        except Exception as e:
            error_msg = f"Re-tuning failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Return original thresholds on failure
            return False, current_thresholds, error_msg
    
    def get_optimization_strategies(self) -> List[str]:
        """
        Get list of available optimization strategies.
        
        Returns:
            List of strategy names
        """
        return ['conservative', 'balanced', 'adaptive']
    
    def get_tuning_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of tuning history including next scheduled tuning time.
        
        Returns:
            Dictionary with tuning history summary including:
            - total_tunings: Total number of tuning operations performed
            - last_tuning_date: Date of last tuning
            - days_since_last_tuning: Days since last tuning
            - next_scheduled_tuning: Date of next scheduled tuning
            - days_until_next_tuning: Days until next scheduled tuning
            - retuning_interval_days: Configured re-tuning interval
            - tuning_history: List of recent tuning summaries
        """
        summary = {
            'total_tunings': len(self.tuning_history),
            'last_tuning_date': None,
            'days_since_last_tuning': None,
            'next_scheduled_tuning': None,
            'days_until_next_tuning': None,
            'retuning_interval_days': self.retuning_interval_days,
            'tuning_history': []
        }
        
        if not self.tuning_history:
            return summary
        
        # Get last tuning info
        last_tuning = self.tuning_history[-1]
        summary['last_tuning_date'] = last_tuning.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        days_since = (datetime.now() - last_tuning.timestamp).days
        summary['days_since_last_tuning'] = days_since
        
        # Calculate next scheduled tuning
        next_tuning_date = last_tuning.timestamp + timedelta(days=self.retuning_interval_days)
        summary['next_scheduled_tuning'] = next_tuning_date.strftime('%Y-%m-%d %H:%M:%S')
        
        days_until = (next_tuning_date - datetime.now()).days
        summary['days_until_next_tuning'] = max(0, days_until)
        
        # Add recent tuning history (last 5 entries)
        for entry in self.tuning_history[-5:]:
            history_entry = {
                'tuning_id': entry.tuning_id,
                'timestamp': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'total_anomalies_before': entry.total_anomalies_before,
                'total_anomalies_after': entry.total_anomalies_after,
                'anomaly_reduction_pct': (
                    (1 - entry.total_anomalies_after / max(entry.total_anomalies_before, 1)) * 100
                ),
                'avg_fpr_before': entry.avg_fpr_before,
                'avg_fpr_after': entry.avg_fpr_after,
                'fpr_reduction_pct': (
                    (1 - entry.avg_fpr_after / max(entry.avg_fpr_before, 1)) * 100
                ),
                'detectors_tuned': [r.detector_name for r in entry.results]
            }
            summary['tuning_history'].append(history_entry)
        
        return summary
    
    def get_next_tuning_date(self) -> Optional[datetime]:
        """
        Get the date of the next scheduled tuning.
        
        Returns:
            DateTime of next scheduled tuning, or None if never tuned before
        """
        if not self.tuning_history:
            return None
        
        last_tuning = self.tuning_history[-1]
        next_tuning = last_tuning.timestamp + timedelta(days=self.retuning_interval_days)
        
        return next_tuning
    
    def _save_tuning_results(self, results: List[ThresholdOptimizationResult]):
        """
        Save tuning results to history.
        
        Args:
            results: List of optimization results
        """
        total_before = sum(r.anomaly_count_before for r in results)
        total_after = sum(r.anomaly_count_after for r in results)
        avg_fpr_before = np.mean([r.estimated_fpr_before for r in results])
        avg_fpr_after = np.mean([r.estimated_fpr_after for r in results])
        
        history_entry = TuningHistory(
            tuning_id=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            results=results,
            total_anomalies_before=total_before,
            total_anomalies_after=total_after,
            avg_fpr_before=avg_fpr_before,
            avg_fpr_after=avg_fpr_after
        )
        
        self.tuning_history.append(history_entry)
        
        # Persist to file
        self._persist_tuning_history()
        
        self.logger.info(
            f"Tuning results saved: {total_before} -> {total_after} anomalies, "
            f"FPR {avg_fpr_before:.3f} -> {avg_fpr_after:.3f}"
        )
    
    def _load_tuning_history(self):
        """Load historical tuning data from file."""
        history_file = Path(self.config.get('export', {}).get('output_dir', 'output')) / 'tuning_history.json'
        
        if not history_file.exists():
            self.logger.debug("No tuning history file found")
            return
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to TuningHistory objects
            for entry in data:
                # Reconstruct ThresholdOptimizationResult objects
                results = []
                for r in entry.get('results', []):
                    result = ThresholdOptimizationResult(
                        detector_name=r['detector_name'],
                        original_thresholds=r.get('original_thresholds', {}),
                        optimized_thresholds=r.get('optimized_thresholds', {}),
                        estimated_fpr_before=r['estimated_fpr_before'],
                        estimated_fpr_after=r['estimated_fpr_after'],
                        anomaly_count_before=r.get('anomaly_count_before', 0),
                        anomaly_count_after=r.get('anomaly_count_after', 0),
                        optimization_strategy=r['optimization_strategy'],
                        optimization_time=datetime.fromisoformat(entry['timestamp']),
                        confidence_score=r['confidence_score']
                    )
                    results.append(result)
                
                # Create TuningHistory object
                history = TuningHistory(
                    tuning_id=entry['tuning_id'],
                    timestamp=datetime.fromisoformat(entry['timestamp']),
                    results=results,
                    total_anomalies_before=entry['total_anomalies_before'],
                    total_anomalies_after=entry['total_anomalies_after'],
                    avg_fpr_before=entry['avg_fpr_before'],
                    avg_fpr_after=entry['avg_fpr_after']
                )
                self.tuning_history.append(history)
            
            self.logger.info(f"Loaded {len(self.tuning_history)} historical tuning records")
        
        except Exception as e:
            self.logger.warning(f"Failed to load tuning history: {e}")
    
    def _persist_tuning_history(self):
        """Persist tuning history to file with complete threshold information."""
        output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
        output_dir.mkdir(exist_ok=True)
        
        history_file = output_dir / 'tuning_history.json'
        
        try:
            # Convert to serializable format
            history_data = []
            for entry in self.tuning_history[-10:]:  # Keep last 10 entries
                history_data.append({
                    'tuning_id': entry.tuning_id,
                    'timestamp': entry.timestamp.isoformat(),
                    'total_anomalies_before': entry.total_anomalies_before,
                    'total_anomalies_after': entry.total_anomalies_after,
                    'avg_fpr_before': entry.avg_fpr_before,
                    'avg_fpr_after': entry.avg_fpr_after,
                    'results': [
                        {
                            'detector_name': r.detector_name,
                            'original_thresholds': r.original_thresholds,
                            'optimized_thresholds': r.optimized_thresholds,
                            'optimization_strategy': r.optimization_strategy,
                            'estimated_fpr_before': r.estimated_fpr_before,
                            'estimated_fpr_after': r.estimated_fpr_after,
                            'anomaly_count_before': r.anomaly_count_before,
                            'anomaly_count_after': r.anomaly_count_after,
                            'confidence_score': r.confidence_score
                        }
                        for r in entry.results
                    ]
                })
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Tuning history persisted to {history_file} ({len(history_data)} entries)")
        
        except Exception as e:
            self.logger.error(f"Failed to persist tuning history: {e}")
    
    def validate_thresholds(
        self,
        df: pd.DataFrame,
        thresholds: Dict[str, Dict[str, float]],
        detector_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate threshold values to ensure they meet quality criteria.
        
        This method validates that:
        1. At least 95% of normal municipalities are not flagged
        2. Threshold values are within acceptable ranges
        3. Anomaly counts per detector are within min/max bounds
        
        Args:
            df: DataFrame containing municipal data
            thresholds: Threshold configuration to validate
            detector_name: Optional specific detector to validate (validates all if None)
            
        Returns:
            Dictionary with validation results including:
            - is_valid: Overall validation status
            - validation_errors: List of validation errors
            - validation_warnings: List of validation warnings
            - detector_results: Per-detector validation details
        """
        self.logger.info("Validating thresholds")
        
        validation_results = {
            'is_valid': True,
            'validation_errors': [],
            'validation_warnings': [],
            'detector_results': {}
        }
        
        total_municipalities = len(df)
        
        # Handle empty data
        if total_municipalities == 0:
            self.logger.warning("Cannot validate thresholds with empty data")
            validation_results['validation_warnings'].append(
                "Cannot validate thresholds with empty data"
            )
            return validation_results
        
        max_flagged_municipalities = int(total_municipalities * 0.05)  # 5% max (95% not flagged)
        
        # Validate each detector
        detectors_to_validate = [detector_name] if detector_name else thresholds.keys()
        
        for det_name in detectors_to_validate:
            if det_name not in thresholds:
                continue
            
            det_thresholds = thresholds[det_name]
            det_result = {
                'threshold_range_valid': True,
                'anomaly_count_valid': True,
                'municipality_coverage_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Validate threshold ranges
            range_validation = self._validate_threshold_ranges(det_name, det_thresholds)
            det_result['threshold_range_valid'] = range_validation['is_valid']
            det_result['errors'].extend(range_validation['errors'])
            det_result['warnings'].extend(range_validation['warnings'])
            
            # Add range warnings to overall warnings
            validation_results['validation_warnings'].extend(range_validation['warnings'])
            
            # If range validation failed, mark overall as invalid
            if not range_validation['is_valid']:
                validation_results['is_valid'] = False
                validation_results['validation_errors'].extend(range_validation['errors'])
            
            # Estimate anomaly counts with these thresholds
            estimated_anomalies = self._estimate_anomaly_count(df, det_name, det_thresholds)
            estimated_flagged_municipalities = min(estimated_anomalies, total_municipalities)
            
            # Validate anomaly count is within bounds
            if estimated_anomalies < self.min_anomalies:
                det_result['anomaly_count_valid'] = False
                error_msg = (
                    f"{det_name}: Estimated anomaly count ({estimated_anomalies}) "
                    f"is below minimum ({self.min_anomalies}). "
                    f"Thresholds may be too strict."
                )
                det_result['errors'].append(error_msg)
                validation_results['validation_errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            if estimated_anomalies > self.max_anomalies:
                det_result['anomaly_count_valid'] = False
                error_msg = (
                    f"{det_name}: Estimated anomaly count ({estimated_anomalies}) "
                    f"exceeds maximum ({self.max_anomalies}). "
                    f"Thresholds may be too relaxed."
                )
                det_result['errors'].append(error_msg)
                validation_results['validation_errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            # Validate that at least 95% of municipalities are not flagged
            if estimated_flagged_municipalities > max_flagged_municipalities:
                det_result['municipality_coverage_valid'] = False
                flagged_pct = (estimated_flagged_municipalities / total_municipalities) * 100
                error_msg = (
                    f"{det_name}: Estimated {estimated_flagged_municipalities} municipalities "
                    f"({flagged_pct:.1f}%) would be flagged, exceeding 5% threshold. "
                    f"Requirement: at least 95% of normal municipalities not flagged."
                )
                det_result['errors'].append(error_msg)
                validation_results['validation_errors'].append(error_msg)
                validation_results['is_valid'] = False
            
            # Add warnings for borderline cases
            if self.min_anomalies <= estimated_anomalies < self.min_anomalies * 1.5:
                warning_msg = (
                    f"{det_name}: Anomaly count ({estimated_anomalies}) is close to minimum. "
                    f"Consider slightly relaxing thresholds."
                )
                det_result['warnings'].append(warning_msg)
                validation_results['validation_warnings'].append(warning_msg)
            
            if self.max_anomalies * 0.8 < estimated_anomalies <= self.max_anomalies:
                warning_msg = (
                    f"{det_name}: Anomaly count ({estimated_anomalies}) is close to maximum. "
                    f"Consider slightly tightening thresholds."
                )
                det_result['warnings'].append(warning_msg)
                validation_results['validation_warnings'].append(warning_msg)
            
            # Store detector-specific results
            det_result['estimated_anomalies'] = estimated_anomalies
            det_result['estimated_flagged_municipalities'] = estimated_flagged_municipalities
            det_result['flagged_percentage'] = (estimated_flagged_municipalities / total_municipalities) * 100
            validation_results['detector_results'][det_name] = det_result
        
        # Log validation summary
        if validation_results['is_valid']:
            self.logger.info("Threshold validation passed")
        else:
            self.logger.error(
                f"Threshold validation failed with {len(validation_results['validation_errors'])} errors"
            )
            for error in validation_results['validation_errors']:
                self.logger.error(f"  - {error}")
        
        if validation_results['validation_warnings']:
            self.logger.warning(
                f"Threshold validation has {len(validation_results['validation_warnings'])} warnings"
            )
            for warning in validation_results['validation_warnings']:
                self.logger.warning(f"  - {warning}")
        
        return validation_results
    
    def _validate_threshold_ranges(
        self,
        detector_name: str,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate that threshold values are within acceptable ranges.
        
        Args:
            detector_name: Name of the detector
            thresholds: Threshold values to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Define acceptable ranges for each detector type
        threshold_ranges = {
            'statistical': {
                'z_score': (1.5, 5.0),
                'iqr_multiplier': (1.0, 3.5),
                'percentile_lower': (0.1, 5.0),
                'percentile_upper': (95.0, 99.9)
            },
            'geographic': {
                'regional_z_score': (1.0, 4.5),
                'cluster_threshold': (1.5, 4.0),
                'neighbor_threshold': (1.5, 4.0)
            },
            'temporal': {
                'spike_threshold': (50, 300),
                'drop_threshold': (-100, -20),
                'volatility_multiplier': (1.5, 4.0),
                'min_periods': (2, 12)
            },
            'cross_source': {
                'discrepancy_threshold': (20, 100),
                'correlation_threshold': (0.2, 0.8),
                'min_correlation': (0.1, 0.7)
            },
            'logical': {
                'min_value': (-1e10, 1e10),
                'max_value': (-1e10, 1e10),
                'ratio_threshold': (0.1, 10.0)
            }
        }
        
        if detector_name not in threshold_ranges:
            result['warnings'].append(
                f"No validation ranges defined for detector '{detector_name}'"
            )
            return result
        
        ranges = threshold_ranges[detector_name]
        
        for param_name, param_value in thresholds.items():
            if param_name not in ranges:
                # Unknown parameter - just warn
                result['warnings'].append(
                    f"{detector_name}.{param_name}: Unknown parameter (no validation range)"
                )
                continue
            
            min_val, max_val = ranges[param_name]
            
            if param_value < min_val:
                result['is_valid'] = False
                result['errors'].append(
                    f"{detector_name}.{param_name}: Value {param_value} is below minimum {min_val}"
                )
            elif param_value > max_val:
                result['is_valid'] = False
                result['errors'].append(
                    f"{detector_name}.{param_name}: Value {param_value} exceeds maximum {max_val}"
                )
            
            # Warn if close to boundaries
            range_span = max_val - min_val
            if param_value < min_val + range_span * 0.1:
                result['warnings'].append(
                    f"{detector_name}.{param_name}: Value {param_value} is close to minimum {min_val}"
                )
            elif param_value > max_val - range_span * 0.1:
                result['warnings'].append(
                    f"{detector_name}.{param_name}: Value {param_value} is close to maximum {max_val}"
                )
        
        return result
    
    def _estimate_anomaly_count(
        self,
        df: pd.DataFrame,
        detector_name: str,
        thresholds: Dict[str, float]
    ) -> int:
        """
        Estimate the number of anomalies that would be detected with given thresholds.
        
        Args:
            df: DataFrame containing municipal data
            detector_name: Name of the detector
            thresholds: Threshold values to use for estimation
            
        Returns:
            Estimated number of anomalies
        """
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['territory_id', 'oktmo']
        indicator_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not indicator_cols:
            return 0
        
        total_anomalies = 0
        
        if detector_name == 'statistical':
            # Estimate based on z-score threshold
            z_threshold = thresholds.get('z_score', 3.0)
            
            for col in indicator_cols[:10]:  # Sample first 10 indicators
                values = df[col].dropna()
                if len(values) < 10:
                    continue
                
                z_scores = np.abs(stats.zscore(values))
                outliers = (z_scores > z_threshold).sum()
                total_anomalies += outliers
            
            # Extrapolate to all indicators
            if len(indicator_cols) > 10:
                total_anomalies = int(total_anomalies * len(indicator_cols) / 10)
        
        elif detector_name == 'geographic':
            # Estimate based on regional z-score
            if 'region_name' not in df.columns:
                return 0
            
            z_threshold = thresholds.get('regional_z_score', 2.5)
            
            for col in indicator_cols[:5]:  # Sample first 5 indicators
                for region, group_df in df.groupby('region_name'):
                    values = group_df[col].dropna()
                    if len(values) < 3:
                        continue
                    
                    median_val = values.median()
                    mad_val = np.median(np.abs(values - median_val))
                    
                    if mad_val > 0:
                        robust_z = np.abs((values - median_val) / (1.4826 * mad_val))
                        outliers = (robust_z > z_threshold).sum()
                        total_anomalies += outliers
            
            # Extrapolate to all indicators
            if len(indicator_cols) > 5:
                total_anomalies = int(total_anomalies * len(indicator_cols) / 5)
        
        elif detector_name == 'temporal':
            # Temporal anomalies are typically rare
            # Estimate ~1-2% of municipalities per indicator
            total_anomalies = int(len(df) * len(indicator_cols) * 0.015)
        
        elif detector_name == 'cross_source':
            # Cross-source discrepancies affect ~5-10% of municipalities
            total_anomalies = int(len(df) * 0.075)
        
        elif detector_name == 'logical':
            # Logical inconsistencies are typically rare
            # Estimate ~2-3% of municipalities
            total_anomalies = int(len(df) * 0.025)
        
        else:
            # Unknown detector - use conservative estimate
            total_anomalies = int(len(df) * 0.05)
        
        return total_anomalies
    
    def export_tuned_thresholds(
        self,
        optimized_thresholds: Dict[str, Dict[str, float]],
        output_file: Optional[str] = None
    ) -> str:
        """
        Export tuned thresholds to a YAML configuration file.
        
        This method generates a configuration file that can be used to apply
        the optimized thresholds. The file includes both the threshold values
        and metadata about the tuning process.
        
        Args:
            optimized_thresholds: Dictionary of optimized thresholds for each detector
            output_file: Optional path to output file (auto-generated if None)
            
        Returns:
            Path to the exported configuration file
        """
        import yaml
        
        # Generate output filename if not provided
        if output_file is None:
            output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f'tuned_thresholds_{timestamp}.yaml'
        else:
            output_file = Path(output_file)
        
        # Get latest tuning info if available
        tuning_metadata = {}
        if self.tuning_history:
            latest = self.tuning_history[-1]
            tuning_metadata = {
                'tuning_id': latest.tuning_id,
                'tuning_timestamp': latest.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'optimization_strategy': latest.results[0].optimization_strategy if latest.results else 'unknown',
                'total_anomalies_before': latest.total_anomalies_before,
                'total_anomalies_after': latest.total_anomalies_after,
                'anomaly_reduction_pct': round(
                    (1 - latest.total_anomalies_after / max(latest.total_anomalies_before, 1)) * 100, 1
                ),
                'avg_fpr_before': round(latest.avg_fpr_before, 4),
                'avg_fpr_after': round(latest.avg_fpr_after, 4),
                'fpr_reduction_pct': round(
                    (1 - latest.avg_fpr_after / max(latest.avg_fpr_before, 1)) * 100, 1
                )
            }
        
        # Build configuration structure
        config_data = {
            'auto_tuning_metadata': {
                'generated_by': 'AutoTuner',
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'target_fpr': self.target_fpr,
                **tuning_metadata
            },
            'thresholds': optimized_thresholds
        }
        
        # Write to YAML file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"Tuned thresholds exported to {output_file}")
            return str(output_file)
        
        except Exception as e:
            self.logger.error(f"Failed to export tuned thresholds: {e}")
            raise
    
    def generate_tuning_report(
        self,
        include_rationale: bool = True,
        include_statistics: bool = True
    ) -> str:
        """
        Generate a comprehensive human-readable tuning report.
        
        This method creates a detailed report including:
        - Summary of tuning results
        - Detector-specific threshold changes
        - Statistical analysis and rationale
        - Recommendations for future tuning
        
        Args:
            include_rationale: If True, includes detailed rationale for threshold changes
            include_statistics: If True, includes detailed statistics
            
        Returns:
            Markdown-formatted report string
        """
        if not self.tuning_history:
            return "# Auto-Tuning Report\n\nNo tuning history available.\n"
        
        latest = self.tuning_history[-1]
        strategy = latest.results[0].optimization_strategy if latest.results else 'N/A'
        
        report = [
            "# Auto-Tuning Report",
            "",
            f"**Tuning ID:** {latest.tuning_id}",
            f"**Timestamp:** {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Optimization Strategy:** {strategy}",
            f"**Target False Positive Rate:** {self.target_fpr:.3f} ({self.target_fpr * 100:.1f}%)",
            "",
            "## Executive Summary",
            "",
            f"The auto-tuning process optimized detection thresholds using the **{strategy}** strategy "
            f"to minimize false positives while maintaining detection capability.",
            ""
        ]
        
        # Add summary statistics
        if include_statistics:
            anomaly_reduction = latest.total_anomalies_before - latest.total_anomalies_after
            anomaly_reduction_pct = (1 - latest.total_anomalies_after / max(latest.total_anomalies_before, 1)) * 100
            fpr_reduction_pct = (1 - latest.avg_fpr_after / max(latest.avg_fpr_before, 1)) * 100
            
            report.extend([
                "### Key Metrics",
                "",
                f"- **Total Anomalies Before:** {latest.total_anomalies_before:,}",
                f"- **Total Anomalies After:** {latest.total_anomalies_after:,}",
                f"- **Anomaly Reduction:** {anomaly_reduction:,} ({anomaly_reduction_pct:.1f}%)",
                f"- **Average FPR Before:** {latest.avg_fpr_before:.4f} ({latest.avg_fpr_before * 100:.2f}%)",
                f"- **Average FPR After:** {latest.avg_fpr_after:.4f} ({latest.avg_fpr_after * 100:.2f}%)",
                f"- **FPR Reduction:** {fpr_reduction_pct:.1f}%",
                ""
            ])
            
            # Add interpretation
            if latest.avg_fpr_after <= self.target_fpr:
                report.append(f"✅ **Target FPR achieved:** The optimized thresholds meet the target FPR of {self.target_fpr:.3f}.")
            elif latest.avg_fpr_after <= self.target_fpr * 1.2:
                report.append(f"⚠️ **Near target:** The optimized FPR is within 20% of the target ({self.target_fpr:.3f}).")
            else:
                report.append(f"❌ **Above target:** The optimized FPR exceeds the target. Consider more aggressive threshold adjustments.")
            
            report.append("")
        
        # Add detector-specific results
        report.extend([
            "## Detector-Specific Results",
            "",
            "The following sections detail the threshold optimizations for each detector.",
            ""
        ])
        
        for result in latest.results:
            detector_title = result.detector_name.replace('_', ' ').title()
            report.extend([
                f"### {detector_title} Detector",
                ""
            ])
            
            # Threshold comparison table
            report.extend([
                "#### Threshold Changes",
                "",
                "| Parameter | Original | Optimized | Change |",
                "|-----------|----------|-----------|--------|"
            ])
            
            for key in result.original_thresholds.keys():
                orig_val = result.original_thresholds[key]
                opt_val = result.optimized_thresholds.get(key, orig_val)
                
                if orig_val != 0:
                    change_pct = ((opt_val - orig_val) / abs(orig_val)) * 100
                    change_str = f"{change_pct:+.1f}%"
                else:
                    change_str = "N/A"
                
                report.append(f"| `{key}` | {orig_val:.3f} | {opt_val:.3f} | {change_str} |")
            
            report.append("")
            
            # Performance metrics
            if include_statistics:
                fpr_improvement = (1 - result.estimated_fpr_after / max(result.estimated_fpr_before, 1)) * 100
                
                report.extend([
                    "#### Performance Metrics",
                    "",
                    f"- **Estimated FPR Before:** {result.estimated_fpr_before:.4f} ({result.estimated_fpr_before * 100:.2f}%)",
                    f"- **Estimated FPR After:** {result.estimated_fpr_after:.4f} ({result.estimated_fpr_after * 100:.2f}%)",
                    f"- **FPR Improvement:** {fpr_improvement:.1f}%",
                    f"- **Confidence Score:** {result.confidence_score:.2f}/1.00",
                    ""
                ])
                
                if result.anomaly_count_before > 0 or result.anomaly_count_after > 0:
                    anomaly_change = result.anomaly_count_before - result.anomaly_count_after
                    anomaly_change_pct = (1 - result.anomaly_count_after / max(result.anomaly_count_before, 1)) * 100
                    
                    report.extend([
                        f"- **Estimated Anomalies Before:** {result.anomaly_count_before:,}",
                        f"- **Estimated Anomalies After:** {result.anomaly_count_after:,}",
                        f"- **Anomaly Reduction:** {anomaly_change:,} ({anomaly_change_pct:.1f}%)",
                        ""
                    ])
            
            # Add rationale
            if include_rationale:
                rationale = self._generate_threshold_rationale(result)
                report.extend([
                    "#### Rationale",
                    "",
                    rationale,
                    ""
                ])
        
        # Add recommendations
        report.extend([
            "## Recommendations",
            ""
        ])
        
        recommendations = self._generate_recommendations(latest)
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report.extend([
            "",
            "## Next Steps",
            "",
            "1. **Review the optimized thresholds** in the exported configuration file",
            "2. **Apply the thresholds** by updating your `config.yaml` with the optimized values",
            "3. **Run detection** with the new thresholds and validate results",
            f"4. **Schedule re-tuning** in {self.retuning_interval_days} days or when data patterns change",
            "",
            "---",
            "",
            f"*Report generated by AutoTuner on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(report)
    
    def _generate_threshold_rationale(self, result: ThresholdOptimizationResult) -> str:
        """
        Generate rationale for threshold changes.
        
        Args:
            result: Optimization result for a detector
            
        Returns:
            Human-readable rationale string
        """
        strategy = result.optimization_strategy
        detector = result.detector_name
        
        # Calculate average threshold change
        changes = []
        for key in result.original_thresholds.keys():
            orig = result.original_thresholds[key]
            opt = result.optimized_thresholds.get(key, orig)
            if orig != 0:
                change_pct = ((opt - orig) / abs(orig)) * 100
                changes.append(change_pct)
        
        avg_change = np.mean(changes) if changes else 0
        
        # Generate rationale based on strategy and changes
        rationale_parts = []
        
        # Strategy explanation
        if strategy == 'conservative':
            rationale_parts.append(
                "The **conservative** strategy increases thresholds to significantly reduce false positives, "
                "prioritizing precision over recall."
            )
        elif strategy == 'balanced':
            rationale_parts.append(
                "The **balanced** strategy adjusts thresholds based on data characteristics, "
                "balancing false positive reduction with detection capability."
            )
        elif strategy == 'adaptive':
            rationale_parts.append(
                "The **adaptive** strategy uses data-driven threshold optimization, "
                "analyzing actual distributions to find optimal detection points."
            )
        
        # Change magnitude explanation
        if avg_change > 20:
            rationale_parts.append(
                f"Thresholds were increased by an average of {avg_change:.1f}%, indicating that the original "
                "thresholds were too sensitive and likely producing many false positives."
            )
        elif avg_change > 5:
            rationale_parts.append(
                f"Thresholds were moderately increased by {avg_change:.1f}% to reduce false positives "
                "while maintaining good detection capability."
            )
        elif avg_change > -5:
            rationale_parts.append(
                "Thresholds required minimal adjustment, suggesting they were already well-calibrated."
            )
        else:
            rationale_parts.append(
                f"Thresholds were decreased by {abs(avg_change):.1f}%, indicating the original thresholds "
                "may have been too strict and missing true anomalies."
            )
        
        # Detector-specific rationale
        if detector == 'statistical':
            rationale_parts.append(
                "For statistical detection, higher z-score thresholds reduce sensitivity to natural variation "
                "in the data, focusing on more extreme outliers."
            )
        elif detector == 'geographic':
            rationale_parts.append(
                "For geographic detection, adjusted thresholds account for natural regional variation, "
                "reducing false positives from legitimate urban-rural differences."
            )
        elif detector == 'temporal':
            rationale_parts.append(
                "For temporal detection, threshold adjustments balance sensitivity to genuine changes "
                "against normal seasonal or cyclical variations."
            )
        elif detector == 'cross_source':
            rationale_parts.append(
                "For cross-source comparison, threshold adjustments account for expected discrepancies "
                "between different data sources while flagging significant inconsistencies."
            )
        
        # Confidence note
        if result.confidence_score >= 0.8:
            rationale_parts.append(
                f"The optimization has **high confidence** (score: {result.confidence_score:.2f}) "
                "based on sufficient data for analysis."
            )
        elif result.confidence_score >= 0.6:
            rationale_parts.append(
                f"The optimization has **moderate confidence** (score: {result.confidence_score:.2f}). "
                "Results should be validated with actual detection runs."
            )
        else:
            rationale_parts.append(
                f"The optimization has **low confidence** (score: {result.confidence_score:.2f}) "
                "due to limited data. Manual review and adjustment may be needed."
            )
        
        return " ".join(rationale_parts)
    
    def _generate_recommendations(self, tuning_history: TuningHistory) -> List[str]:
        """
        Generate recommendations based on tuning results.
        
        Args:
            tuning_history: Latest tuning history entry
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check if target FPR was achieved
        if tuning_history.avg_fpr_after <= self.target_fpr:
            recommendations.append(
                "✅ Target FPR achieved. The optimized thresholds are ready for production use."
            )
        elif tuning_history.avg_fpr_after <= self.target_fpr * 1.2:
            recommendations.append(
                "⚠️ FPR is slightly above target. Consider running another tuning iteration with "
                "a more conservative strategy."
            )
        else:
            recommendations.append(
                "❌ FPR significantly exceeds target. Consider using the 'conservative' strategy "
                "or manually increasing thresholds further."
            )
        
        # Check anomaly reduction
        if tuning_history.total_anomalies_after > 0:
            reduction_pct = (1 - tuning_history.total_anomalies_after / max(tuning_history.total_anomalies_before, 1)) * 100
            
            if reduction_pct > 50:
                recommendations.append(
                    f"Significant anomaly reduction ({reduction_pct:.1f}%) achieved. "
                    "Verify that true anomalies are still being detected."
                )
            elif reduction_pct < 10:
                recommendations.append(
                    f"Limited anomaly reduction ({reduction_pct:.1f}%). "
                    "Consider more aggressive threshold adjustments if false positives remain high."
                )
        
        # Check confidence scores
        low_confidence_detectors = [
            r.detector_name for r in tuning_history.results if r.confidence_score < 0.6
        ]
        
        if low_confidence_detectors:
            recommendations.append(
                f"Low confidence in optimization for: {', '.join(low_confidence_detectors)}. "
                "Validate these detectors with actual detection runs and adjust manually if needed."
            )
        
        # Re-tuning schedule
        next_tuning = self.get_next_tuning_date()
        if next_tuning:
            recommendations.append(
                f"Schedule next re-tuning for {next_tuning.strftime('%Y-%m-%d')} "
                f"({self.retuning_interval_days} days from last tuning)."
            )
        
        # General recommendations
        recommendations.append(
            "Monitor detection results over the next few runs to ensure thresholds are working as expected."
        )
        
        recommendations.append(
            "Consider creating a custom threshold profile based on these optimized values for future use."
        )
        
        return recommendations
    
    def export_tuning_package(
        self,
        optimized_thresholds: Dict[str, Dict[str, float]],
        output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export a complete tuning package including configuration and report.
        
        This method creates a comprehensive export package containing:
        - Tuned threshold configuration file (YAML)
        - Human-readable tuning report (Markdown)
        - Tuning statistics (JSON)
        
        Args:
            optimized_thresholds: Dictionary of optimized thresholds
            output_dir: Optional output directory (uses config default if None)
            
        Returns:
            Dictionary mapping file types to their paths
        """
        # Determine output directory
        if output_dir is None:
            output_dir = Path(self.config.get('export', {}).get('output_dir', 'output'))
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = {}
        
        try:
            # 1. Export threshold configuration
            config_file = output_dir / f'tuned_thresholds_{timestamp}.yaml'
            self.export_tuned_thresholds(optimized_thresholds, str(config_file))
            exported_files['config'] = str(config_file)
            self.logger.info(f"Exported threshold configuration to {config_file}")
            
            # 2. Export tuning report
            report_file = output_dir / f'tuning_report_{timestamp}.md'
            report_content = self.generate_tuning_report(
                include_rationale=True,
                include_statistics=True
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            exported_files['report'] = str(report_file)
            self.logger.info(f"Exported tuning report to {report_file}")
            
            # 3. Export tuning statistics (JSON)
            if self.tuning_history:
                stats_file = output_dir / f'tuning_statistics_{timestamp}.json'
                latest = self.tuning_history[-1]
                
                stats_data = {
                    'tuning_id': latest.tuning_id,
                    'timestamp': latest.timestamp.isoformat(),
                    'summary': {
                        'total_anomalies_before': latest.total_anomalies_before,
                        'total_anomalies_after': latest.total_anomalies_after,
                        'anomaly_reduction': latest.total_anomalies_before - latest.total_anomalies_after,
                        'anomaly_reduction_pct': round(
                            (1 - latest.total_anomalies_after / max(latest.total_anomalies_before, 1)) * 100, 2
                        ),
                        'avg_fpr_before': round(latest.avg_fpr_before, 4),
                        'avg_fpr_after': round(latest.avg_fpr_after, 4),
                        'fpr_reduction_pct': round(
                            (1 - latest.avg_fpr_after / max(latest.avg_fpr_before, 1)) * 100, 2
                        ),
                        'target_fpr': self.target_fpr,
                        'target_achieved': latest.avg_fpr_after <= self.target_fpr
                    },
                    'detector_results': [
                        {
                            'detector_name': r.detector_name,
                            'optimization_strategy': r.optimization_strategy,
                            'original_thresholds': r.original_thresholds,
                            'optimized_thresholds': r.optimized_thresholds,
                            'estimated_fpr_before': round(r.estimated_fpr_before, 4),
                            'estimated_fpr_after': round(r.estimated_fpr_after, 4),
                            'anomaly_count_before': r.anomaly_count_before,
                            'anomaly_count_after': r.anomaly_count_after,
                            'confidence_score': round(r.confidence_score, 2)
                        }
                        for r in latest.results
                    ]
                }
                
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_data, f, indent=2, ensure_ascii=False)
                
                exported_files['statistics'] = str(stats_file)
                self.logger.info(f"Exported tuning statistics to {stats_file}")
            
            self.logger.info(
                f"Tuning package exported successfully: {len(exported_files)} files created"
            )
            
            return exported_files
        
        except Exception as e:
            self.logger.error(f"Failed to export tuning package: {e}", exc_info=True)
            raise
