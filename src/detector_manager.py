"""
Detector Manager Module for СберИндекс Anomaly Detection System

This module provides centralized management of anomaly detectors with:
- Error handling and graceful degradation
- Detector statistics tracking
- Threshold management
- Configuration profile support
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.error_handler import get_error_handler


logger = logging.getLogger(__name__)


@dataclass
class DetectorStats:
    """Statistics for a single detector execution."""
    detector_name: str
    success: bool
    execution_time_seconds: float
    anomalies_detected: int
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ThresholdManager:
    """
    Manages detection thresholds and configuration profiles.
    
    Supports multiple profiles (strict, normal, relaxed) and allows
    runtime threshold adjustments.
    """
    
    # Required threshold parameters for each detector type
    REQUIRED_THRESHOLDS = {
        'statistical': ['z_score', 'iqr_multiplier', 'percentile_lower', 'percentile_upper'],
        'temporal': ['spike_threshold', 'drop_threshold', 'volatility_multiplier'],
        'geographic': ['regional_z_score', 'cluster_threshold'],
        'cross_source': ['correlation_threshold', 'discrepancy_threshold'],
        'logical': ['check_negative_values', 'check_impossible_ratios']
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize threshold manager with configuration.
        
        Args:
            config: Configuration dictionary containing thresholds
        """
        self.config = config
        self.profile = config.get('detection_profile', 'normal')
        self.logger = logging.getLogger(f"{__name__}.ThresholdManager")
        
        # Load default thresholds
        self.default_thresholds = config.get('thresholds', {})
        
        # Load profile-specific thresholds if available
        self.profile_thresholds = self._load_profile_thresholds()
        
        self.logger.info(f"ThresholdManager initialized with profile: {self.profile}")
    
    def _load_profile_thresholds(self) -> Dict[str, Any]:
        """
        Load threshold profiles from configuration with validation and merging.
        
        Returns:
            Dictionary containing profile-specific thresholds merged with defaults
        """
        # Check if threshold_profiles section exists in config
        profiles = self.config.get('threshold_profiles', {})
        
        if not profiles:
            self.logger.debug("No threshold_profiles found in config, using default thresholds")
            return self.default_thresholds.copy()
        
        # Get the selected profile
        if self.profile not in profiles:
            self.logger.warning(
                f"Profile '{self.profile}' not found in threshold_profiles, "
                f"using default thresholds"
            )
            return self.default_thresholds.copy()
        
        # Get profile thresholds
        profile_thresholds = profiles[self.profile]
        
        # Merge with defaults to fill in missing parameters
        merged_thresholds = self._merge_with_defaults(profile_thresholds)
        
        # Validate completeness
        validation_result = self._validate_profile_completeness(merged_thresholds)
        
        if not validation_result['is_valid']:
            self.logger.warning(
                f"Profile '{self.profile}' is incomplete. Missing parameters: "
                f"{validation_result['missing_params']}. Using defaults for missing values."
            )
        else:
            self.logger.info(f"Profile '{self.profile}' loaded and validated successfully")
        
        return merged_thresholds
    
    def _merge_with_defaults(self, profile_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge profile thresholds with default thresholds.
        
        Missing parameters in the profile are filled from defaults.
        
        Args:
            profile_thresholds: Profile-specific threshold values
            
        Returns:
            Merged threshold dictionary
        """
        merged = {}
        
        # Iterate through all detector types
        for detector_type in self.REQUIRED_THRESHOLDS.keys():
            merged[detector_type] = {}
            
            # Get default values for this detector type
            default_values = self.default_thresholds.get(detector_type, {})
            
            # Get profile values for this detector type
            profile_values = profile_thresholds.get(detector_type, {})
            
            # Merge: profile values take precedence, defaults fill gaps
            for param in self.REQUIRED_THRESHOLDS[detector_type]:
                if param in profile_values:
                    merged[detector_type][param] = profile_values[param]
                elif param in default_values:
                    merged[detector_type][param] = default_values[param]
                    self.logger.debug(
                        f"Using default value for {detector_type}.{param}: "
                        f"{default_values[param]}"
                    )
                else:
                    self.logger.warning(
                        f"No value found for {detector_type}.{param} in profile or defaults"
                    )
        
        return merged
    
    def _validate_profile_completeness(self, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all required threshold parameters are present.
        
        Args:
            thresholds: Threshold dictionary to validate
            
        Returns:
            Dictionary with validation results:
            - is_valid: bool indicating if all required params are present
            - missing_params: list of missing parameter paths
            - complete_params: list of present parameter paths
        """
        missing_params = []
        complete_params = []
        
        for detector_type, required_params in self.REQUIRED_THRESHOLDS.items():
            detector_thresholds = thresholds.get(detector_type, {})
            
            for param in required_params:
                param_path = f"{detector_type}.{param}"
                
                if param not in detector_thresholds or detector_thresholds[param] is None:
                    missing_params.append(param_path)
                else:
                    complete_params.append(param_path)
        
        return {
            'is_valid': len(missing_params) == 0,
            'missing_params': missing_params,
            'complete_params': complete_params,
            'completeness_percentage': (
                len(complete_params) / (len(complete_params) + len(missing_params)) * 100
                if (len(complete_params) + len(missing_params)) > 0 else 0
            )
        }
    
    def get_thresholds(self, detector_name: str) -> Dict[str, float]:
        """
        Get thresholds for a specific detector based on current profile.
        
        Args:
            detector_name: Name of the detector (e.g., 'statistical', 'geographic')
            
        Returns:
            Dictionary of threshold values for the detector
        """
        # Profile thresholds are already merged with defaults
        if self.profile_thresholds and detector_name in self.profile_thresholds:
            thresholds = self.profile_thresholds[detector_name]
            self.logger.debug(
                f"Using profile '{self.profile}' thresholds for {detector_name}: {thresholds}"
            )
            return thresholds
        
        # Fall back to default thresholds from config if detector not in profile
        default_thresholds = self.default_thresholds.get(detector_name, {})
        self.logger.debug(
            f"Using default thresholds for {detector_name}: {default_thresholds}"
        )
        return default_thresholds
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Load a specific threshold profile with validation and merging.
        
        Args:
            profile_name: Name of the profile to load ('strict', 'normal', 'relaxed')
            
        Returns:
            Dictionary containing all thresholds for the profile (merged with defaults)
            
        Raises:
            ValueError: If profile_name is not found in configuration
        """
        profiles = self.config.get('threshold_profiles', {})
        
        if profile_name not in profiles:
            self.logger.error(f"Profile '{profile_name}' not found in configuration")
            available_profiles = ', '.join(profiles.keys()) if profiles else 'none'
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available profiles: {available_profiles}"
            )
        
        # Update current profile
        self.profile = profile_name
        
        # Get profile thresholds
        profile_thresholds = profiles[profile_name]
        
        # Merge with defaults
        self.profile_thresholds = self._merge_with_defaults(profile_thresholds)
        
        # Validate completeness
        validation_result = self._validate_profile_completeness(self.profile_thresholds)
        
        if not validation_result['is_valid']:
            self.logger.warning(
                f"Profile '{profile_name}' is incomplete "
                f"({validation_result['completeness_percentage']:.1f}% complete). "
                f"Missing parameters: {validation_result['missing_params']}. "
                f"Using defaults for missing values."
            )
        else:
            self.logger.info(
                f"Profile '{profile_name}' loaded and validated successfully "
                f"(100% complete)"
            )
        
        return self.profile_thresholds
    
    def load_custom_profile(self, custom_thresholds: Dict[str, Any], profile_name: str = 'custom') -> Dict[str, Any]:
        """
        Load a custom threshold profile with validation and merging.
        
        Custom profiles are merged with default thresholds to fill in missing parameters.
        
        Args:
            custom_thresholds: Dictionary containing custom threshold values
            profile_name: Name for the custom profile (default: 'custom')
            
        Returns:
            Dictionary containing merged and validated thresholds
        """
        self.logger.info(f"Loading custom profile: {profile_name}")
        
        # Update current profile name
        self.profile = profile_name
        
        # Merge with defaults
        self.profile_thresholds = self._merge_with_defaults(custom_thresholds)
        
        # Validate completeness
        validation_result = self._validate_profile_completeness(self.profile_thresholds)
        
        if not validation_result['is_valid']:
            self.logger.warning(
                f"Custom profile '{profile_name}' is incomplete "
                f"({validation_result['completeness_percentage']:.1f}% complete). "
                f"Missing parameters: {validation_result['missing_params']}. "
                f"Using defaults for missing values."
            )
        else:
            self.logger.info(
                f"Custom profile '{profile_name}' loaded and validated successfully "
                f"(100% complete)"
            )
        
        return self.profile_thresholds
    
    def apply_auto_tuned_thresholds(self, tuned_thresholds: Dict[str, float]):
        """
        Apply thresholds from auto-tuner.
        
        Args:
            tuned_thresholds: Dictionary of auto-tuned threshold values
        """
        # Merge auto-tuned thresholds with current profile
        if not self.profile_thresholds:
            self.profile_thresholds = {}
        
        for detector_name, thresholds in tuned_thresholds.items():
            if detector_name not in self.profile_thresholds:
                self.profile_thresholds[detector_name] = {}
            
            self.profile_thresholds[detector_name].update(thresholds)
        
        self.logger.info(f"Applied auto-tuned thresholds for {len(tuned_thresholds)} detectors")
    
    def get_profile_info(self) -> Dict[str, Any]:
        """
        Get information about the current profile.
        
        Returns:
            Dictionary containing profile information:
            - profile_name: Name of current profile
            - validation: Validation results
            - thresholds: Current threshold values
        """
        validation_result = self._validate_profile_completeness(self.profile_thresholds)
        
        return {
            'profile_name': self.profile,
            'validation': validation_result,
            'thresholds': self.profile_thresholds
        }


class DetectorManager:
    """
    Manages execution of all anomaly detectors with error handling.
    
    Provides centralized detector orchestration, error handling, and
    statistics tracking. Ensures that failure of one detector doesn't
    prevent execution of others.
    """
    
    def __init__(self, config: Dict[str, Any], source_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize detector manager with configuration.
        
        Args:
            config: Configuration dictionary containing thresholds and settings
            source_mapping: Optional dictionary mapping column names to data sources
        """
        self.config = config
        self.source_mapping = source_mapping or {}
        self.threshold_manager = ThresholdManager(config)
        self.logger = logging.getLogger(f"{__name__}.DetectorManager")
        
        # Initialize error handler
        self.error_handler = get_error_handler()
        
        # Statistics tracking
        self.detector_stats: List[DetectorStats] = []
        
        # Apply profile thresholds to config before initializing detectors
        self._apply_profile_to_config()
        
        # Initialize detectors
        self.detectors = self._initialize_detectors()
        
        profile_name = self.threshold_manager.profile
        self.logger.info(
            f"DetectorManager initialized with {len(self.detectors)} detectors "
            f"using profile '{profile_name}'"
        )
    
    def _apply_profile_to_config(self):
        """
        Apply profile thresholds to config.
        
        Updates the config['thresholds'] section with values from the loaded profile.
        This ensures detectors use profile-specific thresholds when initialized.
        """
        # Get profile thresholds from threshold manager
        profile_thresholds = self.threshold_manager.profile_thresholds
        
        if profile_thresholds:
            # Update config with profile thresholds
            self.config['thresholds'] = profile_thresholds
            self.logger.debug(
                f"Applied profile '{self.threshold_manager.profile}' thresholds to config"
            )
        else:
            self.logger.warning(
                "No profile thresholds available, using default config thresholds"
            )
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """
        Initialize all detector instances.
        
        Returns:
            Dictionary mapping detector names to detector instances
        """
        from src.anomaly_detector import (
            StatisticalOutlierDetector,
            CrossSourceComparator,
            TemporalAnomalyDetector,
            GeographicAnomalyDetector,
            LogicalConsistencyChecker
        )
        
        detectors = {}
        
        try:
            detector = StatisticalOutlierDetector(self.config)
            detector.source_mapping = self.source_mapping
            detectors['statistical'] = detector
            self.logger.debug("Initialized StatisticalOutlierDetector")
        except Exception as e:
            self.logger.error(f"Failed to initialize StatisticalOutlierDetector: {e}")
        
        try:
            detector = CrossSourceComparator(self.config)
            detector.source_mapping = self.source_mapping
            detectors['cross_source'] = detector
            self.logger.debug("Initialized CrossSourceComparator")
        except Exception as e:
            self.logger.error(f"Failed to initialize CrossSourceComparator: {e}")
        
        try:
            detector = TemporalAnomalyDetector(self.config)
            detector.source_mapping = self.source_mapping
            detectors['temporal'] = detector
            self.logger.debug("Initialized TemporalAnomalyDetector")
        except Exception as e:
            self.logger.error(f"Failed to initialize TemporalAnomalyDetector: {e}")
        
        try:
            detector = GeographicAnomalyDetector(self.config)
            detector.source_mapping = self.source_mapping
            detectors['geographic'] = detector
            self.logger.debug("Initialized GeographicAnomalyDetector")
        except Exception as e:
            self.logger.error(f"Failed to initialize GeographicAnomalyDetector: {e}")
        
        try:
            detector = LogicalConsistencyChecker(self.config)
            detector.source_mapping = self.source_mapping
            detectors['logical'] = detector
            self.logger.debug("Initialized LogicalConsistencyChecker")
        except Exception as e:
            self.logger.error(f"Failed to initialize LogicalConsistencyChecker: {e}")
        
        return detectors
    
    def run_all_detectors(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Run all detectors with error handling.
        
        Each detector runs independently. If one fails, others continue.
        Statistics are tracked for each detector execution.
        
        Args:
            df: DataFrame containing municipal data
            
        Returns:
            List of DataFrames containing anomalies from each detector
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting detector execution")
        self.logger.info(f"Input data shape: {df.shape}")
        self.logger.info("=" * 80)
        
        results = []
        self.detector_stats = []  # Reset statistics
        
        for detector_name, detector in self.detectors.items():
            anomalies_df = self.run_detector_safe(detector_name, detector, df)
            
            if anomalies_df is not None and len(anomalies_df) > 0:
                results.append(anomalies_df)
        
        # Log summary statistics
        self._log_execution_summary()
        
        return results
    
    def run_detector_safe(
        self, 
        detector_name: str, 
        detector: Any, 
        df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Run a single detector with try-catch error handling.
        
        Args:
            detector_name: Name of the detector for logging
            detector: Detector instance to run
            df: DataFrame containing municipal data
            
        Returns:
            DataFrame with detected anomalies, or None if detector failed
        """
        start_time = datetime.now()
        
        self.logger.info(f"Running {detector_name} detector...")
        
        try:
            # Run the detector
            anomalies = detector.detect(df)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Track statistics
            stats = DetectorStats(
                detector_name=detector_name,
                success=True,
                execution_time_seconds=execution_time,
                anomalies_detected=len(anomalies) if anomalies is not None else 0,
                error_message=None,
                started_at=start_time,
                completed_at=end_time
            )
            self.detector_stats.append(stats)
            
            # Log success
            anomaly_count = len(anomalies) if anomalies is not None else 0
            self.logger.info(
                f"✓ {detector_name} completed successfully: "
                f"{anomaly_count} anomalies detected in {execution_time:.2f}s"
            )
            
            return anomalies
            
        except Exception as e:
            # Calculate execution time even for failures
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Use enhanced error handler to capture full context
            error_context = self.error_handler.handle_error(
                exception=e,
                component_name=f"DetectorManager.{detector_name}",
                data=df,
                config=self.config,
                additional_context={
                    'detector_name': detector_name,
                    'detector_class': detector.__class__.__name__,
                    'execution_time_seconds': execution_time,
                    'started_at': start_time.isoformat(),
                    'failed_at': end_time.isoformat()
                }
            )
            
            # Track failure statistics
            stats = DetectorStats(
                detector_name=detector_name,
                success=False,
                execution_time_seconds=execution_time,
                anomalies_detected=0,
                error_message=error_context.get('error_message', str(e)),
                started_at=start_time,
                completed_at=end_time
            )
            self.detector_stats.append(stats)
            
            # Return None to indicate failure
            return None
    
    def get_detector_statistics(self) -> Dict[str, DetectorStats]:
        """
        Get execution statistics for each detector.
        
        Returns:
            Dictionary mapping detector names to their statistics
        """
        return {stat.detector_name: stat for stat in self.detector_stats}
    
    def switch_profile(self, profile_name: str):
        """
        Switch to a different threshold profile at runtime.
        
        This method loads a new profile and reinitializes all detectors with
        the new thresholds. Any previous detection results are preserved.
        
        Args:
            profile_name: Name of the profile to switch to ('strict', 'normal', 'relaxed')
            
        Raises:
            ValueError: If profile_name is not found in configuration
            
        Example:
            >>> manager = DetectorManager(config)
            >>> manager.switch_profile('strict')  # Switch to strict mode
            >>> results = manager.run_all_detectors(df)
        """
        self.logger.info(f"Switching from profile '{self.threshold_manager.profile}' to '{profile_name}'")
        
        # Load new profile in threshold manager
        self.threshold_manager.load_profile(profile_name)
        
        # Apply new profile thresholds to config
        self._apply_profile_to_config()
        
        # Reinitialize detectors with new thresholds
        old_detector_count = len(self.detectors)
        self.detectors = self._initialize_detectors()
        new_detector_count = len(self.detectors)
        
        self.logger.info(
            f"Profile switched to '{profile_name}'. "
            f"Reinitialized {new_detector_count}/{old_detector_count} detectors."
        )
    
    def get_current_profile(self) -> str:
        """
        Get the name of the currently active profile.
        
        Returns:
            Name of the current profile (e.g., 'normal', 'strict', 'relaxed')
        """
        return self.threshold_manager.profile
    
    def get_profile_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current profile.
        
        Returns:
            Dictionary containing:
            - profile_name: Name of current profile
            - validation: Validation results (completeness, missing params)
            - thresholds: Current threshold values for all detectors
        """
        return self.threshold_manager.get_profile_info()
    
    def _log_execution_summary(self):
        """Log summary of detector execution statistics."""
        self.logger.info("=" * 80)
        self.logger.info("DETECTOR EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        
        total_detectors = len(self.detector_stats)
        successful_detectors = sum(1 for stat in self.detector_stats if stat.success)
        failed_detectors = total_detectors - successful_detectors
        
        total_anomalies = sum(stat.anomalies_detected for stat in self.detector_stats)
        total_time = sum(stat.execution_time_seconds for stat in self.detector_stats)
        
        self.logger.info(f"Total detectors: {total_detectors}")
        self.logger.info(f"Successful: {successful_detectors}")
        self.logger.info(f"Failed: {failed_detectors}")
        self.logger.info(f"Total anomalies detected: {total_anomalies}")
        self.logger.info(f"Total execution time: {total_time:.2f}s")
        self.logger.info("")
        
        # Log individual detector statistics
        self.logger.info("Individual Detector Statistics:")
        for stat in self.detector_stats:
            status = "✓" if stat.success else "✗"
            self.logger.info(
                f"  {status} {stat.detector_name}: "
                f"{stat.anomalies_detected} anomalies, "
                f"{stat.execution_time_seconds:.2f}s"
            )
            if not stat.success:
                self.logger.info(f"    Error: {stat.error_message}")
        
        self.logger.info("=" * 80)
