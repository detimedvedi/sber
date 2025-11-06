"""
Configuration Validator Module

Provides comprehensive validation for system configuration including:
- Schema validation
- Required field checks
- Value range validation
- Type checking
- Profile validation
- Configuration migration from old to new format
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a configuration validation error."""
    field: str
    message: str
    severity: str  # 'error' or 'warning'


class ConfigValidator:
    """
    Validates configuration against schema and business rules.
    
    Ensures configuration has all required fields, correct types,
    and values within acceptable ranges.
    """
    
    # Required top-level fields
    REQUIRED_FIELDS = [
        'thresholds',
        'export',
        'data_paths'
    ]
    
    # Required threshold categories
    REQUIRED_THRESHOLD_CATEGORIES = [
        'statistical',
        'temporal',
        'geographic',
        'cross_source',
        'logical'
    ]
    
    # Field type specifications
    FIELD_TYPES = {
        'detection_profile': str,
        'export.output_dir': str,
        'export.timestamp_format': str,
        'export.top_n_municipalities': int,
        'logging.level': str,
        'data_processing.random_seed': int,
        'data_processing.min_data_completeness': float,
        'missing_value_handling.indicator_threshold': float,
        'missing_value_handling.municipality_threshold': float,
        'temporal.enabled': bool,
        'temporal.aggregation_method': str,
        'temporal.auto_detect': bool,
        'municipality_classification.enabled': bool,
        'municipality_classification.urban_population_threshold': int,
        'robust_statistics.enabled': bool,
        'robust_statistics.use_median': bool,
        'robust_statistics.use_mad': bool,
        'robust_statistics.log_transform_skewness_threshold': float,
        'auto_tuning.enabled': bool,
        'auto_tuning.target_false_positive_rate': float,
        'auto_tuning.min_anomalies_per_detector': int,
        'auto_tuning.max_anomalies_per_detector': int,
        'auto_tuning.retuning_interval_days': int,
        'auto_tuning.min_data_points': int,
        'auto_tuning.validation_confidence': float,
        'auto_tuning.export_tuned_config': bool,
        'auto_tuning.export_path': str,
    }
    
    # Value range specifications (min, max)
    VALUE_RANGES = {
        'thresholds.statistical.z_score': (0.0, 10.0),
        'thresholds.statistical.iqr_multiplier': (0.0, 10.0),
        'thresholds.statistical.percentile_lower': (0.0, 50.0),
        'thresholds.statistical.percentile_upper': (50.0, 100.0),
        'thresholds.temporal.spike_threshold': (0.0, 1000.0),
        'thresholds.temporal.drop_threshold': (-100.0, 100.0),  # Allow positive but warn
        'thresholds.temporal.volatility_multiplier': (0.0, 10.0),
        'thresholds.geographic.regional_z_score': (0.0, 10.0),
        'thresholds.geographic.cluster_threshold': (0.0, 10.0),
        'thresholds.cross_source.correlation_threshold': (0.0, 1.0),
        'thresholds.cross_source.discrepancy_threshold': (0.0, 1000.0),
        'export.top_n_municipalities': (1, 10000),
        'data_processing.min_data_completeness': (0.0, 1.0),
        'missing_value_handling.indicator_threshold': (0.0, 100.0),
        'missing_value_handling.municipality_threshold': (0.0, 100.0),
        'municipality_classification.urban_population_threshold': (0, 10000000),
        'robust_statistics.log_transform_skewness_threshold': (0.0, 100.0),
        'auto_tuning.target_false_positive_rate': (0.0, 1.0),
        'auto_tuning.min_anomalies_per_detector': (0, 100000),
        'auto_tuning.max_anomalies_per_detector': (1, 1000000),
        'auto_tuning.retuning_interval_days': (1, 365),
        'auto_tuning.min_data_points': (10, 1000000),
        'auto_tuning.validation_confidence': (0.0, 1.0),
    }
    
    # Valid enum values
    VALID_VALUES = {
        'detection_profile': ['strict', 'normal', 'relaxed'],
        'temporal.aggregation_method': ['latest', 'mean', 'median'],
        'logging.level': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        'data_processing.handle_missing': ['log_and_continue', 'raise_error', 'skip'],
    }
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """
        Validate configuration against all rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, list of errors/warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_required_fields(config)
        self._validate_threshold_structure(config)
        self._validate_field_types(config)
        self._validate_value_ranges(config)
        self._validate_enum_values(config)
        self._validate_profiles(config)
        self._validate_data_paths(config)
        self._validate_logical_consistency(config)
        
        # Combine errors and warnings
        all_issues = self.errors + self.warnings
        
        # Log results
        if self.errors:
            logger.error(f"Configuration validation failed with {len(self.errors)} error(s)")
            for error in self.errors:
                logger.error(f"  {error.field}: {error.message}")
        
        if self.warnings:
            logger.warning(f"Configuration has {len(self.warnings)} warning(s)")
            for warning in self.warnings:
                logger.warning(f"  {warning.field}: {warning.message}")
        
        if not self.errors and not self.warnings:
            logger.info("Configuration validation passed")
        
        return len(self.errors) == 0, all_issues
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> None:
        """Validate that all required top-level fields are present."""
        for field in self.REQUIRED_FIELDS:
            if field not in config:
                self.errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity='error'
                ))
    
    def _validate_threshold_structure(self, config: Dict[str, Any]) -> None:
        """Validate threshold structure and required categories."""
        if 'thresholds' not in config:
            return  # Already reported in required fields check
        
        thresholds = config['thresholds']
        if not isinstance(thresholds, dict):
            self.errors.append(ValidationError(
                field='thresholds',
                message="'thresholds' must be a dictionary",
                severity='error'
            ))
            return
        
        # Check required threshold categories
        for category in self.REQUIRED_THRESHOLD_CATEGORIES:
            if category not in thresholds:
                self.errors.append(ValidationError(
                    field=f'thresholds.{category}',
                    message=f"Required threshold category '{category}' is missing",
                    severity='error'
                ))
    
    def _validate_field_types(self, config: Dict[str, Any]) -> None:
        """Validate field types match specifications."""
        for field_path, expected_type in self.FIELD_TYPES.items():
            value = self._get_nested_value(config, field_path)
            
            if value is None:
                continue  # Field is optional or will be caught by required check
            
            if not isinstance(value, expected_type):
                self.errors.append(ValidationError(
                    field=field_path,
                    message=f"Expected type {expected_type.__name__}, got {type(value).__name__}",
                    severity='error'
                ))
    
    def _validate_value_ranges(self, config: Dict[str, Any]) -> None:
        """Validate numeric values are within acceptable ranges."""
        for field_path, (min_val, max_val) in self.VALUE_RANGES.items():
            value = self._get_nested_value(config, field_path)
            
            if value is None:
                continue  # Field is optional
            
            if not isinstance(value, (int, float)):
                continue  # Type error will be caught elsewhere
            
            if value < min_val or value > max_val:
                self.errors.append(ValidationError(
                    field=field_path,
                    message=f"Value {value} is outside valid range [{min_val}, {max_val}]",
                    severity='error'
                ))
    
    def _validate_enum_values(self, config: Dict[str, Any]) -> None:
        """Validate enum fields have valid values."""
        for field_path, valid_values in self.VALID_VALUES.items():
            value = self._get_nested_value(config, field_path)
            
            if value is None:
                continue  # Field is optional
            
            # Special handling for detection_profile - allow custom profiles
            if field_path == 'detection_profile':
                if value not in valid_values:
                    # Check if it's a custom profile defined in threshold_profiles
                    profiles = self._get_nested_value(config, 'threshold_profiles')
                    if profiles is None or value not in profiles:
                        self.errors.append(ValidationError(
                            field=field_path,
                            message=f"Invalid value '{value}'. Must be one of: {', '.join(valid_values)}",
                            severity='error'
                        ))
            elif value not in valid_values:
                self.errors.append(ValidationError(
                    field=field_path,
                    message=f"Invalid value '{value}'. Must be one of: {', '.join(valid_values)}",
                    severity='error'
                ))
    
    def _validate_profiles(self, config: Dict[str, Any]) -> None:
        """Validate threshold profiles if present."""
        if 'threshold_profiles' not in config:
            return
        
        profiles = config['threshold_profiles']
        if not isinstance(profiles, dict):
            self.errors.append(ValidationError(
                field='threshold_profiles',
                message="'threshold_profiles' must be a dictionary",
                severity='error'
            ))
            return
        
        # Validate each profile has same structure as main thresholds
        for profile_name, profile_config in profiles.items():
            if not isinstance(profile_config, dict):
                self.errors.append(ValidationError(
                    field=f'threshold_profiles.{profile_name}',
                    message=f"Profile '{profile_name}' must be a dictionary",
                    severity='error'
                ))
                continue
            
            # Check profile has all required threshold categories
            for category in self.REQUIRED_THRESHOLD_CATEGORIES:
                if category not in profile_config:
                    self.warnings.append(ValidationError(
                        field=f'threshold_profiles.{profile_name}.{category}',
                        message=f"Profile '{profile_name}' missing category '{category}'",
                        severity='warning'
                    ))
            
            # Validate ranges for profile thresholds
            for category in profile_config:
                if not isinstance(profile_config[category], dict):
                    continue
                
                for param, value in profile_config[category].items():
                    field_path = f'thresholds.{category}.{param}'
                    if field_path in self.VALUE_RANGES:
                        min_val, max_val = self.VALUE_RANGES[field_path]
                        if isinstance(value, (int, float)) and (value < min_val or value > max_val):
                            self.errors.append(ValidationError(
                                field=f'threshold_profiles.{profile_name}.{category}.{param}',
                                message=f"Value {value} is outside valid range [{min_val}, {max_val}]",
                                severity='error'
                            ))
    
    def _validate_data_paths(self, config: Dict[str, Any]) -> None:
        """Validate data paths structure."""
        if 'data_paths' not in config:
            return  # Already reported in required fields check
        
        data_paths = config['data_paths']
        if not isinstance(data_paths, dict):
            self.errors.append(ValidationError(
                field='data_paths',
                message="'data_paths' must be a dictionary",
                severity='error'
            ))
            return
        
        # Check required data source categories
        required_sources = ['sberindex', 'rosstat', 'municipal_dict']
        for source in required_sources:
            if source not in data_paths:
                self.errors.append(ValidationError(
                    field=f'data_paths.{source}',
                    message=f"Required data source '{source}' is missing",
                    severity='error'
                ))
    
    def _validate_logical_consistency(self, config: Dict[str, Any]) -> None:
        """Validate logical consistency between related fields."""
        # Check percentile bounds
        lower = self._get_nested_value(config, 'thresholds.statistical.percentile_lower')
        upper = self._get_nested_value(config, 'thresholds.statistical.percentile_upper')
        
        if lower is not None and upper is not None:
            if lower >= upper:
                self.errors.append(ValidationError(
                    field='thresholds.statistical',
                    message=f"percentile_lower ({lower}) must be less than percentile_upper ({upper})",
                    severity='error'
                ))
        
        # Check winsorization limits
        winsor_limits = self._get_nested_value(config, 'robust_statistics.winsorization_limits')
        if winsor_limits is not None:
            if not isinstance(winsor_limits, list) or len(winsor_limits) != 2:
                self.errors.append(ValidationError(
                    field='robust_statistics.winsorization_limits',
                    message="winsorization_limits must be a list of two values [lower, upper]",
                    severity='error'
                ))
            elif winsor_limits[0] >= winsor_limits[1]:
                self.errors.append(ValidationError(
                    field='robust_statistics.winsorization_limits',
                    message=f"Lower limit ({winsor_limits[0]}) must be less than upper limit ({winsor_limits[1]})",
                    severity='error'
                ))
        
        # Check auto-tuning min/max anomalies
        min_anomalies = self._get_nested_value(config, 'auto_tuning.min_anomalies_per_detector')
        max_anomalies = self._get_nested_value(config, 'auto_tuning.max_anomalies_per_detector')
        
        if min_anomalies is not None and max_anomalies is not None:
            if min_anomalies >= max_anomalies:
                self.errors.append(ValidationError(
                    field='auto_tuning',
                    message=f"min_anomalies_per_detector ({min_anomalies}) must be less than max_anomalies_per_detector ({max_anomalies})",
                    severity='error'
                ))
        
        # Check temporal drop threshold is negative
        drop_threshold = self._get_nested_value(config, 'thresholds.temporal.drop_threshold')
        if drop_threshold is not None and drop_threshold > 0:
            self.warnings.append(ValidationError(
                field='thresholds.temporal.drop_threshold',
                message=f"drop_threshold ({drop_threshold}) should typically be negative",
                severity='warning'
            ))
        

    
    def _get_nested_value(self, config: Dict[str, Any], field_path: str) -> Optional[Any]:
        """
        Get value from nested dictionary using dot notation.
        
        Args:
            config: Configuration dictionary
            field_path: Dot-separated path (e.g., 'thresholds.statistical.z_score')
            
        Returns:
            Value at the path, or None if not found
        """
        parts = field_path.split('.')
        value = config
        
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        
        return value


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
    """
    Convenience function to validate configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, list of errors/warnings)
    """
    validator = ConfigValidator()
    return validator.validate(config)


class ConfigMigrator:
    """
    Handles migration from old configuration format to new format.
    
    Old format (pre-Phase 1):
    - Only has basic fields: thresholds, export, data_paths, logging, visualization
    - No detection_profile, temporal, municipality_classification, etc.
    
    New format (Phase 1+):
    - Adds detection_profile, temporal, municipality_classification
    - Adds threshold_profiles, auto_tuning, robust_statistics
    - Adds priority_weights, missing_value_handling
    """
    
    # Fields that indicate new format
    NEW_FORMAT_INDICATORS = [
        'detection_profile',
        'temporal',
        'municipality_classification',
        'threshold_profiles',
        'auto_tuning',
        'robust_statistics',
        'priority_weights',
        'missing_value_handling'
    ]
    
    # Default values for new fields
    NEW_FIELD_DEFAULTS = {
        'detection_profile': 'normal',
        
        'temporal': {
            'enabled': False,
            'aggregation_method': 'latest',
            'auto_detect': True
        },
        
        'municipality_classification': {
            'enabled': True,
            'urban_population_threshold': 50000,
            'capital_cities': [
                "Москва",
                "Санкт-Петербург",
                "Севастополь",
                "Екатеринбург",
                "Новосибирск",
                "Казань",
                "Нижний Новгород",
                "Челябинск",
                "Самара",
                "Омск",
                "Ростов-на-Дону",
                "Уфа",
                "Красноярск",
                "Воронеж",
                "Пермь",
                "Волгоград"
            ]
        },
        
        'robust_statistics': {
            'enabled': True,
            'use_median': True,
            'use_mad': True,
            'winsorization_limits': [0.01, 0.99],
            'log_transform_skewness_threshold': 2.0
        },
        
        'priority_weights': {
            'anomaly_types': {
                'logical_inconsistency': 1.5,
                'cross_source_discrepancy': 1.2,
                'temporal_anomaly': 1.1,
                'statistical_outlier': 1.0,
                'geographic_anomaly': 0.8
            },
            'indicators': {
                'population': 1.3,
                'consumption_total': 1.2,
                'salary': 1.1,
                'default': 1.0
            }
        },
        
        'missing_value_handling': {
            'indicator_threshold': 50.0,
            'municipality_threshold': 70.0
        },
        
        'auto_tuning': {
            'enabled': False,
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'min_data_points': 100,
            'validation_confidence': 0.95,
            'export_tuned_config': True,
            'export_path': 'output/tuned_thresholds.yaml'
        }
    }
    
    def __init__(self):
        """Initialize the configuration migrator."""
        self.migration_warnings: List[str] = []
    
    def is_old_format(self, config: Dict[str, Any]) -> bool:
        """
        Detect if configuration is in old format.
        
        Old format is detected if:
        - Has required basic fields (thresholds, export, data_paths)
        - Missing any of the new format indicators
        
        Args:
            config: Configuration dictionary to check
            
        Returns:
            True if old format, False if new format
        """
        # Check if has basic required fields
        has_basic_fields = all(
            field in config 
            for field in ['thresholds', 'export', 'data_paths']
        )
        
        if not has_basic_fields:
            # Not even a valid old format
            return False
        
        # Check if missing any new format indicators
        has_new_fields = any(
            field in config 
            for field in self.NEW_FORMAT_INDICATORS
        )
        
        # Old format if has basic fields but no new fields
        return has_basic_fields and not has_new_fields
    
    def migrate(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate old configuration format to new format.
        
        Args:
            old_config: Configuration in old format
            
        Returns:
            Configuration in new format with all new fields added
        """
        self.migration_warnings = []
        
        # Start with a deep copy of old config
        new_config = deepcopy(old_config)
        
        # Add new fields with defaults
        for field, default_value in self.NEW_FIELD_DEFAULTS.items():
            if field not in new_config:
                new_config[field] = deepcopy(default_value)
                self.migration_warnings.append(
                    f"Added new field '{field}' with default value"
                )
        
        # Generate threshold profiles based on existing thresholds
        if 'threshold_profiles' not in new_config and 'thresholds' in new_config:
            new_config['threshold_profiles'] = self._generate_threshold_profiles(
                new_config['thresholds']
            )
            self.migration_warnings.append(
                "Generated threshold profiles (strict, normal, relaxed) based on existing thresholds"
            )
        
        # Update export settings with new fields
        if 'export' in new_config:
            export_defaults = {
                'generate_executive_summary': True,
                'generate_dashboard': True,
                'use_management_descriptions': True,
                'highlight_critical_threshold': 90
            }
            for field, default_value in export_defaults.items():
                if field not in new_config['export']:
                    new_config['export'][field] = default_value
        
        # Update data_processing with new fields if it exists
        if 'data_processing' in new_config:
            if 'min_data_completeness' not in new_config['data_processing']:
                new_config['data_processing']['min_data_completeness'] = 0.5
        
        # Log migration summary
        logger.info(
            f"Migrated configuration from old format to new format. "
            f"Added {len(self.migration_warnings)} new fields/sections."
        )
        
        for warning in self.migration_warnings:
            logger.warning(f"Config migration: {warning}")
        
        return new_config
    
    def _generate_threshold_profiles(self, base_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strict, normal, and relaxed profiles based on base thresholds.
        
        Args:
            base_thresholds: Base threshold configuration
            
        Returns:
            Dictionary with three profiles
        """
        profiles = {}
        
        # Normal profile = base thresholds
        profiles['normal'] = deepcopy(base_thresholds)
        
        # Strict profile = 80% of base thresholds (more sensitive)
        profiles['strict'] = self._scale_thresholds(base_thresholds, 0.8)
        
        # Relaxed profile = 120% of base thresholds (less sensitive)
        profiles['relaxed'] = self._scale_thresholds(base_thresholds, 1.2)
        
        return profiles
    
    def _scale_thresholds(self, thresholds: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """
        Scale numeric threshold values by a factor.
        
        Args:
            thresholds: Threshold configuration
            scale: Scaling factor
            
        Returns:
            Scaled threshold configuration
        """
        scaled = deepcopy(thresholds)
        
        for category, params in scaled.items():
            if not isinstance(params, dict):
                continue
            
            for param, value in params.items():
                # Skip boolean values
                if isinstance(value, bool):
                    continue
                
                if isinstance(value, (int, float)) and param != 'percentile_lower' and param != 'percentile_upper':
                    # Scale numeric values, but keep percentiles as-is
                    scaled[category][param] = round(value * scale, 2)
                elif param == 'percentile_lower':
                    # For lower percentile, scale towards 0
                    scaled[category][param] = max(0.5, round(value * scale, 1))
                elif param == 'percentile_upper':
                    # For upper percentile, scale towards 100
                    scaled[category][param] = min(99.5, 100 - round((100 - value) * scale, 1))
        
        return scaled
    
    def get_migration_warnings(self) -> List[str]:
        """
        Get list of migration warnings.
        
        Returns:
            List of warning messages
        """
        return self.migration_warnings


def migrate_config_if_needed(config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Detect and migrate old configuration format if needed.
    
    Args:
        config: Configuration dictionary (old or new format)
        
    Returns:
        Tuple of (migrated_config, was_migrated)
    """
    migrator = ConfigMigrator()
    
    if migrator.is_old_format(config):
        logger.info("Detected old configuration format. Performing automatic migration...")
        migrated_config = migrator.migrate(config)
        
        # Log all warnings
        warnings = migrator.get_migration_warnings()
        logger.warning(
            f"Configuration has been automatically migrated from old format. "
            f"{len(warnings)} changes were made. "
            f"Consider updating your config.yaml to the new format."
        )
        
        return migrated_config, True
    else:
        logger.debug("Configuration is already in new format. No migration needed.")
        return config, False
