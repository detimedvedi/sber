"""
Main entry point for СберИндекс Anomaly Detection System

This script orchestrates the entire anomaly detection pipeline:
1. Load configuration
2. Load and merge data from СберИндекс and Росстат
3. Run all anomaly detectors
4. Aggregate and rank results
5. Export results to CSV/Excel with visualizations
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

from src.config_validator import validate_config


def setup_logging(config: dict) -> None:
    """
    Configure structured logging based on config settings.
    
    Sets up logging with support for structured data through extra fields,
    proper log levels, and context information.
    
    Args:
        config: Configuration dictionary with logging settings
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Enhanced format with support for structured data
    # Uses %(message)s for main message, extra fields can be added programmatically
    log_format = log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    log_file = log_config.get('file', 'output/anomaly_detection.log')
    
    # Ensure output directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            'log_level': logging.getLevelName(log_level),
            'log_file': log_file,
            'format': 'structured'
        }
    )


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration
        is_valid, issues = validate_config(config)
        
        if not is_valid:
            print("Configuration validation failed:")
            for issue in issues:
                if issue.severity == 'error':
                    print(f"  ERROR - {issue.field}: {issue.message}")
            print("\nPlease fix the configuration errors and try again.")
            sys.exit(1)
        
        # Log warnings but continue
        warnings = [issue for issue in issues if issue.severity == 'warning']
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  WARNING - {warning.field}: {warning.message}")
        
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def validate_output_files(output_dir: str, logger: logging.Logger) -> bool:
    """
    Validate that output files were created successfully.
    
    Checks for the existence of expected output files and logs any missing files.
    
    Args:
        output_dir: Directory where output files should be located
        logger: Logger instance for logging validation results
        
    Returns:
        True if all expected files exist, False otherwise
    """
    import os
    import glob
    
    logger.info("Validating output files...")
    
    all_valid = True
    
    # Check for CSV files
    csv_files = glob.glob(os.path.join(output_dir, "anomalies_master_*.csv"))
    if csv_files:
        logger.info(f"✓ Found {len(csv_files)} master CSV file(s)")
    else:
        logger.warning("✗ No master CSV files found")
        all_valid = False
    
    # Check for Excel files
    excel_files = glob.glob(os.path.join(output_dir, "anomalies_summary_*.xlsx"))
    if excel_files:
        logger.info(f"✓ Found {len(excel_files)} summary Excel file(s)")
    else:
        logger.warning("✗ No summary Excel files found")
        all_valid = False
    
    # Check for visualization files
    viz_files = glob.glob(os.path.join(output_dir, "viz_*.png"))
    if viz_files:
        logger.info(f"✓ Found {len(viz_files)} visualization file(s)")
    else:
        logger.warning("✗ No visualization files found")
        all_valid = False
    
    # Check for report/documentation files
    doc_files = glob.glob(os.path.join(output_dir, "report_*.txt")) + \
                glob.glob(os.path.join(output_dir, "report_*.md"))
    if doc_files:
        logger.info(f"✓ Found {len(doc_files)} documentation file(s)")
    else:
        logger.warning("✗ No documentation files found")
        all_valid = False
    
    # Check for log file
    log_file = os.path.join(output_dir, "anomaly_detection.log")
    if os.path.exists(log_file):
        logger.info(f"✓ Log file exists: {log_file}")
    else:
        logger.warning(f"✗ Log file not found: {log_file}")
        all_valid = False
    
    if all_valid:
        logger.info("All expected output files validated successfully")
    else:
        logger.warning("Some expected output files are missing")
    
    return all_valid


def main():
    """Main execution function"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("СберИндекс Anomaly Detection System")
    print("=" * 80)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting anomaly detection pipeline")
    logger.info(f"Configuration loaded from config.yaml")
    
    # Initialize statistics tracking
    pipeline_stats = {
        'start_time': start_time,
        'steps_completed': [],
        'steps_failed': [],
        'warnings': [],
        'data_loaded': {},
        'anomalies_detected': {},
        'files_exported': []
    }
    
    # Import components
    try:
        from src.data_loader import DataLoader
        from src.detector_manager import DetectorManager
        from src.results_aggregator import ResultsAggregator
        from src.exporter import ResultsExporter
        logger.info("All components imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import required components: {e}", exc_info=True)
        print(f"\n✗ Error: Failed to import components - {e}")
        print("Check that all required modules are installed and available.")
        sys.exit(1)
    
    # Step 1: Load data
    unified_df = None
    validation_results = None
    
    try:
        logger.info("=" * 80)
        logger.info(
            "Step 1: Loading data...",
            extra={
                'step': 'data_loading',
                'phase': 'start'
            }
        )
        print("Step 1: Loading data...")
        
        data_loader = DataLoader()
        
        # Load СберИндекс data with error handling
        try:
            sberindex_data = data_loader.load_sberindex_data()
            pipeline_stats['data_loaded']['sberindex'] = {
                'connection': len(sberindex_data.get('connection', [])),
                'consumption': len(sberindex_data.get('consumption', [])),
                'market_access': len(sberindex_data.get('market_access', []))
            }
            logger.info(
                "СберИндекс data loaded successfully",
                extra={
                    'data_source': 'sberindex',
                    'connection_rows': len(sberindex_data.get('connection', [])),
                    'consumption_rows': len(sberindex_data.get('consumption', [])),
                    'market_access_rows': len(sberindex_data.get('market_access', []))
                }
            )
        except Exception as e:
            logger.error(
                "Error loading СберИндекс data",
                exc_info=True,
                extra={
                    'data_source': 'sberindex',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            pipeline_stats['warnings'].append(f"СберИндекс data loading failed: {e}")
            sberindex_data = {}
            print(f"  ⚠ Warning: Could not load СберИндекс data - {e}")
        
        # Load Росстат data with error handling
        try:
            rosstat_data = data_loader.load_rosstat_data()
            pipeline_stats['data_loaded']['rosstat'] = {
                'population': len(rosstat_data.get('population', [])),
                'migration': len(rosstat_data.get('migration', [])),
                'salary': len(rosstat_data.get('salary', []))
            }
            logger.info(
                "Росстат data loaded successfully",
                extra={
                    'data_source': 'rosstat',
                    'population_rows': len(rosstat_data.get('population', [])),
                    'migration_rows': len(rosstat_data.get('migration', [])),
                    'salary_rows': len(rosstat_data.get('salary', []))
                }
            )
        except Exception as e:
            logger.error(
                "Error loading Росстат data",
                exc_info=True,
                extra={
                    'data_source': 'rosstat',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            pipeline_stats['warnings'].append(f"Росстат data loading failed: {e}")
            rosstat_data = {}
            print(f"  ⚠ Warning: Could not load Росстат data - {e}")
        
        # Load municipal dictionary with error handling
        try:
            municipal_dict = data_loader.load_municipal_dict()
            pipeline_stats['data_loaded']['municipal_dict'] = len(municipal_dict)
            logger.info(
                "Municipal dictionary loaded",
                extra={
                    'data_source': 'municipal_dict',
                    'entries_count': len(municipal_dict)
                }
            )
        except Exception as e:
            logger.error(
                "Error loading municipal dictionary",
                exc_info=True,
                extra={
                    'data_source': 'municipal_dict',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            pipeline_stats['warnings'].append(f"Municipal dictionary loading failed: {e}")
            municipal_dict = None
            print(f"  ⚠ Warning: Could not load municipal dictionary - {e}")
        
        # BUGFIX: Load connection graph explicitly for GeographicAnomalyDetector
        try:
            connections = data_loader.load_connection_data()
            pipeline_stats['data_loaded']['connections'] = len(connections) if not connections.empty else 0
            if not connections.empty:
                logger.info(
                    "Connection graph loaded",
                    extra={
                        'data_source': 'connection_graph',
                        'connections_count': len(connections)
                    }
                )
                print(f"  ✓ Connection graph loaded: {len(connections):,} connections")
            else:
                logger.warning("Connection graph is empty - geographic analysis will use fallback")
                print(f"  ⚠ Warning: Connection graph is empty")
        except Exception as e:
            logger.error(
                "Error loading connection graph",
                exc_info=True,
                extra={
                    'data_source': 'connection_graph',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            )
            pipeline_stats['warnings'].append(f"Connection graph loading failed: {e}")
            connections = pd.DataFrame()
            print(f"  ⚠ Warning: Could not load connection graph - {e}")
        
        # Check if we have enough data to continue
        if not sberindex_data and not rosstat_data:
            logger.critical(
                "No data sources available - cannot continue",
                extra={
                    'sberindex_available': bool(sberindex_data),
                    'rosstat_available': bool(rosstat_data),
                    'step': 'data_loading'
                }
            )
            print(f"\n✗ Error: No data sources could be loaded. Cannot continue.")
            sys.exit(1)
        
        # Merge datasets with error handling
        try:
            unified_df = data_loader.merge_datasets(sberindex_data, rosstat_data, municipal_dict)
            validation_results = data_loader.validate_data(unified_df)
            
            # Create explicit source mapping for all columns
            source_mapping = data_loader.create_source_mapping(unified_df.columns.tolist())
            logger.info(
                "Created source mapping for columns",
                extra={
                    'total_columns': len(source_mapping),
                    'sberindex_columns': sum(1 for s in source_mapping.values() if s == 'sberindex'),
                    'rosstat_columns': sum(1 for s in source_mapping.values() if s == 'rosstat'),
                    'unknown_columns': sum(1 for s in source_mapping.values() if s == 'unknown')
                }
            )
            
            # Detect and log duplicates
            duplicate_report = data_loader.detect_duplicates(unified_df)
            if duplicate_report.duplicate_count > 0:
                logger.warning(
                    f"DATA QUALITY WARNING: Found {duplicate_report.duplicate_count} duplicate territory_id entries",
                    extra={
                        'duplicate_count': duplicate_report.duplicate_count,
                        'affected_territories': len(duplicate_report.affected_territories),
                        'is_temporal': duplicate_report.is_temporal,
                        'recommendation': duplicate_report.recommendation,
                        'data_quality_issue': 'duplicates'
                    }
                )
                print(f"  ⚠ Data Quality Warning: {duplicate_report.duplicate_count} duplicate territory_ids found")
                print(f"    Recommendation: {duplicate_report.recommendation}")
                pipeline_stats['warnings'].append(f"Duplicate territory_ids: {duplicate_report.duplicate_count}")
            
            # Log missing value statistics
            if validation_results['missing_values']:
                total_missing = validation_results['summary']['total_missing']
                columns_with_missing = len(validation_results['missing_values'])
                logger.warning(
                    f"DATA QUALITY WARNING: Missing values detected in {columns_with_missing} columns",
                    extra={
                        'total_missing_values': total_missing,
                        'columns_with_missing': columns_with_missing,
                        'overall_completeness': validation_results['summary']['overall_completeness'],
                        'data_quality_issue': 'missing_values'
                    }
                )
                print(f"  ⚠ Data Quality Warning: Missing values in {columns_with_missing} columns ({total_missing} total)")
                pipeline_stats['warnings'].append(f"Missing values: {total_missing} in {columns_with_missing} columns")
            
            pipeline_stats['data_loaded']['unified'] = {
                'municipalities': len(unified_df),
                'indicators': len(unified_df.columns),
                'completeness': validation_results['summary']['overall_completeness']
            }
            
            logger.info(
                "Data merged successfully",
                extra={
                    'municipalities_count': len(unified_df),
                    'indicators_count': len(unified_df.columns),
                    'data_completeness': validation_results['summary']['overall_completeness'],
                    'missing_values': validation_results['summary']['total_missing'],
                    'step': 'data_loading'
                }
            )
            print(f"  ✓ Loaded {len(unified_df)} municipalities with {len(unified_df.columns)} indicators")
            print(f"  ✓ Data completeness: {validation_results['summary']['overall_completeness']:.2%}")
            
            pipeline_stats['steps_completed'].append('data_loading')
            
        except Exception as e:
            logger.error(
                "Error merging datasets",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'data_loading'
                }
            )
            pipeline_stats['steps_failed'].append(('data_loading', str(e)))
            print(f"\n✗ Error: Failed to merge datasets - {e}")
            sys.exit(1)
            
    except Exception as e:
        logger.critical(
            "Unexpected error in data loading step",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step': 'data_loading',
                'phase': 'unexpected_error'
            }
        )
        pipeline_stats['steps_failed'].append(('data_loading', str(e)))
        print(f"\n✗ Error in data loading step: {e}")
        print("Check the log file for details.")
        sys.exit(1)
        
    # Step 2: Run anomaly detection
    all_anomalies = []
    
    try:
        logger.info("=" * 80)
        logger.info(
            "Step 2: Running anomaly detectors...",
            extra={
                'step': 'anomaly_detection',
                'phase': 'start',
                'input_municipalities': len(unified_df)
            }
        )
        print("\nStep 2: Running anomaly detectors...")
        
        # Check if auto-tuning is enabled
        auto_tuning_config = config.get('auto_tuning', {})
        auto_tuning_enabled = auto_tuning_config.get('enabled', False)
        
        if auto_tuning_enabled:
            logger.info("Auto-tuning is enabled - running threshold optimization before detection")
            print("  → Auto-tuning enabled: Optimizing thresholds...")
            
            try:
                from src.auto_tuner import AutoTuner
                
                # Initialize auto-tuner
                auto_tuner = AutoTuner(config)
                
                # Get current thresholds
                current_thresholds = config.get('thresholds', {})
                
                # Check if periodic re-tuning is needed
                should_tune, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
                    df=unified_df,
                    current_thresholds=current_thresholds,
                    strategy=auto_tuning_config.get('optimization_strategy', 'adaptive'),
                    force=False
                )
                
                if should_tune:
                    logger.info(
                        f"Auto-tuning completed: {message}",
                        extra={
                            'auto_tuning': True,
                            'tuning_performed': True,
                            'step': 'auto_tuning'
                        }
                    )
                    print(f"    ✓ Auto-tuning completed: {message}")
                    
                    # Apply tuned thresholds to config
                    config['thresholds'] = tuned_thresholds
                    
                    # Log tuned threshold values
                    logger.info("Tuned thresholds applied:")
                    for detector_name, thresholds in tuned_thresholds.items():
                        logger.info(
                            f"  {detector_name}: {thresholds}",
                            extra={
                                'detector': detector_name,
                                'tuned_thresholds': thresholds,
                                'step': 'auto_tuning'
                            }
                        )
                        print(f"      - {detector_name}: {thresholds}")
                    
                    # Export tuned configuration if enabled
                    if auto_tuning_config.get('export_tuned_config', True):
                        export_path = auto_tuning_config.get('export_path', 'output/tuned_thresholds.yaml')
                        try:
                            import yaml
                            from pathlib import Path
                            import numpy as np
                            
                            # Ensure output directory exists
                            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                            
                            # Convert numpy types to Python native types for YAML serialization
                            def convert_numpy_types(obj):
                                """Recursively convert numpy types to Python native types."""
                                if isinstance(obj, dict):
                                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy_types(item) for item in obj]
                                elif isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                else:
                                    return obj
                            
                            # Export tuned thresholds
                            tuned_config = {
                                'tuning_timestamp': datetime.now().isoformat(),
                                'tuning_strategy': auto_tuning_config.get('optimization_strategy', 'adaptive'),
                                'thresholds': convert_numpy_types(tuned_thresholds)
                            }
                            
                            with open(export_path, 'w', encoding='utf-8') as f:
                                yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
                            
                            logger.info(f"Tuned configuration exported to {export_path}")
                            print(f"    ✓ Tuned configuration exported to {export_path}")
                            pipeline_stats['files_exported'].append(export_path)
                            
                        except Exception as e:
                            logger.error(
                                f"Failed to export tuned configuration: {e}",
                                exc_info=True,
                                extra={
                                    'error_type': type(e).__name__,
                                    'error_message': str(e),
                                    'step': 'auto_tuning'
                                }
                            )
                            print(f"    ⚠ Warning: Failed to export tuned configuration - {e}")
                    
                    # Generate tuning report
                    try:
                        tuning_report = auto_tuner.generate_tuning_report()
                        report_path = Path(config['export']['output_dir']) / 'auto_tuning_report.md'
                        report_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(report_path, 'w', encoding='utf-8') as f:
                            f.write(tuning_report)
                        
                        logger.info(f"Auto-tuning report generated: {report_path}")
                        print(f"    ✓ Auto-tuning report generated: {report_path}")
                        pipeline_stats['files_exported'].append(str(report_path))
                        
                    except Exception as e:
                        logger.error(
                            f"Failed to generate tuning report: {e}",
                            exc_info=True,
                            extra={
                                'error_type': type(e).__name__,
                                'error_message': str(e),
                                'step': 'auto_tuning'
                            }
                        )
                        print(f"    ⚠ Warning: Failed to generate tuning report - {e}")
                    
                    pipeline_stats['steps_completed'].append('auto_tuning')
                    
                else:
                    logger.info(
                        f"Auto-tuning skipped: {message}",
                        extra={
                            'auto_tuning': True,
                            'tuning_performed': False,
                            'reason': message,
                            'step': 'auto_tuning'
                        }
                    )
                    print(f"    ℹ Auto-tuning skipped: {message}")
                
            except Exception as e:
                logger.error(
                    "Auto-tuning failed - continuing with default thresholds",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'auto_tuning'
                    }
                )
                print(f"    ⚠ Warning: Auto-tuning failed - {e}")
                print(f"    → Continuing with default thresholds")
                pipeline_stats['warnings'].append(f"Auto-tuning failed: {e}")
        else:
            logger.info("Auto-tuning is disabled - using configured thresholds")
            print("  → Auto-tuning disabled: Using configured thresholds")
        
        # Initialize DetectorManager with source mapping and temporal data status
        # BUGFIX: Pass is_temporal flag to skip duplicate checks for temporal data
        detector_manager = DetectorManager(
            config, 
            source_mapping, 
            is_temporal=duplicate_report.is_temporal
        )
        
        # Run all detectors through the manager (BUGFIX: pass connections for geographic analysis)
        all_anomalies = detector_manager.run_all_detectors(unified_df, connections=connections)
        
        # Get detector statistics for reporting
        detector_stats = detector_manager.get_detector_statistics()
        
        # Log detector execution times summary
        total_detection_time = sum(stats.execution_time_seconds for stats in detector_stats.values())
        logger.info(
            "Detector execution times summary",
            extra={
                'total_detection_time': round(total_detection_time, 2),
                'detector_count': len(detector_stats),
                'step': 'anomaly_detection'
            }
        )
        
        # Update pipeline stats from detector statistics
        for detector_name, stats in detector_stats.items():
            if stats.success:
                pipeline_stats['anomalies_detected'][detector_name] = stats.anomalies_detected
                logger.info(
                    f"{detector_name} detector completed successfully",
                    extra={
                        'detector': detector_name,
                        'anomalies_detected': stats.anomalies_detected,
                        'execution_time_seconds': round(stats.execution_time_seconds, 2),
                        'success': True,
                        'step': 'anomaly_detection'
                    }
                )
                print(f"  ✓ {detector_name}: {stats.anomalies_detected} anomalies detected ({stats.execution_time_seconds:.2f}s)")
            else:
                pipeline_stats['warnings'].append(f"{detector_name} detector failed: {stats.error_message}")
                logger.warning(
                    f"{detector_name} detector failed",
                    extra={
                        'detector': detector_name,
                        'error_message': stats.error_message,
                        'execution_time_seconds': round(stats.execution_time_seconds, 2),
                        'success': False,
                        'step': 'anomaly_detection'
                    }
                )
                print(f"  ⚠ Warning: {detector_name} detector failed - {stats.error_message}")
        
        # Check if we have any anomalies to continue
        if not all_anomalies or all(len(a) == 0 for a in all_anomalies):
            logger.warning(
                "No anomalies detected by any detector",
                extra={
                    'step': 'anomaly_detection',
                    'total_detectors': len(detector_stats),
                    'successful_detectors': sum(1 for s in detector_stats.values() if s.success)
                }
            )
            pipeline_stats['warnings'].append("No anomalies detected")
            print(f"  ⚠ Warning: No anomalies detected by any detector")
        
        pipeline_stats['steps_completed'].append('anomaly_detection')
        
    except Exception as e:
        logger.error(
            "Unexpected error in anomaly detection step",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step': 'anomaly_detection',
                'phase': 'unexpected_error'
            }
        )
        pipeline_stats['steps_failed'].append(('anomaly_detection', str(e)))
        print(f"\n✗ Error in anomaly detection step: {e}")
        print("Continuing with available data...")
        # Continue with empty anomalies list if detection fails completely
        if not all_anomalies:
            all_anomalies = []
        
    # Step 3: Aggregate results
    combined_anomalies = None
    municipality_scores = None
    ranked_anomalies = None
    categorized_anomalies = None
    summary_stats = None
    
    try:
        logger.info("=" * 80)
        logger.info(
            "Step 3: Aggregating results...",
            extra={
                'step': 'aggregation',
                'phase': 'start',
                'detector_results_count': len(all_anomalies)
            }
        )
        print("\nStep 3: Aggregating results...")
        
        aggregator = ResultsAggregator(config)
        
        try:
            combined_anomalies = aggregator.combine_anomalies(all_anomalies)
            logger.info(
                "Combined anomalies from all detectors",
                extra={
                    'combined_count': len(combined_anomalies),
                    'step': 'aggregation',
                    'operation': 'combine'
                }
            )
        except Exception as e:
            logger.error(
                "Error combining anomalies",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'aggregation',
                    'operation': 'combine'
                }
            )
            pipeline_stats['warnings'].append(f"Anomaly combination failed: {e}")
            print(f"  ⚠ Warning: Failed to combine anomalies - {e}")
            # Create empty dataframe to continue
            import pandas as pd
            combined_anomalies = pd.DataFrame()
        
        # BUGFIX: Apply legitimate pattern filter after aggregation
        if len(combined_anomalies) > 0:
            try:
                from src.legitimate_pattern_filter import LegitimatePatternFilter
                
                logger.info("Applying legitimate pattern filter...")
                print("  → Applying legitimate pattern filter...")
                pattern_filter = LegitimatePatternFilter()
                
                # Filter anomalies
                filtered_df = pattern_filter.filter_anomalies(combined_anomalies)
                
                # Count reclassified
                legitimate_count = (filtered_df['is_legitimate_pattern'] == True).sum()
                logger.info(
                    f"Reclassified {legitimate_count} anomalies as legitimate patterns",
                    extra={
                        'legitimate_count': legitimate_count,
                        'step': 'aggregation',
                        'operation': 'legitimate_pattern_filter'
                    }
                )
                print(f"    ✓ Reclassified {legitimate_count} anomalies as legitimate patterns")
                
                # Remove legitimate patterns
                combined_anomalies = filtered_df[
                    filtered_df['is_legitimate_pattern'] == False
                ].copy()
                
                logger.info(
                    f"After filtering: {len(combined_anomalies)} anomalies remain",
                    extra={
                        'remaining_anomalies': len(combined_anomalies),
                        'step': 'aggregation',
                        'operation': 'legitimate_pattern_filter'
                    }
                )
                print(f"    ✓ After filtering: {len(combined_anomalies)} anomalies remain")
                
            except ImportError:
                logger.warning("LegitimatePatternFilter not found - skipping pattern filtering")
                print(f"  ⚠ Warning: LegitimatePatternFilter not found - skipping")
            except Exception as e:
                logger.warning(
                    f"Failed to apply pattern filter: {e}",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'legitimate_pattern_filter'
                    }
                )
                print(f"  ⚠ Warning: Failed to apply pattern filter - {e}")
                pipeline_stats['warnings'].append(f"Pattern filter failed: {e}")
        
        if len(combined_anomalies) > 0:
            try:
                # Add priority scores to anomalies
                combined_anomalies = aggregator.add_priority_scores(combined_anomalies)
                logger.info(
                    "Added priority scores to anomalies",
                    extra={
                        'anomalies_scored': len(combined_anomalies),
                        'step': 'aggregation',
                        'operation': 'priority_scoring'
                    }
                )
            except Exception as e:
                logger.error(
                    "Error adding priority scores",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'priority_scoring'
                    }
                )
                pipeline_stats['warnings'].append(f"Priority scoring failed: {e}")
                print(f"  ⚠ Warning: Failed to add priority scores - {e}")
            
            try:
                municipality_scores = aggregator.calculate_municipality_scores(combined_anomalies)
                logger.info(
                    "Calculated municipality scores",
                    extra={
                        'municipalities_scored': len(municipality_scores),
                        'step': 'aggregation',
                        'operation': 'calculate_scores'
                    }
                )
            except Exception as e:
                logger.error(
                    "Error calculating municipality scores",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'calculate_scores'
                    }
                )
                pipeline_stats['warnings'].append(f"Municipality score calculation failed: {e}")
                print(f"  ⚠ Warning: Failed to calculate municipality scores - {e}")
                import pandas as pd
                municipality_scores = pd.DataFrame()
            
            try:
                ranked_anomalies = aggregator.rank_anomalies(combined_anomalies)
                logger.info(
                    "Ranked anomalies by severity",
                    extra={
                        'ranked_count': len(ranked_anomalies),
                        'step': 'aggregation',
                        'operation': 'rank'
                    }
                )
            except Exception as e:
                logger.error(
                    "Error ranking anomalies",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'rank'
                    }
                )
                pipeline_stats['warnings'].append(f"Anomaly ranking failed: {e}")
                print(f"  ⚠ Warning: Failed to rank anomalies - {e}")
                ranked_anomalies = combined_anomalies
            
            try:
                categorized_anomalies = aggregator.categorize_anomalies(combined_anomalies)
                logger.info(
                    "Categorized anomalies by type",
                    extra={
                        'category_count': len(categorized_anomalies),
                        'step': 'aggregation',
                        'operation': 'categorize'
                    }
                )
            except Exception as e:
                logger.error(
                    "Error categorizing anomalies",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'categorize'
                    }
                )
                pipeline_stats['warnings'].append(f"Anomaly categorization failed: {e}")
                print(f"  ⚠ Warning: Failed to categorize anomalies - {e}")
                categorized_anomalies = {}
            
            try:
                summary_stats = aggregator.get_summary_statistics(combined_anomalies)
                logger.info(
                    "Generated summary statistics",
                    extra={
                        'total_anomalies': summary_stats.get('total_anomalies', 0),
                        'municipalities_affected': summary_stats.get('total_municipalities_affected', 0),
                        'step': 'aggregation',
                        'operation': 'summary_stats'
                    }
                )
            except Exception as e:
                logger.error(
                    "Error generating summary statistics",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'summary_stats'
                    }
                )
                pipeline_stats['warnings'].append(f"Summary statistics generation failed: {e}")
                print(f"  ⚠ Warning: Failed to generate summary statistics - {e}")
                summary_stats = {}
            
            # Calculate detection metrics
            try:
                total_municipalities = len(unified_df) if unified_df is not None else 0
                detection_metrics = aggregator.calculate_detection_metrics(
                    combined_anomalies,
                    total_municipalities
                )
                
                # Add detection metrics to summary stats
                summary_stats['detection_metrics'] = detection_metrics
                
                logger.info(
                    "Calculated detection metrics",
                    extra={
                        'total_anomalies': detection_metrics.get('total_anomalies', 0),
                        'municipalities_affected': detection_metrics.get('municipalities_affected', 0),
                        'municipalities_affected_pct': detection_metrics.get('municipalities_affected_pct', 0),
                        'anomaly_rate_per_1000': detection_metrics.get('anomaly_rate_per_1000', 0),
                        'step': 'aggregation',
                        'operation': 'detection_metrics'
                    }
                )
                
                # Log key metrics
                print(f"  ✓ Detection metrics calculated:")
                print(f"    - Municipalities affected: {detection_metrics['municipalities_affected']} "
                      f"({detection_metrics['municipalities_affected_pct']:.1f}%)")
                print(f"    - Anomaly rate per 1000: {detection_metrics['anomaly_rate_per_1000']:.2f}")
                print(f"    - Critical anomalies: {detection_metrics['anomalies_by_severity']['critical']}")
                
                # Check for anomaly count warnings
                try:
                    warnings = aggregator.check_anomaly_count_warnings(detection_metrics, config)
                    
                    if warnings:
                        print(f"  ⚠ Anomaly count warnings ({len(warnings)}):")
                        
                        # Group warnings by severity
                        critical_warnings = [w for w in warnings if w['severity'] == 'critical']
                        warning_level = [w for w in warnings if w['severity'] == 'warning']
                        info_level = [w for w in warnings if w['severity'] == 'info']
                        
                        # Display critical warnings first
                        for warning in critical_warnings:
                            print(f"    🔴 CRITICAL: {warning['message']}")
                            print(f"       → {warning['recommendation']}")
                        
                        # Display warnings
                        for warning in warning_level:
                            print(f"    ⚠️  WARNING: {warning['message']}")
                            print(f"       → {warning['recommendation']}")
                        
                        # Display info (only first 3 to avoid clutter)
                        for warning in info_level[:3]:
                            print(f"    ℹ️  INFO: {warning['message']}")
                            print(f"       → {warning['recommendation']}")
                        
                        if len(info_level) > 3:
                            print(f"    ... и еще {len(info_level) - 3} информационных предупреждений")
                        
                        # Add warnings to pipeline stats
                        pipeline_stats['warnings'].extend([w['message'] for w in warnings])
                        
                        logger.info(
                            "Anomaly count warnings generated",
                            extra={
                                'warning_count': len(warnings),
                                'critical_count': len(critical_warnings),
                                'warning_count_level': len(warning_level),
                                'info_count': len(info_level),
                                'step': 'aggregation',
                                'operation': 'anomaly_count_warnings'
                            }
                        )
                    else:
                        print(f"  ✓ Anomaly counts are within expected ranges")
                        logger.info("No anomaly count warnings - metrics within expected ranges")
                    
                except Exception as e:
                    logger.error(
                        "Error checking anomaly count warnings",
                        exc_info=True,
                        extra={
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'step': 'aggregation',
                            'operation': 'anomaly_count_warnings'
                        }
                    )
                    print(f"  ⚠ Warning: Failed to check anomaly count warnings - {e}")
                
            except Exception as e:
                logger.error(
                    "Error calculating detection metrics",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'detection_metrics'
                    }
                )
                pipeline_stats['warnings'].append(f"Detection metrics calculation failed: {e}")
                print(f"  ⚠ Warning: Failed to calculate detection metrics - {e}")
            
            # Calculate data quality metrics
            try:
                data_quality_metrics = aggregator.calculate_data_quality_metrics(
                    unified_df,
                    validation_results
                )
                
                # Add data quality metrics to summary stats
                summary_stats['data_quality_metrics'] = data_quality_metrics
                
                logger.info(
                    "Calculated data quality metrics",
                    extra={
                        'data_completeness_score': data_quality_metrics.get('data_completeness_score', 0),
                        'consistency_score': data_quality_metrics.get('consistency_score', 0),
                        'quality_grade': data_quality_metrics.get('quality_grade', 'N/A'),
                        'quality_issues_count': len(data_quality_metrics.get('quality_issues', [])),
                        'step': 'aggregation',
                        'operation': 'data_quality_metrics'
                    }
                )
                
                # Log key metrics
                print(f"  ✓ Data quality metrics calculated:")
                print(f"    - Data completeness: {data_quality_metrics['data_completeness_score']:.1%}")
                print(f"    - Consistency score: {data_quality_metrics['consistency_score']:.1%}")
                print(f"    - Quality grade: {data_quality_metrics['quality_grade']}")
                if data_quality_metrics.get('quality_issues'):
                    print(f"    - Quality issues: {len(data_quality_metrics['quality_issues'])}")
                
            except Exception as e:
                logger.error(
                    "Error calculating data quality metrics",
                    exc_info=True,
                    extra={
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'step': 'aggregation',
                        'operation': 'data_quality_metrics'
                    }
                )
                pipeline_stats['warnings'].append(f"Data quality metrics calculation failed: {e}")
                print(f"  ⚠ Warning: Failed to calculate data quality metrics - {e}")
            
            logger.info(
                "Aggregation completed",
                extra={
                    'total_anomalies': len(combined_anomalies),
                    'municipalities_affected': len(municipality_scores) if municipality_scores is not None else 0,
                    'step': 'aggregation',
                    'phase': 'complete'
                }
            )
            print(f"  ✓ Combined {len(combined_anomalies)} unique anomalies")
            print(f"  ✓ {len(municipality_scores) if municipality_scores is not None else 0} municipalities affected")
        else:
            logger.warning(
                "No anomalies to aggregate",
                extra={
                    'step': 'aggregation',
                    'combined_count': 0
                }
            )
            print(f"  ⚠ Warning: No anomalies to aggregate")
            import pandas as pd
            municipality_scores = pd.DataFrame()
            ranked_anomalies = pd.DataFrame()
            categorized_anomalies = {}
            summary_stats = {}
        
        pipeline_stats['steps_completed'].append('aggregation')
        
    except Exception as e:
        logger.error(
            "Unexpected error in aggregation step",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step': 'aggregation',
                'phase': 'unexpected_error'
            }
        )
        pipeline_stats['steps_failed'].append(('aggregation', str(e)))
        print(f"\n✗ Error in aggregation step: {e}")
        print("Continuing with available data...")
        # Create empty structures to continue
        import pandas as pd
        if combined_anomalies is None:
            combined_anomalies = pd.DataFrame()
        if municipality_scores is None:
            municipality_scores = pd.DataFrame()
        if ranked_anomalies is None:
            ranked_anomalies = combined_anomalies
        if categorized_anomalies is None:
            categorized_anomalies = {}
        if summary_stats is None:
            summary_stats = {}
        
    # Step 4: Export results
    try:
        logger.info("=" * 80)
        logger.info(
            "Step 4: Exporting results...",
            extra={
                'step': 'export',
                'phase': 'start',
                'anomalies_to_export': len(ranked_anomalies) if ranked_anomalies is not None else 0
            }
        )
        print("\nStep 4: Exporting results...")
        
        exporter = ResultsExporter(config)
        
        # Export master CSV
        try:
            print("  → Exporting master CSV...")
            csv_path = exporter.export_master_csv(ranked_anomalies)
            pipeline_stats['files_exported'].append(csv_path)
            logger.info(
                "Master CSV exported successfully",
                extra={
                    'file_path': csv_path,
                    'records_exported': len(ranked_anomalies) if ranked_anomalies is not None else 0,
                    'step': 'export',
                    'format': 'csv'
                }
            )
            print(f"    ✓ Master CSV: {csv_path}")
        except Exception as e:
            logger.error(
                "Error exporting master CSV",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'csv'
                }
            )
            pipeline_stats['warnings'].append(f"CSV export failed: {e}")
            print(f"    ⚠ Warning: Failed to export master CSV - {e}")
        
        # Export summary Excel
        try:
            print("  → Exporting summary Excel...")
            excel_path = exporter.export_summary_excel(
                ranked_anomalies,
                categorized_anomalies,
                municipality_scores,
                summary_stats
            )
            pipeline_stats['files_exported'].append(excel_path)
            logger.info(
                "Summary Excel exported successfully",
                extra={
                    'file_path': excel_path,
                    'sheets_count': len(categorized_anomalies) + 3,  # categories + overview + top municipalities + dictionary
                    'step': 'export',
                    'format': 'excel'
                }
            )
            print(f"    ✓ Summary Excel: {excel_path}")
        except Exception as e:
            logger.error(
                "Error exporting summary Excel",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'excel'
                }
            )
            pipeline_stats['warnings'].append(f"Excel export failed: {e}")
            print(f"    ⚠ Warning: Failed to export summary Excel - {e}")
        
        # Generate visualizations
        try:
            print("  → Generating visualizations...")
            viz_files = exporter.generate_visualizations(
                ranked_anomalies,
                municipality_scores,
                summary_stats
            )
            pipeline_stats['files_exported'].extend(viz_files)
            logger.info(
                "Visualizations generated successfully",
                extra={
                    'visualizations_count': len(viz_files),
                    'step': 'export',
                    'format': 'visualization'
                }
            )
            print(f"    ✓ Generated {len(viz_files)} visualizations")
        except Exception as e:
            logger.error(
                "Error generating visualizations",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'visualization'
                }
            )
            pipeline_stats['warnings'].append(f"Visualization generation failed: {e}")
            print(f"    ⚠ Warning: Failed to generate visualizations - {e}")
        
        # Generate dashboard summary
        try:
            print("  → Generating dashboard summary...")
            dashboard_file = exporter.create_dashboard_summary(
                ranked_anomalies,
                municipality_scores,
                summary_stats
            )
            pipeline_stats['files_exported'].append(dashboard_file)
            logger.info(
                "Dashboard summary generated successfully",
                extra={
                    'dashboard_file': dashboard_file,
                    'step': 'export',
                    'format': 'dashboard'
                }
            )
            print(f"    ✓ Generated dashboard summary")
        except Exception as e:
            logger.error(
                "Error generating dashboard summary",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'dashboard'
                }
            )
            pipeline_stats['warnings'].append(f"Dashboard generation failed: {e}")
            print(f"    ⚠ Warning: Failed to generate dashboard summary - {e}")
        
        # Generate documentation
        print("  → Generating documentation...")
        
        # Generate methodology document
        try:
            methodology_path = exporter.generate_methodology_document(config)
            pipeline_stats['files_exported'].append(methodology_path)
            logger.info(
                "Methodology document generated",
                extra={
                    'file_path': methodology_path,
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            print(f"    ✓ Methodology: {methodology_path}")
        except Exception as e:
            logger.error(
                "Error generating methodology document",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            pipeline_stats['warnings'].append(f"Methodology document generation failed: {e}")
            print(f"    ⚠ Warning: Failed to generate methodology document - {e}")
        
        # Generate example cases
        try:
            examples_path = exporter.generate_example_cases(ranked_anomalies)
            pipeline_stats['files_exported'].append(examples_path)
            logger.info(
                "Example cases generated",
                extra={
                    'file_path': examples_path,
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            print(f"    ✓ Examples: {examples_path}")
        except Exception as e:
            logger.error(
                "Error generating example cases",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            pipeline_stats['warnings'].append(f"Example cases generation failed: {e}")
            print(f"    ⚠ Warning: Failed to generate example cases - {e}")
        
        # Generate README
        try:
            readme_path = exporter.generate_readme(summary_stats, config)
            pipeline_stats['files_exported'].append(readme_path)
            logger.info(
                "README generated",
                extra={
                    'file_path': readme_path,
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            print(f"    ✓ README: {readme_path}")
        except Exception as e:
            logger.error(
                "Error generating README",
                exc_info=True,
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'step': 'export',
                    'format': 'documentation'
                }
            )
            pipeline_stats['warnings'].append(f"README generation failed: {e}")
            print(f"    ⚠ Warning: Failed to generate README - {e}")
        
        pipeline_stats['steps_completed'].append('export')
        
    except Exception as e:
        logger.error(
            "Unexpected error in export step",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step': 'export',
                'phase': 'unexpected_error'
            }
        )
        pipeline_stats['steps_failed'].append(('export', str(e)))
        print(f"\n✗ Error in export step: {e}")
        print("Some output files may not have been created.")
        
    # Validate output files
    try:
        logger.info("=" * 80)
        logger.info(
            "Step 5: Validating output files...",
            extra={
                'step': 'validation',
                'phase': 'start',
                'files_to_validate': len(pipeline_stats['files_exported'])
            }
        )
        print("\nStep 5: Validating output files...")
        
        output_dir = config['export']['output_dir']
        validation_passed = validate_output_files(output_dir, logger)
        
        if validation_passed:
            logger.info(
                "All output files validated successfully",
                extra={
                    'step': 'validation',
                    'validation_passed': True,
                    'files_validated': len(pipeline_stats['files_exported'])
                }
            )
            print(f"  ✓ All output files validated successfully")
            pipeline_stats['steps_completed'].append('validation')
        else:
            logger.warning(
                "Some output files may be missing",
                extra={
                    'step': 'validation',
                    'validation_passed': False,
                    'files_expected': len(pipeline_stats['files_exported'])
                }
            )
            print(f"  ⚠ Some output files may be missing (see log for details)")
            pipeline_stats['warnings'].append("Some output files missing")
            
    except Exception as e:
        logger.error(
            "Error during output validation",
            exc_info=True,
            extra={
                'error_type': type(e).__name__,
                'error_message': str(e),
                'step': 'validation'
            }
        )
        pipeline_stats['warnings'].append(f"Output validation failed: {e}")
        print(f"  ⚠ Warning: Could not validate output files - {e}")
    
    # Calculate execution time and generate final summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    pipeline_stats['end_time'] = end_time
    pipeline_stats['execution_time'] = execution_time
    
    # Log comprehensive summary statistics with structured data
    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "Pipeline execution completed",
        extra={
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time.total_seconds(),
            'steps_completed': len(pipeline_stats['steps_completed']),
            'steps_failed': len(pipeline_stats['steps_failed']),
            'warnings_count': len(pipeline_stats['warnings']),
            'files_exported': len(pipeline_stats['files_exported']),
            'total_anomalies': len(combined_anomalies) if combined_anomalies is not None else 0,
            'phase': 'summary'
        }
    )
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {execution_time}")
    logger.info("")
    logger.info(f"Steps completed: {', '.join(pipeline_stats['steps_completed'])}")
    if pipeline_stats['steps_failed']:
        logger.warning(
            f"Steps failed: {', '.join([step for step, _ in pipeline_stats['steps_failed']])}",
            extra={
                'failed_steps': [step for step, _ in pipeline_stats['steps_failed']],
                'phase': 'summary'
            }
        )
    logger.info("")
    logger.info("Data Loading Summary:")
    for source, data in pipeline_stats['data_loaded'].items():
        logger.info(f"  - {source}: {data}")
    logger.info("")
    logger.info("Anomaly Detection Summary:")
    for detector, count in pipeline_stats['anomalies_detected'].items():
        logger.info(
            f"  - {detector}: {count} anomalies",
            extra={
                'detector': detector,
                'anomalies_detected': count,
                'phase': 'summary'
            }
        )
    logger.info(f"  - Total anomalies: {len(combined_anomalies) if combined_anomalies is not None else 0}")
    logger.info("")
    logger.info(f"Files exported: {len(pipeline_stats['files_exported'])}")
    for file_path in pipeline_stats['files_exported']:
        logger.debug(f"  - {file_path}")
    logger.info("")
    if pipeline_stats['warnings']:
        logger.info(f"Warnings ({len(pipeline_stats['warnings'])}):")
        for warning in pipeline_stats['warnings']:
            logger.warning(f"  - {warning}")
    logger.info("")
    if pipeline_stats['steps_failed']:
        logger.info(f"Errors ({len(pipeline_stats['steps_failed'])}):")
        for step, error in pipeline_stats['steps_failed']:
            logger.error(
                f"  - {step}: {error}",
                extra={
                    'failed_step': step,
                    'error_message': error,
                    'phase': 'summary'
                }
            )
    logger.info("=" * 80)
    
    # Print user-friendly summary
    print()
    print("=" * 80)
    if not pipeline_stats['steps_failed']:
        print("✓ Pipeline completed successfully!")
        logger.info(
            "Pipeline completed successfully",
            extra={
                'success': True,
                'execution_time_seconds': execution_time.total_seconds(),
                'phase': 'complete'
            }
        )
    else:
        print("⚠ Pipeline completed with errors")
        logger.warning(
            "Pipeline completed with errors",
            extra={
                'success': False,
                'errors_count': len(pipeline_stats['steps_failed']),
                'execution_time_seconds': execution_time.total_seconds(),
                'phase': 'complete'
            }
        )
    print("=" * 80)
    print(f"Results saved to: {config['export']['output_dir']}/")
    print(f"\nExecution Summary:")
    print(f"  • Execution time: {execution_time}")
    print(f"  • Steps completed: {len(pipeline_stats['steps_completed'])}/{len(pipeline_stats['steps_completed']) + len(pipeline_stats['steps_failed'])}")
    
    if combined_anomalies is not None and len(combined_anomalies) > 0:
        print(f"\nAnomaly Detection Results:")
        print(f"  • Total anomalies detected: {len(combined_anomalies)}")
        if municipality_scores is not None and len(municipality_scores) > 0:
            print(f"  • Municipalities affected: {len(municipality_scores)}")
        if categorized_anomalies:
            print(f"  • Anomaly types: {len(categorized_anomalies)}")
        if municipality_scores is not None and len(municipality_scores) > 0:
            top_muni = municipality_scores.iloc[0]
            print(f"  • Most anomalous municipality: {top_muni['municipal_name']}")
            print(f"    ({top_muni['total_anomalies_count']} anomalies, severity: {top_muni['total_severity_score']:.1f})")
    
    if pipeline_stats['files_exported']:
        print(f"\nOutput Files:")
        print(f"  • {len(pipeline_stats['files_exported'])} files created")
    
    if pipeline_stats['warnings']:
        print(f"\nWarnings: {len(pipeline_stats['warnings'])}")
        print(f"  (See log file for details)")
    
    if pipeline_stats['steps_failed']:
        print(f"\nErrors: {len(pipeline_stats['steps_failed'])}")
        for step, error in pipeline_stats['steps_failed']:
            print(f"  • {step}: {error}")
        print(f"  (See log file for full details)")
    
    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {config['export']['output_dir']}/anomaly_detection.log")
    print("=" * 80)
    
    # Exit with appropriate code
    if pipeline_stats['steps_failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
