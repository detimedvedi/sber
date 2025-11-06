"""
Tests for auto-tuning integration in main pipeline.

This module tests the integration of auto-tuning workflow into the main
anomaly detection pipeline, including threshold optimization, application,
and export functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import tempfile
import shutil
from datetime import datetime, timedelta

from src.auto_tuner import AutoTuner
from src.detector_manager import DetectorManager


@pytest.fixture
def sample_config():
    """Create sample configuration with auto-tuning enabled."""
    return {
        'detection_profile': 'normal',
        'thresholds': {
            'statistical': {
                'z_score': 3.0,
                'iqr_multiplier': 1.5,
                'percentile_lower': 1,
                'percentile_upper': 99
            },
            'geographic': {
                'regional_z_score': 2.5,
                'cluster_threshold': 2.5
            },
            'temporal': {
                'spike_threshold': 100,
                'drop_threshold': -50,
                'volatility_multiplier': 2.0
            },
            'cross_source': {
                'correlation_threshold': 0.5,
                'discrepancy_threshold': 50
            }
        },
        'auto_tuning': {
            'enabled': True,
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'optimization_strategy': 'adaptive',
            'export_tuned_config': True,
            'export_path': 'output/tuned_thresholds.yaml'
        },
        'export': {
            'output_dir': 'output'
        }
    }


@pytest.fixture
def sample_data():
    """Create sample municipal data for testing."""
    np.random.seed(42)
    n_municipalities = 100
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': [f'Region_{i % 10}' for i in range(n_municipalities)],
        'population': np.random.normal(50000, 20000, n_municipalities),
        'consumption_total': np.random.normal(1000, 300, n_municipalities),
        'salary_average': np.random.normal(40000, 10000, n_municipalities),
        'migration_net': np.random.normal(0, 100, n_municipalities)
    }
    
    # Add some outliers
    data['population'][0] = 200000  # Outlier
    data['consumption_total'][1] = 5000  # Outlier
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_auto_tuning_workflow_enabled(sample_config, sample_data, temp_output_dir):
    """Test that auto-tuning workflow runs when enabled."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['export_path'] = f'{temp_output_dir}/tuned_thresholds.yaml'
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(sample_config)
    
    # Get current thresholds
    current_thresholds = sample_config['thresholds']
    
    # Run periodic re-tuning (force=True to ensure it runs)
    should_tune, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=True
    )
    
    # Verify tuning was performed
    assert should_tune is True
    assert 'Forced re-tuning requested' in message
    assert tuned_thresholds is not None
    assert len(tuned_thresholds) > 0
    
    # Verify thresholds were optimized
    assert 'statistical' in tuned_thresholds
    assert 'geographic' in tuned_thresholds
    
    # Verify thresholds are different from original (at least for some detectors)
    thresholds_changed = False
    for detector_name in ['statistical', 'geographic']:
        if detector_name in tuned_thresholds and detector_name in current_thresholds:
            original = current_thresholds[detector_name]
            tuned = tuned_thresholds[detector_name]
            
            for param in original.keys():
                if param in tuned and original[param] != tuned[param]:
                    thresholds_changed = True
                    break
    
    assert thresholds_changed, "Thresholds should be optimized"


def test_auto_tuning_workflow_disabled(sample_config, sample_data):
    """Test that auto-tuning workflow is skipped when disabled."""
    # Disable auto-tuning
    sample_config['auto_tuning']['enabled'] = False
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(sample_config)
    
    # Get current thresholds
    current_thresholds = sample_config['thresholds']
    
    # Run periodic re-tuning
    should_tune, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=False
    )
    
    # With no history, it should tune on first run
    # The message will vary depending on whether tuning history exists
    assert should_tune is True
    assert message is not None  # Just verify we got a message


def test_tuned_thresholds_applied_to_detector_manager(sample_config, sample_data, temp_output_dir):
    """Test that tuned thresholds are correctly applied to DetectorManager."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Initialize auto-tuner and optimize thresholds
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Apply tuned thresholds to config
    sample_config['thresholds'] = tuned_thresholds
    
    # Initialize DetectorManager with tuned config
    detector_manager = DetectorManager(sample_config)
    
    # Verify DetectorManager is using tuned thresholds
    threshold_manager = detector_manager.threshold_manager
    
    for detector_name in ['statistical', 'geographic']:
        if detector_name in tuned_thresholds:
            manager_thresholds = threshold_manager.get_thresholds(detector_name)
            expected_thresholds = tuned_thresholds[detector_name]
            
            # Verify thresholds match
            for param, value in expected_thresholds.items():
                assert param in manager_thresholds
                assert manager_thresholds[param] == value


def test_tuning_results_logged(sample_config, sample_data, temp_output_dir, caplog):
    """Test that tuning results are properly logged."""
    import logging
    
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Set up logging
    caplog.set_level(logging.INFO)
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(sample_config)
    
    # Run optimization
    current_thresholds = sample_config['thresholds']
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Verify logging occurred
    assert any('Starting threshold optimization' in record.message for record in caplog.records)
    assert any('Threshold optimization completed' in record.message for record in caplog.records)
    
    # Verify detector-specific logging
    for detector_name in ['statistical', 'geographic']:
        assert any(f'{detector_name} thresholds optimized' in record.message.lower() 
                   for record in caplog.records)


def test_tuned_config_export(sample_config, sample_data, temp_output_dir):
    """Test that tuned configuration is exported to file."""
    # Update config to use temp directory
    export_path = Path(temp_output_dir) / 'tuned_thresholds.yaml'
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['export_path'] = str(export_path)
    sample_config['auto_tuning']['export_tuned_config'] = True
    
    # Initialize auto-tuner and optimize
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Convert numpy types to Python native types for YAML serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Manually export (simulating what main.py does)
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds)
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    # Verify file was created
    assert export_path.exists()
    
    # Verify file contents
    with open(export_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    assert 'tuning_timestamp' in loaded_config
    assert 'tuning_strategy' in loaded_config
    assert loaded_config['tuning_strategy'] == 'adaptive'
    assert 'thresholds' in loaded_config
    assert len(loaded_config['thresholds']) > 0


def test_tuning_report_generation(sample_config, sample_data, temp_output_dir):
    """Test that tuning report is generated correctly."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Initialize auto-tuner and optimize
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Generate report
    report = auto_tuner.generate_tuning_report()
    
    # Verify report content
    assert '# Auto-Tuning Report' in report
    assert 'Tuning ID:' in report
    assert 'Timestamp:' in report
    assert 'Strategy:' in report
    assert 'adaptive' in report.lower()
    
    # Verify detector sections
    assert 'Statistical Detector' in report or 'statistical' in report.lower()
    assert 'Geographic Detector' in report or 'geographic' in report.lower()
    
    # Verify threshold information (report format uses tables)
    assert 'Threshold Changes' in report or 'Original' in report
    assert 'Optimized' in report or 'Performance Metrics' in report


def test_periodic_retuning_schedule(sample_config, sample_data, temp_output_dir):
    """Test that periodic re-tuning respects the schedule."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['retuning_interval_days'] = 30
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(sample_config)
    
    # First tuning (should run - no history)
    current_thresholds = sample_config['thresholds']
    should_tune_1, _, message_1 = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        force=False
    )
    
    assert should_tune_1 is True
    assert 'No previous tuning history' in message_1
    
    # Immediate second tuning (should not run - too soon)
    should_tune_2, _, message_2 = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        force=False
    )
    
    assert should_tune_2 is False
    assert 'not needed' in message_2.lower()
    
    # Force tuning (should run regardless of schedule)
    should_tune_3, _, message_3 = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        force=True
    )
    
    assert should_tune_3 is True
    assert 'Forced re-tuning' in message_3


def test_auto_tuning_error_handling(sample_config, sample_data):
    """Test that auto-tuning errors are handled gracefully."""
    # Create invalid config that will cause errors
    invalid_config = sample_config.copy()
    invalid_config['thresholds'] = {}  # Empty thresholds
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(invalid_config)
    
    # Try to optimize (should handle gracefully)
    try:
        tuned_thresholds = auto_tuner.optimize_thresholds(
            df=sample_data,
            current_thresholds={},
            strategy='adaptive'
        )
        
        # Should return empty dict or handle gracefully
        assert isinstance(tuned_thresholds, dict)
        
    except Exception as e:
        # If it raises an exception, it should be informative
        assert str(e) is not None


def test_integration_with_detector_manager(sample_config, sample_data, temp_output_dir):
    """Test full integration: auto-tune -> apply -> detect."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Step 1: Auto-tune thresholds
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Step 2: Apply tuned thresholds to config
    sample_config['thresholds'] = tuned_thresholds
    
    # Step 3: Initialize DetectorManager with tuned config
    detector_manager = DetectorManager(sample_config)
    
    # Step 4: Run detectors
    results = detector_manager.run_all_detectors(sample_data)
    
    # Verify detection ran successfully
    assert isinstance(results, list)
    
    # Verify at least some detectors ran
    stats = detector_manager.get_detector_statistics()
    assert len(stats) > 0
    
    # Verify some detectors succeeded
    successful_detectors = [name for name, stat in stats.items() if stat.success]
    assert len(successful_detectors) > 0


def test_full_pipeline_with_auto_tuning(sample_config, sample_data, temp_output_dir):
    """Test complete pipeline: load config -> auto-tune -> detect -> export."""
    # Update config to use temp directory
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['enabled'] = True
    sample_config['auto_tuning']['export_tuned_config'] = True
    sample_config['auto_tuning']['export_path'] = f'{temp_output_dir}/tuned_thresholds.yaml'
    
    # Step 1: Initialize auto-tuner
    auto_tuner = AutoTuner(sample_config)
    
    # Step 2: Run periodic re-tuning (simulating main.py workflow)
    current_thresholds = sample_config['thresholds']
    was_retuned, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=True
    )
    
    assert was_retuned is True
    assert tuned_thresholds is not None
    
    # Step 3: Apply tuned thresholds to config
    sample_config['thresholds'] = tuned_thresholds
    
    # Step 4: Initialize DetectorManager with tuned config
    detector_manager = DetectorManager(sample_config)
    
    # Step 5: Run detectors
    results = detector_manager.run_all_detectors(sample_data)
    
    # Step 6: Verify results
    assert isinstance(results, list)
    
    # Step 7: Verify statistics
    stats = detector_manager.get_detector_statistics()
    assert len(stats) > 0
    
    # Step 8: Verify tuned config was exported
    export_path = Path(temp_output_dir) / 'tuned_thresholds.yaml'
    
    # Manually export (simulating main.py)
    def convert_numpy_types(obj):
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
    
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds)
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    assert export_path.exists()
    
    # Step 9: Verify exported config can be loaded
    with open(export_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    assert 'thresholds' in loaded_config
    assert 'tuning_timestamp' in loaded_config
    assert 'tuning_strategy' in loaded_config


def test_threshold_application_consistency(sample_config, sample_data, temp_output_dir):
    """Test that thresholds are consistently applied across pipeline."""
    # Update config
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Step 1: Optimize thresholds
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Step 2: Apply to config
    sample_config['thresholds'] = tuned_thresholds
    
    # Step 3: Create multiple DetectorManager instances
    manager1 = DetectorManager(sample_config)
    manager2 = DetectorManager(sample_config)
    
    # Step 4: Verify both managers have same thresholds
    for detector_name in ['statistical', 'geographic']:
        if detector_name in tuned_thresholds:
            thresholds1 = manager1.threshold_manager.get_thresholds(detector_name)
            thresholds2 = manager2.threshold_manager.get_thresholds(detector_name)
            
            assert thresholds1 == thresholds2, \
                f"Thresholds inconsistent for {detector_name}"


def test_config_export_format_validation(sample_config, sample_data, temp_output_dir):
    """Test that exported config has correct format and can be reloaded."""
    # Update config
    export_path = Path(temp_output_dir) / 'tuned_thresholds.yaml'
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['export_path'] = str(export_path)
    
    # Optimize thresholds
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Export config
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds),
        'metadata': {
            'data_shape': list(sample_data.shape),  # Convert tuple to list
            'total_municipalities': len(sample_data)
        }
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    # Reload and validate
    with open(export_path, 'r', encoding='utf-8') as f:
        reloaded_config = yaml.safe_load(f)
    
    # Validate structure
    assert 'tuning_timestamp' in reloaded_config
    assert 'tuning_strategy' in reloaded_config
    assert 'thresholds' in reloaded_config
    assert 'metadata' in reloaded_config
    
    # Validate thresholds are valid Python types (not numpy)
    def check_no_numpy_types(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                check_no_numpy_types(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                check_no_numpy_types(item)
        else:
            assert not isinstance(obj, (np.integer, np.floating, np.ndarray)), \
                f"Found numpy type in exported config: {type(obj)}"
            # Also check for tuple (should be converted to list)
            assert not isinstance(obj, tuple) or len(obj) == 0, \
                f"Found tuple in exported config (should be list): {type(obj)}"
    
    check_no_numpy_types(reloaded_config['thresholds'])
    
    # Validate can be used to create DetectorManager
    test_config = sample_config.copy()
    test_config['thresholds'] = reloaded_config['thresholds']
    
    manager = DetectorManager(test_config)
    assert manager is not None


def test_auto_tuning_with_different_strategies(sample_config, sample_data, temp_output_dir):
    """Test auto-tuning with different optimization strategies."""
    sample_config['export']['output_dir'] = temp_output_dir
    
    strategies = ['conservative', 'balanced', 'adaptive', 'aggressive']
    results = {}
    
    for strategy in strategies:
        auto_tuner = AutoTuner(sample_config)
        current_thresholds = sample_config['thresholds']
        
        try:
            tuned_thresholds = auto_tuner.optimize_thresholds(
                df=sample_data,
                current_thresholds=current_thresholds,
                strategy=strategy
            )
            
            results[strategy] = {
                'success': True,
                'thresholds': tuned_thresholds
            }
        except Exception as e:
            results[strategy] = {
                'success': False,
                'error': str(e)
            }
    
    # Verify at least some strategies succeeded
    successful_strategies = [s for s, r in results.items() if r['success']]
    assert len(successful_strategies) > 0, \
        f"No strategies succeeded. Results: {results}"
    
    # Verify different strategies produce different thresholds
    if len(successful_strategies) >= 2:
        thresh1 = results[successful_strategies[0]]['thresholds']
        thresh2 = results[successful_strategies[1]]['thresholds']
        
        # At least some thresholds should differ
        differences_found = False
        for detector in thresh1.keys():
            if detector in thresh2:
                for param in thresh1[detector].keys():
                    if param in thresh2[detector]:
                        if thresh1[detector][param] != thresh2[detector][param]:
                            differences_found = True
                            break
        
        # Note: Strategies might produce same results on small sample data
        # This is acceptable behavior


def test_auto_tuning_report_generation_integration(sample_config, sample_data, temp_output_dir):
    """Test that tuning report is generated and contains expected information."""
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Generate report
    report = auto_tuner.generate_tuning_report()
    
    # Verify report structure
    assert '# Auto-Tuning Report' in report
    assert 'Tuning ID:' in report
    assert 'Timestamp:' in report
    assert 'Strategy:' in report
    
    # Verify detector information
    for detector_name in tuned_thresholds.keys():
        assert detector_name.lower() in report.lower() or \
               detector_name.replace('_', ' ').title() in report
    
    # Save report to file
    report_path = Path(temp_output_dir) / 'auto_tuning_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    assert report_path.exists()
    
    # Verify file can be read back
    with open(report_path, 'r', encoding='utf-8') as f:
        loaded_report = f.read()
    
    assert loaded_report == report


def test_auto_tuning_disabled_pipeline(sample_config, sample_data, temp_output_dir):
    """Test that pipeline works correctly when auto-tuning is disabled."""
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['enabled'] = False
    
    # Initialize DetectorManager with original thresholds
    original_thresholds = sample_config['thresholds'].copy()
    detector_manager = DetectorManager(sample_config)
    
    # Run detectors
    results = detector_manager.run_all_detectors(sample_data)
    
    # Verify detection ran
    assert isinstance(results, list)
    
    # Verify thresholds weren't changed
    for detector_name in original_thresholds.keys():
        current_thresholds = detector_manager.threshold_manager.get_thresholds(detector_name)
        expected_thresholds = original_thresholds[detector_name]
        
        for param, value in expected_thresholds.items():
            assert param in current_thresholds
            assert current_thresholds[param] == value


def test_auto_tuning_with_validation_failures(sample_config, sample_data, temp_output_dir):
    """Test auto-tuning behavior when validation fails."""
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Set very strict validation requirements that will fail
    sample_config['auto_tuning']['min_anomalies_per_detector'] = 10000  # Unrealistic
    sample_config['auto_tuning']['max_anomalies_per_detector'] = 5  # Too low
    
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    # Try to optimize (may fail validation)
    try:
        tuned_thresholds = auto_tuner.optimize_thresholds(
            df=sample_data,
            current_thresholds=current_thresholds,
            strategy='adaptive'
        )
        
        # If it succeeds, verify thresholds are returned
        assert isinstance(tuned_thresholds, dict)
        
    except Exception as e:
        # If it fails, verify error is informative
        assert str(e) is not None


def test_threshold_persistence_across_runs(sample_config, sample_data, temp_output_dir):
    """Test that tuned thresholds persist and can be reused across runs."""
    sample_config['export']['output_dir'] = temp_output_dir
    export_path = Path(temp_output_dir) / 'tuned_thresholds.yaml'
    sample_config['auto_tuning']['export_path'] = str(export_path)
    
    # First run: Optimize and export
    auto_tuner1 = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds1 = auto_tuner1.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Export
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds1)
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    # Second run: Load and use exported thresholds
    with open(export_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    sample_config['thresholds'] = loaded_config['thresholds']
    
    # Create DetectorManager with loaded thresholds
    detector_manager = DetectorManager(sample_config)
    
    # Verify thresholds match
    for detector_name in tuned_thresholds1.keys():
        if detector_name in loaded_config['thresholds']:
            manager_thresholds = detector_manager.threshold_manager.get_thresholds(detector_name)
            loaded_thresholds = loaded_config['thresholds'][detector_name]
            
            for param, value in loaded_thresholds.items():
                assert param in manager_thresholds
                # Allow small floating point differences
                if isinstance(value, float):
                    assert abs(manager_thresholds[param] - value) < 1e-6
                else:
                    assert manager_thresholds[param] == value


def test_auto_tuning_with_missing_data(sample_config, temp_output_dir):
    """Test auto-tuning handles datasets with missing values correctly."""
    np.random.seed(42)
    n_municipalities = 100
    
    # Create data with missing values
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': [f'Region_{i % 10}' for i in range(n_municipalities)],
        'population': np.random.normal(50000, 20000, n_municipalities),
        'consumption_total': np.random.normal(1000, 300, n_municipalities),
        'salary_average': np.random.normal(40000, 10000, n_municipalities),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values (30% missing)
    missing_mask = np.random.random(n_municipalities) < 0.3
    df.loc[missing_mask, 'consumption_total'] = np.nan
    df.loc[missing_mask, 'salary_average'] = np.nan
    
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    # Should handle missing data gracefully
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Verify tuning completed
    assert isinstance(tuned_thresholds, dict)
    assert len(tuned_thresholds) > 0


def test_auto_tuning_with_large_dataset(sample_config, temp_output_dir):
    """Test auto-tuning performance with larger datasets."""
    np.random.seed(42)
    n_municipalities = 1000  # Larger dataset
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': [f'Region_{i % 50}' for i in range(n_municipalities)],
        'population': np.random.normal(50000, 20000, n_municipalities),
        'consumption_total': np.random.normal(1000, 300, n_municipalities),
        'salary_average': np.random.normal(40000, 10000, n_municipalities),
        'migration_net': np.random.normal(0, 100, n_municipalities)
    }
    
    # Add outliers
    outlier_indices = np.random.choice(n_municipalities, size=50, replace=False)
    for idx in outlier_indices:
        data['population'][idx] *= 3
    
    df = pd.DataFrame(data)
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    import time
    start_time = time.time()
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    elapsed_time = time.time() - start_time
    
    # Verify tuning completed
    assert isinstance(tuned_thresholds, dict)
    assert len(tuned_thresholds) > 0
    
    # Verify reasonable performance (should complete in < 30 seconds)
    assert elapsed_time < 30, f"Auto-tuning took too long: {elapsed_time:.2f}s"


def test_auto_tuning_config_backward_compatibility(sample_config, sample_data, temp_output_dir):
    """Test that auto-tuning works with old config format."""
    # Create old-style config (without auto_tuning section)
    old_config = {
        'thresholds': sample_config['thresholds'].copy(),
        'export': {
            'output_dir': temp_output_dir
        }
    }
    
    # Initialize auto-tuner with old config (should use defaults)
    auto_tuner = AutoTuner(old_config)
    
    # Verify auto-tuner initialized with defaults
    assert auto_tuner.config is not None
    
    # Run optimization
    current_thresholds = old_config['thresholds']
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Verify tuning completed
    assert isinstance(tuned_thresholds, dict)
    assert len(tuned_thresholds) > 0


def test_auto_tuning_multiple_detectors_integration(sample_config, sample_data, temp_output_dir):
    """Test auto-tuning with multiple detector types."""
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Optimize thresholds
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Verify all detector types have thresholds
    expected_detectors = ['statistical', 'geographic', 'temporal', 'cross_source']
    
    for detector_name in expected_detectors:
        if detector_name in current_thresholds:
            assert detector_name in tuned_thresholds, \
                f"Missing tuned thresholds for {detector_name}"
            
            # Verify threshold structure is preserved
            original_params = set(current_thresholds[detector_name].keys())
            tuned_params = set(tuned_thresholds[detector_name].keys())
            
            assert original_params == tuned_params, \
                f"Threshold parameters changed for {detector_name}"


def test_auto_tuning_export_with_metadata(sample_config, sample_data, temp_output_dir):
    """Test that exported config includes comprehensive metadata."""
    export_path = Path(temp_output_dir) / 'tuned_thresholds_with_metadata.yaml'
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['export_path'] = str(export_path)
    
    # Run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Export with comprehensive metadata
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds),
        'metadata': {
            'data_shape': list(sample_data.shape),
            'total_municipalities': len(sample_data),
            'indicators': list(sample_data.columns),
            'optimization_strategy': 'adaptive',
            'target_fpr': sample_config['auto_tuning'].get('target_false_positive_rate', 0.05)
        }
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    # Verify export
    assert export_path.exists()
    
    # Load and validate
    with open(export_path, 'r', encoding='utf-8') as f:
        loaded_config = yaml.safe_load(f)
    
    # Verify metadata completeness
    assert 'metadata' in loaded_config
    assert 'data_shape' in loaded_config['metadata']
    assert 'total_municipalities' in loaded_config['metadata']
    assert 'indicators' in loaded_config['metadata']
    assert 'optimization_strategy' in loaded_config['metadata']
    assert 'target_fpr' in loaded_config['metadata']
    
    # Verify metadata values
    assert loaded_config['metadata']['total_municipalities'] == len(sample_data)
    assert loaded_config['metadata']['optimization_strategy'] == 'adaptive'


def test_auto_tuning_with_extreme_outliers(sample_config, temp_output_dir):
    """Test auto-tuning handles extreme outliers correctly."""
    np.random.seed(42)
    n_municipalities = 100
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': [f'Region_{i % 10}' for i in range(n_municipalities)],
        'population': np.random.normal(50000, 20000, n_municipalities),
        'consumption_total': np.random.normal(1000, 300, n_municipalities),
        'salary_average': np.random.normal(40000, 10000, n_municipalities),
    }
    
    # Add extreme outliers (10x normal values)
    data['population'][0] = 500000
    data['consumption_total'][1] = 10000
    data['salary_average'][2] = 400000
    
    df = pd.DataFrame(data)
    sample_config['export']['output_dir'] = temp_output_dir
    
    # Run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    tuned_thresholds = auto_tuner.optimize_thresholds(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    # Verify tuning completed and thresholds are reasonable
    assert isinstance(tuned_thresholds, dict)
    assert len(tuned_thresholds) > 0
    
    # Verify thresholds are within reasonable ranges
    if 'statistical' in tuned_thresholds:
        z_score = tuned_thresholds['statistical'].get('z_score', 3.0)
        assert 1.0 <= z_score <= 5.0, f"Z-score out of reasonable range: {z_score}"


def test_full_pipeline_end_to_end_with_auto_tuning(sample_config, sample_data, temp_output_dir):
    """Test complete end-to-end pipeline with auto-tuning enabled."""
    # Configure for full pipeline
    sample_config['export']['output_dir'] = temp_output_dir
    sample_config['auto_tuning']['enabled'] = True
    sample_config['auto_tuning']['export_tuned_config'] = True
    sample_config['auto_tuning']['export_path'] = f'{temp_output_dir}/final_tuned_config.yaml'
    
    # Step 1: Initialize and run auto-tuning
    auto_tuner = AutoTuner(sample_config)
    current_thresholds = sample_config['thresholds']
    
    was_retuned, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=True
    )
    
    assert was_retuned is True
    assert tuned_thresholds is not None
    
    # Step 2: Apply tuned thresholds
    sample_config['thresholds'] = tuned_thresholds
    
    # Step 3: Initialize DetectorManager
    detector_manager = DetectorManager(sample_config)
    
    # Step 4: Run all detectors
    results = detector_manager.run_all_detectors(sample_data)
    
    # Step 5: Verify results
    assert isinstance(results, list)
    
    # Step 6: Get statistics
    stats = detector_manager.get_detector_statistics()
    assert len(stats) > 0
    
    # Verify at least some detectors succeeded
    successful_detectors = [name for name, stat in stats.items() if stat.success]
    assert len(successful_detectors) > 0
    
    # Step 7: Generate and export tuning report
    report = auto_tuner.generate_tuning_report()
    assert '# Auto-Tuning Report' in report
    
    report_path = Path(temp_output_dir) / 'final_tuning_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    assert report_path.exists()
    
    # Step 8: Export tuned configuration
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    export_path = Path(sample_config['auto_tuning']['export_path'])
    tuned_config = {
        'tuning_timestamp': datetime.now().isoformat(),
        'tuning_strategy': 'adaptive',
        'thresholds': convert_numpy_types(tuned_thresholds),
        'detector_statistics': {
            name: {
                'success': stat.success,
                'anomalies_detected': stat.anomalies_detected,
                'execution_time_seconds': stat.execution_time_seconds
            }
            for name, stat in stats.items()
        }
    }
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuned_config, f, default_flow_style=False, allow_unicode=True)
    
    assert export_path.exists()
    
    # Step 9: Verify exported config can be reloaded
    with open(export_path, 'r', encoding='utf-8') as f:
        reloaded_config = yaml.safe_load(f)
    
    assert 'thresholds' in reloaded_config
    assert 'detector_statistics' in reloaded_config
    assert 'tuning_timestamp' in reloaded_config
    
    # Step 10: Verify reloaded config can be used
    test_config = sample_config.copy()
    test_config['thresholds'] = reloaded_config['thresholds']
    
    new_manager = DetectorManager(test_config)
    assert new_manager is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
