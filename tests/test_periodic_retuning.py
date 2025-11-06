"""
Tests for periodic re-tuning functionality in AutoTuner.

This module tests the scheduling logic, history tracking, and automatic
re-tuning capabilities of the AutoTuner class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil

from src.auto_tuner import AutoTuner, TuningHistory, ThresholdOptimizationResult


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'auto_tuning': {
            'enabled': True,
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'optimization_strategy': 'adaptive'
        },
        'export': {
            'output_dir': tempfile.mkdtemp()
        },
        'thresholds': {
            'statistical': {
                'z_score': 3.0,
                'iqr_multiplier': 1.5
            },
            'geographic': {
                'regional_z_score': 2.5,
                'cluster_threshold': 2.5
            }
        }
    }


@pytest.fixture
def sample_data():
    """Create sample municipal data for testing."""
    np.random.seed(42)
    n_municipalities = 100
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'region_name': [f'Region_{i % 10}' for i in range(n_municipalities)],
        'indicator_1': np.random.normal(100, 20, n_municipalities),
        'indicator_2': np.random.normal(50, 10, n_municipalities),
        'indicator_3': np.random.normal(200, 40, n_municipalities)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def auto_tuner_with_history(sample_config):
    """Create AutoTuner with some tuning history."""
    tuner = AutoTuner(sample_config)
    
    # Add some fake history
    old_timestamp = datetime.now() - timedelta(days=35)
    
    result1 = ThresholdOptimizationResult(
        detector_name='statistical',
        original_thresholds={'z_score': 3.0},
        optimized_thresholds={'z_score': 3.3},
        estimated_fpr_before=0.027,
        estimated_fpr_after=0.013,
        anomaly_count_before=1000,
        anomaly_count_after=500,
        optimization_strategy='adaptive',
        optimization_time=old_timestamp,
        confidence_score=0.8
    )
    
    history = TuningHistory(
        tuning_id='test_tuning_001',
        timestamp=old_timestamp,
        results=[result1],
        total_anomalies_before=1000,
        total_anomalies_after=500,
        avg_fpr_before=0.027,
        avg_fpr_after=0.013
    )
    
    tuner.tuning_history.append(history)
    
    yield tuner
    
    # Cleanup
    output_dir = Path(sample_config['export']['output_dir'])
    if output_dir.exists():
        shutil.rmtree(output_dir)


def test_should_retune_no_history(sample_config):
    """Test that should_retune returns True when no history exists."""
    tuner = AutoTuner(sample_config)
    
    should_tune, reason = tuner.should_retune()
    
    assert should_tune is True
    assert 'No previous tuning history' in reason


def test_should_retune_time_based(auto_tuner_with_history):
    """Test time-based re-tuning trigger."""
    # History is 35 days old, interval is 30 days
    should_tune, reason = auto_tuner_with_history.should_retune()
    
    assert should_tune is True
    assert '35 days since last tuning' in reason
    assert '30 days' in reason


def test_should_not_retune_recent(sample_config):
    """Test that should_retune returns False for recent tuning."""
    tuner = AutoTuner(sample_config)
    
    # Add recent history (5 days ago)
    recent_timestamp = datetime.now() - timedelta(days=5)
    
    result = ThresholdOptimizationResult(
        detector_name='statistical',
        original_thresholds={'z_score': 3.0},
        optimized_thresholds={'z_score': 3.3},
        estimated_fpr_before=0.027,
        estimated_fpr_after=0.013,
        anomaly_count_before=1000,
        anomaly_count_after=500,
        optimization_strategy='adaptive',
        confidence_score=0.8
    )
    
    history = TuningHistory(
        tuning_id='test_tuning_recent',
        timestamp=recent_timestamp,
        results=[result],
        total_anomalies_before=1000,
        total_anomalies_after=500,
        avg_fpr_before=0.027,
        avg_fpr_after=0.013
    )
    
    tuner.tuning_history.append(history)
    
    should_tune, reason = tuner.should_retune()
    
    assert should_tune is False
    assert 'not needed' in reason.lower()
    assert '25 days until next scheduled tuning' in reason


def test_should_retune_fpr_degradation(sample_config):
    """Test FPR degradation trigger with force_check."""
    tuner = AutoTuner(sample_config)
    
    # Add recent history with high FPR (5 days ago)
    recent_timestamp = datetime.now() - timedelta(days=5)
    
    result = ThresholdOptimizationResult(
        detector_name='statistical',
        original_thresholds={'z_score': 3.0},
        optimized_thresholds={'z_score': 3.3},
        estimated_fpr_before=0.027,
        estimated_fpr_after=0.10,  # High FPR (target is 0.05)
        anomaly_count_before=1000,
        anomaly_count_after=500,
        optimization_strategy='adaptive',
        confidence_score=0.8
    )
    
    history = TuningHistory(
        tuning_id='test_tuning_high_fpr',
        timestamp=recent_timestamp,
        results=[result],
        total_anomalies_before=1000,
        total_anomalies_after=500,
        avg_fpr_before=0.027,
        avg_fpr_after=0.10  # High FPR
    )
    
    tuner.tuning_history.append(history)
    
    # Without force_check, should not retune (time-based)
    should_tune, reason = tuner.should_retune(force_check=False)
    assert should_tune is False
    
    # With force_check, should retune due to FPR
    should_tune, reason = tuner.should_retune(force_check=True)
    assert should_tune is True
    assert 'FPR' in reason
    assert 'exceeds target' in reason


def test_schedule_periodic_retuning_needed(auto_tuner_with_history, sample_data, sample_config):
    """Test schedule_periodic_retuning when re-tuning is needed."""
    current_thresholds = sample_config['thresholds']
    
    was_retuned, new_thresholds, message = auto_tuner_with_history.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    
    assert was_retuned is True
    assert new_thresholds is not None
    assert 'days since last tuning' in message
    
    # Verify new thresholds are different from original
    assert new_thresholds != current_thresholds


def test_schedule_periodic_retuning_not_needed(sample_config, sample_data):
    """Test schedule_periodic_retuning when re-tuning is not needed."""
    tuner = AutoTuner(sample_config)
    
    # Add recent history (5 days ago)
    recent_timestamp = datetime.now() - timedelta(days=5)
    
    result = ThresholdOptimizationResult(
        detector_name='statistical',
        original_thresholds={'z_score': 3.0},
        optimized_thresholds={'z_score': 3.3},
        estimated_fpr_before=0.027,
        estimated_fpr_after=0.013,
        anomaly_count_before=1000,
        anomaly_count_after=500,
        optimization_strategy='adaptive',
        confidence_score=0.8
    )
    
    history = TuningHistory(
        tuning_id='test_tuning_recent',
        timestamp=recent_timestamp,
        results=[result],
        total_anomalies_before=1000,
        total_anomalies_after=500,
        avg_fpr_before=0.027,
        avg_fpr_after=0.013
    )
    
    tuner.tuning_history.append(history)
    
    current_thresholds = sample_config['thresholds']
    
    was_retuned, new_thresholds, message = tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds
    )
    
    assert was_retuned is False
    assert new_thresholds == current_thresholds
    assert 'not needed' in message.lower()


def test_schedule_periodic_retuning_forced(sample_config, sample_data):
    """Test forced re-tuning regardless of schedule."""
    tuner = AutoTuner(sample_config)
    
    # Add recent history (5 days ago)
    recent_timestamp = datetime.now() - timedelta(days=5)
    
    result = ThresholdOptimizationResult(
        detector_name='statistical',
        original_thresholds={'z_score': 3.0},
        optimized_thresholds={'z_score': 3.3},
        estimated_fpr_before=0.027,
        estimated_fpr_after=0.013,
        anomaly_count_before=1000,
        anomaly_count_after=500,
        optimization_strategy='adaptive',
        confidence_score=0.8
    )
    
    history = TuningHistory(
        tuning_id='test_tuning_recent',
        timestamp=recent_timestamp,
        results=[result],
        total_anomalies_before=1000,
        total_anomalies_after=500,
        avg_fpr_before=0.027,
        avg_fpr_after=0.013
    )
    
    tuner.tuning_history.append(history)
    
    current_thresholds = sample_config['thresholds']
    
    # Force re-tuning
    was_retuned, new_thresholds, message = tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        force=True
    )
    
    assert was_retuned is True
    assert 'Forced re-tuning' in message


def test_get_tuning_history_summary_no_history(sample_config):
    """Test get_tuning_history_summary with no history."""
    tuner = AutoTuner(sample_config)
    
    summary = tuner.get_tuning_history_summary()
    
    assert summary['total_tunings'] == 0
    assert summary['last_tuning_date'] is None
    assert summary['days_since_last_tuning'] is None
    assert summary['next_scheduled_tuning'] is None
    assert summary['days_until_next_tuning'] is None
    assert summary['retuning_interval_days'] == 30
    assert len(summary['tuning_history']) == 0


def test_get_tuning_history_summary_with_history(auto_tuner_with_history):
    """Test get_tuning_history_summary with existing history."""
    summary = auto_tuner_with_history.get_tuning_history_summary()
    
    assert summary['total_tunings'] == 1
    assert summary['last_tuning_date'] is not None
    assert summary['days_since_last_tuning'] == 35
    assert summary['next_scheduled_tuning'] is not None
    assert summary['days_until_next_tuning'] == 0  # Overdue
    assert summary['retuning_interval_days'] == 30
    assert len(summary['tuning_history']) == 1
    
    # Check history entry structure
    history_entry = summary['tuning_history'][0]
    assert 'tuning_id' in history_entry
    assert 'timestamp' in history_entry
    assert 'total_anomalies_before' in history_entry
    assert 'total_anomalies_after' in history_entry
    assert 'anomaly_reduction_pct' in history_entry
    assert 'avg_fpr_before' in history_entry
    assert 'avg_fpr_after' in history_entry
    assert 'fpr_reduction_pct' in history_entry
    assert 'detectors_tuned' in history_entry


def test_get_next_tuning_date_no_history(sample_config):
    """Test get_next_tuning_date with no history."""
    tuner = AutoTuner(sample_config)
    
    next_date = tuner.get_next_tuning_date()
    
    assert next_date is None


def test_get_next_tuning_date_with_history(auto_tuner_with_history):
    """Test get_next_tuning_date with existing history."""
    next_date = auto_tuner_with_history.get_next_tuning_date()
    
    assert next_date is not None
    assert isinstance(next_date, datetime)
    
    # Should be 30 days after last tuning (which was 35 days ago)
    # So it should be in the past (5 days ago)
    assert next_date < datetime.now()


def test_tuning_history_persistence(sample_config, sample_data):
    """Test that tuning history is properly persisted and loaded."""
    tuner = AutoTuner(sample_config)
    
    # Perform a tuning
    current_thresholds = sample_config['thresholds']
    
    was_retuned, new_thresholds, message = tuner.schedule_periodic_retuning(
        df=sample_data,
        current_thresholds=current_thresholds,
        force=True
    )
    
    assert was_retuned is True
    
    # Check that history file was created
    output_dir = Path(sample_config['export']['output_dir'])
    history_file = output_dir / 'tuning_history.json'
    
    assert history_file.exists()
    
    # Load and verify history file
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
    
    assert len(history_data) > 0
    
    entry = history_data[0]
    assert 'tuning_id' in entry
    assert 'timestamp' in entry
    assert 'total_anomalies_before' in entry
    assert 'total_anomalies_after' in entry
    assert 'avg_fpr_before' in entry
    assert 'avg_fpr_after' in entry
    assert 'results' in entry
    
    # Verify results contain threshold information
    result = entry['results'][0]
    assert 'detector_name' in result
    assert 'original_thresholds' in result
    assert 'optimized_thresholds' in result
    assert 'optimization_strategy' in result
    
    # Create new tuner and verify it loads history
    tuner2 = AutoTuner(sample_config)
    
    assert len(tuner2.tuning_history) > 0
    assert tuner2.tuning_history[0].tuning_id == entry['tuning_id']


def test_multiple_tunings_history_limit(sample_config, sample_data):
    """Test that history is limited to last 10 entries."""
    tuner = AutoTuner(sample_config)
    
    current_thresholds = sample_config['thresholds']
    
    # Perform 15 tunings
    for i in range(15):
        was_retuned, new_thresholds, message = tuner.schedule_periodic_retuning(
            df=sample_data,
            current_thresholds=current_thresholds,
            force=True
        )
        current_thresholds = new_thresholds
    
    # Check that only last 10 are persisted
    output_dir = Path(sample_config['export']['output_dir'])
    history_file = output_dir / 'tuning_history.json'
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
    
    assert len(history_data) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
