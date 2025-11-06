"""
Demonstration of Periodic Re-tuning Functionality

This script demonstrates how to use the periodic re-tuning features
of the AutoTuner class in the anomaly detection system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_tuner import AutoTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n_municipalities=1000):
    """Create sample municipal data for demonstration."""
    np.random.seed(42)
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'region_name': [f'Region_{i % 20}' for i in range(n_municipalities)],
        'population': np.random.lognormal(10, 1, n_municipalities),
        'consumption_total': np.random.lognormal(8, 1.5, n_municipalities),
        'salary_average': np.random.normal(50000, 15000, n_municipalities),
        'market_access': np.random.uniform(0, 100, n_municipalities)
    }
    
    return pd.DataFrame(data)


def main():
    """Demonstrate periodic re-tuning functionality."""
    
    # Configuration
    config = {
        'auto_tuning': {
            'enabled': True,
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'retuning_interval_days': 30,
            'optimization_strategy': 'adaptive'
        },
        'export': {
            'output_dir': 'output'
        },
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
                'discrepancy_threshold': 50,
                'correlation_threshold': 0.5
            }
        }
    }
    
    # Create sample data
    logger.info("Creating sample municipal data...")
    df = create_sample_data(n_municipalities=1000)
    logger.info(f"Created data with {len(df)} municipalities")
    
    # Initialize AutoTuner
    logger.info("Initializing AutoTuner...")
    auto_tuner = AutoTuner(config)
    
    # Example 1: Check if re-tuning is needed
    logger.info("\n" + "="*60)
    logger.info("Example 1: Check Re-tuning Status")
    logger.info("="*60)
    
    should_tune, reason = auto_tuner.should_retune(force_check=True)
    logger.info(f"Should re-tune: {should_tune}")
    logger.info(f"Reason: {reason}")
    
    # Example 2: Get tuning history summary
    logger.info("\n" + "="*60)
    logger.info("Example 2: Tuning History Summary")
    logger.info("="*60)
    
    summary = auto_tuner.get_tuning_history_summary()
    logger.info(f"Total tunings performed: {summary['total_tunings']}")
    logger.info(f"Last tuning date: {summary['last_tuning_date']}")
    logger.info(f"Days since last tuning: {summary['days_since_last_tuning']}")
    logger.info(f"Next scheduled tuning: {summary['next_scheduled_tuning']}")
    logger.info(f"Days until next tuning: {summary['days_until_next_tuning']}")
    logger.info(f"Re-tuning interval: {summary['retuning_interval_days']} days")
    
    # Example 3: Schedule periodic re-tuning
    logger.info("\n" + "="*60)
    logger.info("Example 3: Schedule Periodic Re-tuning")
    logger.info("="*60)
    
    current_thresholds = config['thresholds']
    
    was_retuned, new_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=True  # Force for demonstration
    )
    
    logger.info(f"Was re-tuned: {was_retuned}")
    logger.info(f"Message: {message}")
    
    if was_retuned:
        logger.info("\nThreshold Changes:")
        for detector_name in new_thresholds:
            logger.info(f"\n{detector_name.upper()} Detector:")
            original = current_thresholds.get(detector_name, {})
            optimized = new_thresholds.get(detector_name, {})
            
            for param_name in optimized:
                orig_val = original.get(param_name, 'N/A')
                new_val = optimized.get(param_name, 'N/A')
                
                if orig_val != 'N/A' and new_val != 'N/A':
                    change_pct = ((new_val - orig_val) / orig_val) * 100
                    logger.info(
                        f"  {param_name}: {orig_val:.3f} → {new_val:.3f} "
                        f"({change_pct:+.1f}%)"
                    )
    
    # Example 4: Get updated tuning history
    logger.info("\n" + "="*60)
    logger.info("Example 4: Updated Tuning History")
    logger.info("="*60)
    
    summary = auto_tuner.get_tuning_history_summary()
    logger.info(f"Total tunings performed: {summary['total_tunings']}")
    
    if summary['tuning_history']:
        logger.info("\nRecent Tuning History:")
        for i, entry in enumerate(summary['tuning_history'], 1):
            logger.info(f"\nTuning #{i}:")
            logger.info(f"  ID: {entry['tuning_id']}")
            logger.info(f"  Timestamp: {entry['timestamp']}")
            logger.info(
                f"  Anomalies: {entry['total_anomalies_before']} → "
                f"{entry['total_anomalies_after']} "
                f"({entry['anomaly_reduction_pct']:.1f}% reduction)"
            )
            logger.info(
                f"  FPR: {entry['avg_fpr_before']:.3f} → "
                f"{entry['avg_fpr_after']:.3f} "
                f"({entry['fpr_reduction_pct']:.1f}% reduction)"
            )
            logger.info(f"  Detectors tuned: {', '.join(entry['detectors_tuned'])}")
    
    # Example 5: Get next tuning date
    logger.info("\n" + "="*60)
    logger.info("Example 5: Next Scheduled Tuning")
    logger.info("="*60)
    
    next_date = auto_tuner.get_next_tuning_date()
    if next_date:
        logger.info(f"Next tuning scheduled for: {next_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        days_until = (next_date - datetime.now()).days
        if days_until > 0:
            logger.info(f"Days until next tuning: {days_until}")
        else:
            logger.info(f"Next tuning is overdue by {abs(days_until)} days")
    else:
        logger.info("No tuning history available")
    
    # Example 6: Demonstrate automatic re-tuning in pipeline
    logger.info("\n" + "="*60)
    logger.info("Example 6: Integration in Main Pipeline")
    logger.info("="*60)
    
    logger.info("\nTypical usage in main.py:")
    logger.info("""
    # At the start of analysis pipeline
    auto_tuner = AutoTuner(config)
    
    # Check and perform periodic re-tuning if needed
    was_retuned, thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=merged_data,
        current_thresholds=config['thresholds'],
        strategy='adaptive'
    )
    
    if was_retuned:
        logger.info(f"Thresholds were re-tuned: {message}")
        # Update config with new thresholds
        config['thresholds'] = thresholds
    else:
        logger.info(f"Using existing thresholds: {message}")
    
    # Continue with normal detection pipeline...
    """)
    
    logger.info("\n" + "="*60)
    logger.info("Demonstration Complete")
    logger.info("="*60)
    logger.info(f"\nTuning history saved to: output/tuning_history.json")


if __name__ == '__main__':
    main()
