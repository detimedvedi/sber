"""
Auto-Tuning Workflow Demo

This script demonstrates the auto-tuning workflow integration in the main pipeline.
It shows how auto-tuning optimizes thresholds before detection runs.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_tuner import AutoTuner
from src.detector_manager import DetectorManager


def create_sample_data(n_municipalities=200):
    """Create sample municipal data for demonstration."""
    np.random.seed(42)
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': [f'Region_{i % 15}' for i in range(n_municipalities)],
        'population': np.random.lognormal(10.5, 0.8, n_municipalities),
        'consumption_total': np.random.lognormal(7.0, 0.6, n_municipalities),
        'consumption_electricity': np.random.lognormal(6.5, 0.5, n_municipalities),
        'salary_average': np.random.lognormal(10.8, 0.4, n_municipalities),
        'salary_Финансы': np.random.lognormal(11.5, 0.5, n_municipalities),
        'migration_net': np.random.normal(0, 200, n_municipalities),
        'market_access_score': np.random.uniform(0, 100, n_municipalities)
    }
    
    # Add some outliers to make it interesting
    outlier_indices = np.random.choice(n_municipalities, size=10, replace=False)
    for idx in outlier_indices:
        data['population'][idx] *= 3.0
        data['consumption_total'][idx] *= 2.5
    
    return pd.DataFrame(data)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def demo_auto_tuning_workflow():
    """Demonstrate the auto-tuning workflow."""
    print("=" * 80)
    print("Auto-Tuning Workflow Demo")
    print("=" * 80)
    print()
    
    # Step 1: Load configuration
    print("Step 1: Loading configuration...")
    config = load_config()
    
    # Enable auto-tuning for demo
    config['auto_tuning']['enabled'] = True
    config['auto_tuning']['optimization_strategy'] = 'adaptive'
    print(f"  ✓ Auto-tuning enabled: {config['auto_tuning']['enabled']}")
    print(f"  ✓ Strategy: {config['auto_tuning']['optimization_strategy']}")
    print(f"  ✓ Target FPR: {config['auto_tuning']['target_false_positive_rate']}")
    print()
    
    # Step 2: Create sample data
    print("Step 2: Creating sample data...")
    df = create_sample_data(n_municipalities=200)
    print(f"  ✓ Created {len(df)} municipalities with {len(df.columns)} indicators")
    print()
    
    # Step 3: Run auto-tuning workflow
    print("Step 3: Running auto-tuning workflow...")
    print()
    
    # Initialize auto-tuner
    auto_tuner = AutoTuner(config)
    
    # Get current thresholds
    current_thresholds = config['thresholds']
    print("  Current thresholds:")
    for detector_name, thresholds in current_thresholds.items():
        print(f"    {detector_name}:")
        for param, value in thresholds.items():
            print(f"      - {param}: {value}")
    print()
    
    # Check if re-tuning is needed
    print("  Checking if re-tuning is needed...")
    should_tune, tuned_thresholds, message = auto_tuner.schedule_periodic_retuning(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive',
        force=True  # Force for demo purposes
    )
    
    if should_tune:
        print(f"  ✓ Re-tuning performed: {message}")
        print()
        
        print("  Optimized thresholds:")
        for detector_name, thresholds in tuned_thresholds.items():
            print(f"    {detector_name}:")
            for param, value in thresholds.items():
                # Show change from original
                original_value = current_thresholds.get(detector_name, {}).get(param)
                if original_value is not None:
                    change = ((value - original_value) / original_value) * 100
                    print(f"      - {param}: {value:.3f} (change: {change:+.1f}%)")
                else:
                    print(f"      - {param}: {value:.3f}")
        print()
        
        # Apply tuned thresholds
        config['thresholds'] = tuned_thresholds
        print("  ✓ Tuned thresholds applied to configuration")
        print()
        
    else:
        print(f"  ℹ Re-tuning skipped: {message}")
        print()
    
    # Step 4: Initialize DetectorManager with tuned thresholds
    print("Step 4: Initializing DetectorManager with tuned thresholds...")
    detector_manager = DetectorManager(config)
    
    profile_info = detector_manager.get_profile_info()
    print(f"  ✓ DetectorManager initialized")
    print(f"  ✓ Active profile: {profile_info['profile_name']}")
    print(f"  ✓ Profile completeness: {profile_info['validation']['completeness_percentage']:.1f}%")
    print()
    
    # Step 5: Run detectors (optional - can be slow)
    run_detection = input("Run anomaly detection? (y/n): ").lower() == 'y'
    
    if run_detection:
        print()
        print("Step 5: Running anomaly detection with tuned thresholds...")
        results = detector_manager.run_all_detectors(df)
        
        # Get statistics
        stats = detector_manager.get_detector_statistics()
        
        print()
        print("  Detection results:")
        total_anomalies = 0
        for detector_name, stat in stats.items():
            if stat.success:
                print(f"    ✓ {detector_name}: {stat.anomalies_detected} anomalies "
                      f"({stat.execution_time_seconds:.2f}s)")
                total_anomalies += stat.anomalies_detected
            else:
                print(f"    ✗ {detector_name}: Failed - {stat.error_message}")
        
        print()
        print(f"  Total anomalies detected: {total_anomalies}")
        print()
    
    # Step 6: Generate tuning report
    print("Step 6: Generating tuning report...")
    report = auto_tuner.generate_tuning_report()
    
    # Save report
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / 'auto_tuning_demo_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  ✓ Tuning report saved to: {report_path}")
    print()
    
    # Display summary from report
    print("  Report summary:")
    lines = report.split('\n')
    for line in lines[0:15]:  # Show first 15 lines
        if line.strip():
            print(f"    {line}")
    print("    ...")
    print()
    
    # Step 7: Show tuning history
    print("Step 7: Tuning history summary...")
    history_summary = auto_tuner.get_tuning_history_summary()
    
    print(f"  Total tunings performed: {history_summary['total_tunings']}")
    if history_summary['last_tuning_date']:
        print(f"  Last tuning: {history_summary['last_tuning_date']}")
        print(f"  Days since last tuning: {history_summary['days_since_last_tuning']}")
        print(f"  Next scheduled tuning: {history_summary['next_scheduled_tuning']}")
        print(f"  Days until next tuning: {history_summary['days_until_next_tuning']}")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        demo_auto_tuning_workflow()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
