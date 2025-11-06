"""
Demonstration of False Positive Rate (FPR) Calculation

This script demonstrates the FPR calculation functionality in the AutoTuner module.
It shows how to:
1. Calculate FPR from historical detection results
2. Perform threshold sweep analysis
3. Identify optimal thresholds
4. Generate comprehensive FPR analysis reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auto_tuner import AutoTuner


def create_sample_data():
    """Create sample municipal data for demonstration."""
    np.random.seed(42)
    n_municipalities = 200
    
    return pd.DataFrame({
        'territory_id': range(1, n_municipalities + 1),
        'region_name': [f'Region_{i % 15}' for i in range(n_municipalities)],
        'municipal_name': [f'Municipality_{i}' for i in range(n_municipalities)],
        'population': np.random.lognormal(10, 1, n_municipalities),
        'consumption_total': np.random.lognormal(8, 0.8, n_municipalities),
        'salary_average': np.random.normal(50000, 15000, n_municipalities),
        'market_access_score': np.random.uniform(0, 100, n_municipalities)
    })


def create_sample_historical_anomalies():
    """Create sample historical anomaly results."""
    np.random.seed(42)
    
    detectors = ['statistical', 'geographic', 'cross_source', 'logical', 'temporal']
    anomalies = []
    
    # Simulate different FPR levels for different detectors
    detector_configs = {
        'statistical': {'count': 30, 'severity_range': (60, 95)},
        'geographic': {'count': 150, 'severity_range': (40, 80)},  # High FPR
        'cross_source': {'count': 80, 'severity_range': (50, 90)},
        'logical': {'count': 45, 'severity_range': (70, 100)},
        'temporal': {'count': 20, 'severity_range': (55, 85)}
    }
    
    for detector, config in detector_configs.items():
        for i in range(config['count']):
            anomalies.append({
                'detector_name': detector,
                'territory_id': np.random.randint(1, 201),
                'severity_score': np.random.uniform(*config['severity_range']),
                'indicator': f'indicator_{np.random.randint(1, 5)}',
                'anomaly_type': detector
            })
    
    return pd.DataFrame(anomalies)


def main():
    """Run FPR calculation demonstration."""
    print("=" * 80)
    print("False Positive Rate (FPR) Calculation Demonstration")
    print("=" * 80)
    print()
    
    # Initialize AutoTuner
    config = {
        'auto_tuning': {
            'target_false_positive_rate': 0.05,
            'min_anomalies_per_detector': 10,
            'max_anomalies_per_detector': 1000,
            'optimization_strategy': 'adaptive'
        },
        'export': {
            'output_dir': 'output'
        }
    }
    
    tuner = AutoTuner(config)
    print(f"✓ AutoTuner initialized with target FPR: {tuner.target_fpr:.3f}")
    print()
    
    # Create sample data
    df = create_sample_data()
    print(f"✓ Created sample data: {len(df)} municipalities")
    print()
    
    # Create sample historical anomalies
    historical_anomalies = create_sample_historical_anomalies()
    print(f"✓ Created historical anomalies: {len(historical_anomalies)} records")
    print()
    
    # Calculate FPR from historical results
    print("-" * 80)
    print("1. FPR Calculation from Historical Results")
    print("-" * 80)
    
    fpr_by_detector = tuner.calculate_fpr_from_historical_results(
        historical_anomalies, len(df)
    )
    
    print("\nEstimated FPR by Detector:")
    for detector, fpr in sorted(fpr_by_detector.items(), key=lambda x: x[1], reverse=True):
        status = "⚠️ HIGH" if fpr > tuner.target_fpr * 2 else "✓ OK"
        print(f"  {detector:20s}: {fpr:.4f} {status}")
    print()
    
    # Perform threshold sweep
    print("-" * 80)
    print("2. Threshold Sweep Analysis (Statistical Detector)")
    print("-" * 80)
    
    threshold_range = np.linspace(2.0, 4.5, 26)
    thresholds, fpr_values = tuner.calculate_fpr_by_threshold_sweep(
        df, 'statistical', 'z_score', threshold_range
    )
    
    print("\nThreshold Sweep Results (sample):")
    print(f"{'Threshold':>12s} {'FPR':>12s} {'Status':>12s}")
    print("-" * 40)
    for i in range(0, len(thresholds), 5):
        threshold = thresholds[i]
        fpr = fpr_values[i]
        status = "Target" if abs(fpr - tuner.target_fpr) < 0.01 else ""
        print(f"{threshold:12.2f} {fpr:12.4f} {status:>12s}")
    print()
    
    # Identify optimal threshold
    print("-" * 80)
    print("3. Optimal Threshold Identification")
    print("-" * 80)
    
    optimal_threshold = tuner.identify_optimal_threshold(
        threshold_range, fpr_values, tuner.target_fpr
    )
    
    optimal_idx = np.argmin(np.abs(threshold_range - optimal_threshold))
    achieved_fpr = fpr_values[optimal_idx]
    
    print(f"\n✓ Optimal z-score threshold: {optimal_threshold:.3f}")
    print(f"  Target FPR: {tuner.target_fpr:.4f}")
    print(f"  Achieved FPR: {achieved_fpr:.4f}")
    print(f"  Difference: {abs(achieved_fpr - tuner.target_fpr):.4f}")
    print()
    
    # Comprehensive FPR analysis
    print("-" * 80)
    print("4. Comprehensive Historical FPR Analysis")
    print("-" * 80)
    
    # Save historical results to temporary file
    temp_file = Path('output') / 'temp_historical_anomalies.csv'
    temp_file.parent.mkdir(exist_ok=True)
    historical_anomalies.to_csv(temp_file, index=False, encoding='utf-8')
    
    analysis = tuner.analyze_historical_fpr(df, str(temp_file))
    
    print("\nDetailed Analysis by Detector:")
    print()
    
    for detector, info in sorted(analysis.items(), key=lambda x: x[1]['estimated_fpr'], reverse=True):
        print(f"📊 {detector.upper()}")
        print(f"   Estimated FPR: {info['estimated_fpr']:.4f}")
        print(f"   Detection Rate: {info['detection_rate']:.2%}")
        print(f"   Flagged Municipalities: {info['flagged_municipalities']}/{len(df)}")
        print(f"   Total Anomalies: {info['total_anomalies']}")
        print(f"   Average Severity: {info['avg_severity']:.1f}")
        print(f"   Severity Distribution:")
        print(f"     Critical (≥90): {info['severity_distribution']['critical']}")
        print(f"     High (70-90): {info['severity_distribution']['high']}")
        print(f"     Medium (50-70): {info['severity_distribution']['medium']}")
        print(f"     Low (<50): {info['severity_distribution']['low']}")
        print(f"   Recommendation: {info['recommendation']}")
        print(f"   Meets Target: {'✓ Yes' if info['meets_target'] else '✗ No'}")
        print()
    
    # Clean up
    temp_file.unlink()
    
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
