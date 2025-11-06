"""
Demonstration of DetectorManager Profile Integration

This script demonstrates how DetectorManager integrates with threshold profiles:
1. Loading profiles on initialization
2. Applying profile thresholds to detectors
3. Runtime profile switching
4. Comparing detection results across profiles
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import numpy as np
from src.detector_manager import DetectorManager


def load_config():
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_sample_data():
    """Create sample municipal data for demonstration."""
    np.random.seed(42)
    
    # Create base data
    n_municipalities = 100
    
    data = {
        'territory_id': range(1, n_municipalities + 1),
        'municipal_name': [f'Municipality_{i}' for i in range(1, n_municipalities + 1)],
        'region_name': ['Region_A'] * 50 + ['Region_B'] * 50,
        'consumption_total': np.random.normal(1000, 200, n_municipalities),
        'population_total': np.random.normal(50000, 10000, n_municipalities),
        'salary_average': np.random.normal(40000, 8000, n_municipalities)
    }
    
    df = pd.DataFrame(data)
    
    # Add some outliers
    df.loc[5, 'consumption_total'] = 3000  # Strong outlier
    df.loc[15, 'population_total'] = 150000  # Strong outlier
    df.loc[25, 'salary_average'] = 80000  # Moderate outlier
    
    return df


def demonstrate_profile_integration():
    """Demonstrate profile integration with DetectorManager."""
    print("=" * 80)
    print("DETECTOR MANAGER PROFILE INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    config = load_config()
    sample_data = create_sample_data()
    
    print(f"Sample data: {len(sample_data)} municipalities")
    print(f"Indicators: {[col for col in sample_data.columns if col not in ['territory_id', 'municipal_name', 'region_name']]}")
    print()
    
    # 1. Initialize with different profiles
    print("1. INITIALIZATION WITH DIFFERENT PROFILES")
    print("-" * 80)
    
    for profile_name in ['strict', 'normal', 'relaxed']:
        config['detection_profile'] = profile_name
        manager = DetectorManager(config)
        
        print(f"\nProfile: {profile_name}")
        print(f"  Current profile: {manager.get_current_profile()}")
        print(f"  Detectors initialized: {len(manager.detectors)}")
        
        # Show some threshold values
        info = manager.get_profile_info()
        stat_thresholds = info['thresholds']['statistical']
        geo_thresholds = info['thresholds']['geographic']
        
        print(f"  Statistical z_score: {stat_thresholds['z_score']}")
        print(f"  Geographic regional_z_score: {geo_thresholds['regional_z_score']}")
    
    # 2. Verify thresholds are applied to detectors
    print("\n\n2. THRESHOLD APPLICATION TO DETECTORS")
    print("-" * 80)
    
    config['detection_profile'] = 'strict'
    manager = DetectorManager(config)
    
    print(f"\nProfile: {manager.get_current_profile()}")
    print("\nDetector thresholds:")
    
    for detector_name, detector in manager.detectors.items():
        print(f"\n  {detector_name}:")
        
        if detector_name == 'statistical':
            print(f"    z_score_threshold: {detector.z_score_threshold}")
            print(f"    iqr_multiplier: {detector.iqr_multiplier}")
        elif detector_name == 'geographic':
            print(f"    regional_z_score_threshold: {detector.regional_z_score_threshold}")
            print(f"    cluster_threshold: {detector.cluster_threshold}")
        elif detector_name == 'temporal':
            print(f"    spike_threshold: {detector.spike_threshold}")
            print(f"    drop_threshold: {detector.drop_threshold}")
        elif detector_name == 'cross_source':
            print(f"    correlation_threshold: {detector.correlation_threshold}")
            print(f"    discrepancy_threshold: {detector.discrepancy_threshold}")
    
    # 3. Runtime profile switching
    print("\n\n3. RUNTIME PROFILE SWITCHING")
    print("-" * 80)
    
    config['detection_profile'] = 'normal'
    manager = DetectorManager(config)
    
    print(f"\nInitial profile: {manager.get_current_profile()}")
    
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        print(f"  Statistical z_score: {detector.z_score_threshold}")
    
    print("\nSwitching to 'strict'...")
    manager.switch_profile('strict')
    
    print(f"  New profile: {manager.get_current_profile()}")
    
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        print(f"  Statistical z_score: {detector.z_score_threshold}")
    
    print("\nSwitching to 'relaxed'...")
    manager.switch_profile('relaxed')
    
    print(f"  New profile: {manager.get_current_profile()}")
    
    if 'statistical' in manager.detectors:
        detector = manager.detectors['statistical']
        print(f"  Statistical z_score: {detector.z_score_threshold}")
    
    # 4. Compare detection results across profiles
    print("\n\n4. DETECTION RESULTS COMPARISON")
    print("-" * 80)
    
    results_by_profile = {}
    
    for profile_name in ['strict', 'normal', 'relaxed']:
        config['detection_profile'] = profile_name
        manager = DetectorManager(config)
        
        print(f"\nRunning detection with '{profile_name}' profile...")
        results = manager.run_all_detectors(sample_data)
        
        # Count anomalies
        total_anomalies = sum(len(df) for df in results if df is not None and len(df) > 0)
        
        # Count by detector
        anomalies_by_detector = {}
        for i, result_df in enumerate(results):
            if result_df is not None and len(result_df) > 0:
                detector_name = result_df['detection_method'].iloc[0] if 'detection_method' in result_df.columns else f'detector_{i}'
                anomalies_by_detector[detector_name] = len(result_df)
        
        results_by_profile[profile_name] = {
            'total': total_anomalies,
            'by_detector': anomalies_by_detector
        }
        
        print(f"  Total anomalies: {total_anomalies}")
        for detector, count in anomalies_by_detector.items():
            print(f"    {detector}: {count}")
    
    # 5. Summary comparison
    print("\n\n5. PROFILE COMPARISON SUMMARY")
    print("-" * 80)
    
    print(f"\n{'Profile':<15} {'Total Anomalies':<20} {'Change from Normal':<20}")
    print("-" * 80)
    
    normal_total = results_by_profile['normal']['total']
    
    for profile_name in ['strict', 'normal', 'relaxed']:
        total = results_by_profile[profile_name]['total']
        
        if profile_name == 'normal':
            change = "baseline"
        else:
            if normal_total > 0:
                change_pct = ((total - normal_total) / normal_total) * 100
                change = f"{change_pct:+.1f}%"
            else:
                change = "N/A"
        
        print(f"{profile_name:<15} {total:<20} {change:<20}")
    
    # 6. Profile info
    print("\n\n6. DETAILED PROFILE INFORMATION")
    print("-" * 80)
    
    config['detection_profile'] = 'normal'
    manager = DetectorManager(config)
    
    info = manager.get_profile_info()
    
    print(f"\nProfile: {info['profile_name']}")
    print(f"Valid: {info['validation']['is_valid']}")
    print(f"Completeness: {info['validation']['completeness_percentage']:.1f}%")
    
    print("\nThresholds by detector:")
    for detector_type, thresholds in info['thresholds'].items():
        print(f"\n  {detector_type}:")
        for param, value in thresholds.items():
            print(f"    {param}: {value}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Features Demonstrated:

1. Profile Loading on Initialization:
   - DetectorManager loads the profile specified in config['detection_profile']
   - Profile thresholds are automatically applied to config['thresholds']
   - All detectors are initialized with profile-specific thresholds

2. Threshold Application:
   - Profile thresholds are merged with defaults to ensure completeness
   - Each detector receives the correct thresholds for its type
   - Thresholds are accessible via detector attributes (e.g., z_score_threshold)

3. Runtime Profile Switching:
   - Use switch_profile() to change profiles without restarting
   - Detectors are automatically reinitialized with new thresholds
   - Previous detection results are preserved

4. Detection Behavior:
   - Strict profile: Lower thresholds → More anomalies detected
   - Normal profile: Balanced thresholds → Moderate anomaly count
   - Relaxed profile: Higher thresholds → Fewer anomalies detected

5. Profile Management:
   - get_current_profile() returns active profile name
   - get_profile_info() provides detailed profile information
   - Profile validation ensures all required parameters are present

Usage Recommendations:

- Use 'strict' for comprehensive data quality audits
- Use 'normal' for regular operations (balanced sensitivity)
- Use 'relaxed' for high-confidence alerts only
- Switch profiles at runtime to compare detection sensitivity
- Monitor anomaly counts to tune profile selection
    """)
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_profile_integration()
