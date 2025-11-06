"""
Demonstration of Threshold Profile Usage

This script demonstrates how to use the three predefined threshold profiles
(strict, normal, relaxed) in the anomaly detection system.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.detector_manager import ThresholdManager


def load_config():
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def demonstrate_profiles():
    """Demonstrate threshold profile functionality."""
    print("=" * 80)
    print("THRESHOLD PROFILE DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_config()
    
    # Show available profiles
    profiles = config.get('threshold_profiles', {})
    print(f"Available profiles: {', '.join(profiles.keys())}")
    print()
    
    # Demonstrate each profile
    for profile_name in ['strict', 'normal', 'relaxed']:
        print(f"\n{'=' * 80}")
        print(f"PROFILE: {profile_name.upper()}")
        print(f"{'=' * 80}")
        
        # Create threshold manager with this profile
        config['detection_profile'] = profile_name
        manager = ThresholdManager(config)
        
        # Show thresholds for each detector type
        detector_types = ['statistical', 'temporal', 'geographic', 'cross_source']
        
        for detector_type in detector_types:
            thresholds = manager.get_thresholds(detector_type)
            print(f"\n{detector_type.upper()} Thresholds:")
            for key, value in thresholds.items():
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("PROFILE COMPARISON")
    print("=" * 80)
    
    # Compare key thresholds across profiles
    comparison_metrics = [
        ('statistical', 'z_score'),
        ('statistical', 'iqr_multiplier'),
        ('temporal', 'spike_threshold'),
        ('geographic', 'regional_z_score'),
        ('cross_source', 'discrepancy_threshold')
    ]
    
    print(f"\n{'Metric':<40} {'Strict':<12} {'Normal':<12} {'Relaxed':<12}")
    print("-" * 80)
    
    for detector_type, metric in comparison_metrics:
        values = []
        for profile_name in ['strict', 'normal', 'relaxed']:
            config['detection_profile'] = profile_name
            manager = ThresholdManager(config)
            thresholds = manager.get_thresholds(detector_type)
            values.append(str(thresholds.get(metric, 'N/A')))
        
        metric_name = f"{detector_type}.{metric}"
        print(f"{metric_name:<40} {values[0]:<12} {values[1]:<12} {values[2]:<12}")
    
    print("\n" + "=" * 80)
    print("RUNTIME PROFILE SWITCHING")
    print("=" * 80)
    
    # Demonstrate runtime profile switching
    config['detection_profile'] = 'normal'
    manager = ThresholdManager(config)
    
    print("\nInitial profile: normal")
    print(f"Statistical z_score: {manager.get_thresholds('statistical')['z_score']}")
    
    print("\nSwitching to strict profile...")
    manager.load_profile('strict')
    print(f"Statistical z_score: {manager.get_thresholds('statistical')['z_score']}")
    
    print("\nSwitching to relaxed profile...")
    manager.load_profile('relaxed')
    print(f"Statistical z_score: {manager.get_thresholds('statistical')['z_score']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
STRICT PROFILE:
  - Use for: Initial data quality assessment, comprehensive audits
  - Detects: 50-100% more anomalies than normal
  - Trade-off: Higher false positive rate
  - Best for: Finding all potential issues

NORMAL PROFILE:
  - Use for: Regular operations, balanced detection
  - Detects: Baseline anomaly count
  - Trade-off: Balanced precision/recall
  - Best for: Day-to-day monitoring

RELAXED PROFILE:
  - Use for: Focus on critical issues, reduce alert fatigue
  - Detects: 30-50% fewer anomalies than normal
  - Trade-off: May miss subtle anomalies
  - Best for: High-confidence critical alerts only
    """)
    
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_profiles()
