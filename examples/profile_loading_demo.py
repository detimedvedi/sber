"""
Demonstration of Profile Loading and Validation

This script demonstrates how to:
1. Load predefined profiles (strict, normal, relaxed)
2. Validate profile completeness
3. Merge custom profiles with defaults
4. Switch profiles at runtime
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


def demonstrate_profile_loading():
    """Demonstrate profile loading functionality."""
    print("=" * 80)
    print("PROFILE LOADING AND VALIDATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    config = load_config()
    
    # 1. Load predefined profiles
    print("1. LOADING PREDEFINED PROFILES")
    print("-" * 80)
    
    for profile_name in ['strict', 'normal', 'relaxed']:
        config['detection_profile'] = profile_name
        manager = ThresholdManager(config)
        
        info = manager.get_profile_info()
        print(f"\nProfile: {profile_name}")
        print(f"  Valid: {info['validation']['is_valid']}")
        print(f"  Completeness: {info['validation']['completeness_percentage']:.1f}%")
        print(f"  Statistical z_score: {info['thresholds']['statistical']['z_score']}")
        print(f"  Geographic regional_z_score: {info['thresholds']['geographic']['regional_z_score']}")
    
    # 2. Validate profile completeness
    print("\n\n2. PROFILE VALIDATION")
    print("-" * 80)
    
    config['detection_profile'] = 'normal'
    manager = ThresholdManager(config)
    info = manager.get_profile_info()
    
    print(f"\nProfile: {info['profile_name']}")
    print(f"Validation Results:")
    print(f"  Is Valid: {info['validation']['is_valid']}")
    print(f"  Completeness: {info['validation']['completeness_percentage']:.1f}%")
    print(f"  Missing Parameters: {info['validation']['missing_params']}")
    print(f"  Complete Parameters: {len(info['validation']['complete_params'])}")
    
    # 3. Merge custom profile with defaults
    print("\n\n3. CUSTOM PROFILE WITH DEFAULT MERGING")
    print("-" * 80)
    
    # Create incomplete custom profile
    custom_profile = {
        'statistical': {
            'z_score': 2.7,  # Custom value
            'iqr_multiplier': 1.8  # Custom value
            # Missing: percentile_lower, percentile_upper
        },
        'geographic': {
            'regional_z_score': 2.2  # Custom value
            # Missing: cluster_threshold
        }
        # Missing: temporal, cross_source, logical
    }
    
    print("\nCustom profile (incomplete):")
    print(f"  statistical.z_score: 2.7")
    print(f"  statistical.iqr_multiplier: 1.8")
    print(f"  geographic.regional_z_score: 2.2")
    print(f"  (other parameters missing)")
    
    manager.load_custom_profile(custom_profile, 'my_custom')
    info = manager.get_profile_info()
    
    print(f"\nAfter merging with defaults:")
    print(f"  Profile: {info['profile_name']}")
    print(f"  Completeness: {info['validation']['completeness_percentage']:.1f}%")
    
    stat_thresholds = manager.get_thresholds('statistical')
    print(f"\nStatistical thresholds:")
    print(f"  z_score: {stat_thresholds['z_score']} (custom)")
    print(f"  iqr_multiplier: {stat_thresholds['iqr_multiplier']} (custom)")
    print(f"  percentile_lower: {stat_thresholds['percentile_lower']} (default)")
    print(f"  percentile_upper: {stat_thresholds['percentile_upper']} (default)")
    
    geo_thresholds = manager.get_thresholds('geographic')
    print(f"\nGeographic thresholds:")
    print(f"  regional_z_score: {geo_thresholds['regional_z_score']} (custom)")
    print(f"  cluster_threshold: {geo_thresholds['cluster_threshold']} (default)")
    
    temp_thresholds = manager.get_thresholds('temporal')
    print(f"\nTemporal thresholds (all defaults):")
    print(f"  spike_threshold: {temp_thresholds['spike_threshold']}")
    print(f"  drop_threshold: {temp_thresholds['drop_threshold']}")
    print(f"  volatility_multiplier: {temp_thresholds['volatility_multiplier']}")
    
    # 4. Runtime profile switching
    print("\n\n4. RUNTIME PROFILE SWITCHING")
    print("-" * 80)
    
    config['detection_profile'] = 'normal'
    manager = ThresholdManager(config)
    
    print("\nInitial profile: normal")
    thresholds = manager.get_thresholds('statistical')
    print(f"  z_score: {thresholds['z_score']}")
    
    print("\nSwitching to strict...")
    manager.load_profile('strict')
    thresholds = manager.get_thresholds('statistical')
    print(f"  z_score: {thresholds['z_score']}")
    
    print("\nSwitching to relaxed...")
    manager.load_profile('relaxed')
    thresholds = manager.get_thresholds('statistical')
    print(f"  z_score: {thresholds['z_score']}")
    
    # 5. Error handling for invalid profiles
    print("\n\n5. ERROR HANDLING")
    print("-" * 80)
    
    print("\nAttempting to load non-existent profile...")
    try:
        manager.load_profile('nonexistent')
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    # 6. Profile comparison
    print("\n\n6. PROFILE COMPARISON")
    print("-" * 80)
    
    comparison_table = []
    detector_types = ['statistical', 'temporal', 'geographic', 'cross_source']
    
    for detector_type in detector_types:
        for profile_name in ['strict', 'normal', 'relaxed']:
            config['detection_profile'] = profile_name
            manager = ThresholdManager(config)
            thresholds = manager.get_thresholds(detector_type)
            
            if detector_type == 'statistical':
                comparison_table.append({
                    'profile': profile_name,
                    'detector': detector_type,
                    'metric': 'z_score',
                    'value': thresholds['z_score']
                })
            elif detector_type == 'temporal':
                comparison_table.append({
                    'profile': profile_name,
                    'detector': detector_type,
                    'metric': 'spike_threshold',
                    'value': thresholds['spike_threshold']
                })
            elif detector_type == 'geographic':
                comparison_table.append({
                    'profile': profile_name,
                    'detector': detector_type,
                    'metric': 'regional_z_score',
                    'value': thresholds['regional_z_score']
                })
            elif detector_type == 'cross_source':
                comparison_table.append({
                    'profile': profile_name,
                    'detector': detector_type,
                    'metric': 'discrepancy_threshold',
                    'value': thresholds['discrepancy_threshold']
                })
    
    print(f"\n{'Detector':<20} {'Metric':<25} {'Strict':<10} {'Normal':<10} {'Relaxed':<10}")
    print("-" * 80)
    
    for i in range(0, len(comparison_table), 3):
        row = comparison_table[i:i+3]
        detector = row[0]['detector']
        metric = row[0]['metric']
        values = [str(r['value']) for r in row]
        print(f"{detector:<20} {metric:<25} {values[0]:<10} {values[1]:<10} {values[2]:<10}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Features Demonstrated:

1. Profile Loading:
   - Predefined profiles (strict, normal, relaxed) loaded from config.yaml
   - Profile selection via 'detection_profile' setting
   - Runtime profile switching with load_profile()

2. Profile Validation:
   - Automatic validation of profile completeness
   - Reports missing parameters
   - Calculates completeness percentage

3. Default Merging:
   - Incomplete profiles automatically merged with defaults
   - Custom values take precedence
   - Missing parameters filled from default thresholds
   - Ensures all required parameters are present

4. Custom Profiles:
   - Load custom profiles with load_custom_profile()
   - Partial profiles supported (merged with defaults)
   - Named custom profiles for tracking

5. Error Handling:
   - Invalid profile names raise ValueError
   - Clear error messages with available profiles listed
   - Graceful fallback to defaults when needed

Usage Recommendations:

- Use 'strict' for comprehensive data quality audits
- Use 'normal' for regular operations (balanced)
- Use 'relaxed' for high-confidence alerts only
- Create custom profiles for specific use cases
- Validate custom profiles before deployment
    """)
    print("=" * 80)


if __name__ == '__main__':
    demonstrate_profile_loading()
