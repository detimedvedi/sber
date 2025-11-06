"""
Demo: Exporting Tuned Configuration

This script demonstrates how to export tuned thresholds and generate
comprehensive tuning reports using the AutoTuner.

The export functionality creates:
1. YAML configuration file with optimized thresholds
2. Markdown tuning report with detailed analysis
3. JSON statistics file for programmatic access
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from src.auto_tuner import AutoTuner
from src.data_loader import DataLoader


def load_sample_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def demo_export_tuned_configuration():
    """
    Demonstrate exporting tuned configuration with comprehensive reports.
    """
    print("=" * 80)
    print("Demo: Exporting Tuned Configuration")
    print("=" * 80)
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_sample_config()
    
    # Initialize AutoTuner
    print("Initializing AutoTuner...")
    tuner = AutoTuner(config)
    print(f"  Target FPR: {tuner.target_fpr}")
    print(f"  Re-tuning interval: {tuner.retuning_interval_days} days")
    print()
    
    # Load data
    print("Loading data...")
    loader = DataLoader(config)
    df = loader.load_all_data()
    print(f"  Loaded {len(df)} municipalities")
    print()
    
    # Get current thresholds
    current_thresholds = config.get('thresholds', {})
    print("Current thresholds:")
    for detector, thresholds in current_thresholds.items():
        print(f"  {detector}:")
        for param, value in thresholds.items():
            print(f"    {param}: {value}")
    print()
    
    # Optimize thresholds
    print("Optimizing thresholds with 'adaptive' strategy...")
    optimized_thresholds = tuner.optimize_thresholds(
        df=df,
        current_thresholds=current_thresholds,
        strategy='adaptive'
    )
    print("  Optimization complete!")
    print()
    
    # Export individual files
    print("-" * 80)
    print("Method 1: Export Individual Files")
    print("-" * 80)
    print()
    
    # Export threshold configuration
    print("1. Exporting threshold configuration...")
    config_file = tuner.export_tuned_thresholds(optimized_thresholds)
    print(f"   ✓ Configuration exported to: {config_file}")
    print()
    
    # Generate and export tuning report
    print("2. Generating tuning report...")
    report = tuner.generate_tuning_report(
        include_rationale=True,
        include_statistics=True
    )
    
    output_dir = Path(config.get('export', {}).get('output_dir', 'output'))
    report_file = output_dir / 'tuning_report_demo.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   ✓ Report exported to: {report_file}")
    print()
    
    # Display report preview
    print("Report Preview (first 20 lines):")
    print("-" * 80)
    report_lines = report.split('\n')
    for line in report_lines[:20]:
        print(line)
    if len(report_lines) > 20:
        print(f"... ({len(report_lines) - 20} more lines)")
    print("-" * 80)
    print()
    
    # Export complete package
    print("-" * 80)
    print("Method 2: Export Complete Tuning Package")
    print("-" * 80)
    print()
    
    print("Exporting complete tuning package...")
    exported_files = tuner.export_tuning_package(optimized_thresholds)
    
    print("✓ Tuning package exported successfully!")
    print()
    print("Exported files:")
    for file_type, file_path in exported_files.items():
        print(f"  - {file_type}: {file_path}")
    print()
    
    # Display tuning history summary
    print("-" * 80)
    print("Tuning History Summary")
    print("-" * 80)
    print()
    
    history_summary = tuner.get_tuning_history_summary()
    
    print(f"Total tunings performed: {history_summary['total_tunings']}")
    print(f"Last tuning: {history_summary['last_tuning_date']}")
    print(f"Days since last tuning: {history_summary['days_since_last_tuning']}")
    print(f"Next scheduled tuning: {history_summary['next_scheduled_tuning']}")
    print(f"Days until next tuning: {history_summary['days_until_next_tuning']}")
    print()
    
    if history_summary['tuning_history']:
        print("Recent tuning history:")
        for entry in history_summary['tuning_history']:
            print(f"  - {entry['timestamp']}")
            print(f"    Anomalies: {entry['total_anomalies_before']} → {entry['total_anomalies_after']} "
                  f"({entry['anomaly_reduction_pct']:.1f}% reduction)")
            print(f"    FPR: {entry['avg_fpr_before']:.4f} → {entry['avg_fpr_after']:.4f} "
                  f"({entry['fpr_reduction_pct']:.1f}% reduction)")
            print(f"    Detectors: {', '.join(entry['detectors_tuned'])}")
            print()
    
    # Show how to apply the tuned configuration
    print("-" * 80)
    print("How to Apply Tuned Configuration")
    print("-" * 80)
    print()
    print("To apply the optimized thresholds:")
    print()
    print("1. Review the tuning report to understand the changes")
    print(f"   Report: {report_file}")
    print()
    print("2. Review the exported configuration file")
    print(f"   Config: {config_file}")
    print()
    print("3. Update your config.yaml with the optimized thresholds:")
    print("   - Copy the 'thresholds' section from the exported file")
    print("   - Paste into your config.yaml")
    print()
    print("4. Run detection with the new thresholds:")
    print("   python main.py")
    print()
    print("5. Validate results and adjust if needed")
    print()
    
    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)


def demo_report_customization():
    """
    Demonstrate different report customization options.
    """
    print()
    print("=" * 80)
    print("Demo: Report Customization Options")
    print("=" * 80)
    print()
    
    # Load configuration
    config = load_sample_config()
    tuner = AutoTuner(config)
    
    # Check if we have tuning history
    if not tuner.tuning_history:
        print("No tuning history available. Run the first demo to generate tuning data.")
        return
    
    output_dir = Path(config.get('export', {}).get('output_dir', 'output'))
    
    # Generate different report variants
    print("Generating different report variants...")
    print()
    
    # 1. Minimal report (no rationale, no statistics)
    print("1. Minimal Report (summary only)")
    minimal_report = tuner.generate_tuning_report(
        include_rationale=False,
        include_statistics=False
    )
    minimal_file = output_dir / 'tuning_report_minimal.md'
    with open(minimal_file, 'w', encoding='utf-8') as f:
        f.write(minimal_report)
    print(f"   ✓ Saved to: {minimal_file}")
    print(f"   Length: {len(minimal_report.split(chr(10)))} lines")
    print()
    
    # 2. Statistics only (no rationale)
    print("2. Statistics Report (with metrics, no rationale)")
    stats_report = tuner.generate_tuning_report(
        include_rationale=False,
        include_statistics=True
    )
    stats_file = output_dir / 'tuning_report_statistics.md'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_report)
    print(f"   ✓ Saved to: {stats_file}")
    print(f"   Length: {len(stats_report.split(chr(10)))} lines")
    print()
    
    # 3. Full report (with rationale and statistics)
    print("3. Full Report (with rationale and statistics)")
    full_report = tuner.generate_tuning_report(
        include_rationale=True,
        include_statistics=True
    )
    full_file = output_dir / 'tuning_report_full.md'
    with open(full_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"   ✓ Saved to: {full_file}")
    print(f"   Length: {len(full_report.split(chr(10)))} lines")
    print()
    
    print("All report variants generated successfully!")
    print()


if __name__ == '__main__':
    # Run main demo
    demo_export_tuned_configuration()
    
    # Run customization demo
    demo_report_customization()
