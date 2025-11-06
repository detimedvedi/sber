# Configuration Profiles Quick Reference

**Quick guide for using detection profiles**

---

## What are Configuration Profiles?

Pre-configured threshold sets for different analysis scenarios. Choose a profile instead of manually adjusting dozens of thresholds.

**Available Profiles**:
- **Strict**: Maximum sensitivity, catch all potential anomalies
- **Normal**: Balanced detection (default)
- **Relaxed**: High confidence, minimize false positives

---

## Quick Start

### Select a Profile

Edit `config.yaml`:

```yaml
detection_profile: "normal"  # Options: strict, normal, relaxed
```

### Run Analysis

```bash
python main.py
```

That's it! The system automatically applies the selected profile's thresholds.

---

## Profile Comparison

| Aspect | Strict | Normal | Relaxed |
|--------|--------|--------|---------|
| **Sensitivity** | Highest | Balanced | Lowest |
| **False Positives** | ~10-15% | ~5% | ~1-2% |
| **Anomaly Count** | 2-3x normal | Baseline | 0.3-0.5x normal |
| **Use Case** | Exploration | Monitoring | Executive reports |

### Example Results (Same Dataset)

| Metric | Strict | Normal | Relaxed |
|--------|--------|--------|---------|
| Total Anomalies | 18,234 | 8,456 | 2,891 |
| Statistical | 3,421 | 1,234 | 342 |
| Geographic | 9,876 | 4,123 | 1,234 |
| Temporal | 1,234 | 567 | 123 |
| Cross-Source | 2,345 | 1,456 | 789 |
| Logical | 1,358 | 1,076 | 403 |
| Municipalities Affected | 92% | 78% | 45% |

---

## When to Use Each Profile

### Strict Profile

**Use When**:
- Initial data exploration
- Data quality audit
- Comprehensive analysis needed
- Research projects
- You have time to review many anomalies

**Avoid When**:
- Need quick results
- Limited review capacity
- Executive reporting
- Production alerts


### Normal Profile (Default)

**Use When**:
- Regular monitoring
- Routine analysis
- Balanced approach needed
- Most use cases
- Unsure which profile to use

**Characteristics**:
- Target 5% false positive rate
- Catches significant anomalies
- Manageable review workload
- Good starting point

### Relaxed Profile

**Use When**:
- Executive reporting
- High-stakes decisions
- Production alerts
- Limited review capacity
- Need high confidence

**Avoid When**:
- Might miss subtle issues
- Comprehensive coverage needed
- Data quality assessment
- Exploratory analysis

---

## Profile Details

### Strict Profile Thresholds

```yaml
threshold_profiles:
  strict:
    statistical:
      z_score: 2.5          # Lower = more sensitive
      iqr_multiplier: 1.2
      percentile_lower: 2
      percentile_upper: 98
    
    geographic:
      regional_z_score: 1.5
      cluster_threshold: 2.0
      capital_threshold: 2.5
      urban_threshold: 2.0
      rural_threshold: 1.5
    
    temporal:
      spike_threshold: 75
      drop_threshold: -40
      volatility_multiplier: 1.5
    
    cross_source:
      correlation_threshold: 0.6
      discrepancy_threshold: 40
```

### Normal Profile Thresholds

```yaml
threshold_profiles:
  normal:
    statistical:
      z_score: 3.0          # Balanced
      iqr_multiplier: 1.5
      percentile_lower: 1
      percentile_upper: 99
    
    geographic:
      regional_z_score: 2.5
      cluster_threshold: 2.5
      capital_threshold: 3.5
      urban_threshold: 2.5
      rural_threshold: 2.0
    
    temporal:
      spike_threshold: 100
      drop_threshold: -50
      volatility_multiplier: 2.0
    
    cross_source:
      correlation_threshold: 0.5
      discrepancy_threshold: 50
```


### Relaxed Profile Thresholds

```yaml
threshold_profiles:
  relaxed:
    statistical:
      z_score: 3.5          # Higher = less sensitive
      iqr_multiplier: 2.0
      percentile_lower: 0.5
      percentile_upper: 99.5
    
    geographic:
      regional_z_score: 3.0
      cluster_threshold: 3.0
      capital_threshold: 4.0
      urban_threshold: 3.0
      rural_threshold: 2.5
    
    temporal:
      spike_threshold: 150
      drop_threshold: -60
      volatility_multiplier: 2.5
    
    cross_source:
      correlation_threshold: 0.4
      discrepancy_threshold: 60
```

---

## Advanced Usage

### Custom Profile

Create your own profile by extending an existing one:

```yaml
threshold_profiles:
  custom:
    # Inherit from normal profile
    _base: "normal"
    
    # Override specific thresholds
    statistical:
      z_score: 3.2  # Custom value
    
    geographic:
      capital_threshold: 4.0  # More relaxed for capitals
      rural_threshold: 1.8    # Stricter for rural

# Use custom profile
detection_profile: "custom"
```

### Command-Line Override

Override profile without editing config:

```bash
# Use strict profile
python main.py --profile strict

# Use relaxed profile
python main.py --profile relaxed

# Use custom profile
python main.py --profile custom
```

### Profile Switching

Switch profiles programmatically:

```python
from src.detector_manager import DetectorManager

# Initialize with normal profile
manager = DetectorManager(config, profile='normal')

# Run detection
results_normal = manager.run_all_detectors(df)

# Switch to strict profile
manager.load_profile('strict')
results_strict = manager.run_all_detectors(df)

# Compare results
print(f"Normal: {len(results_normal)} anomalies")
print(f"Strict: {len(results_strict)} anomalies")
```


---

## Common Scenarios

### Scenario 1: Initial Data Assessment

**Goal**: Understand data quality and identify all potential issues

**Recommended Profile**: **Strict**

**Workflow**:
1. Set `detection_profile: "strict"`
2. Run analysis
3. Review all anomalies
4. Identify patterns and systematic issues
5. Switch to normal profile for ongoing monitoring

### Scenario 2: Regular Monitoring

**Goal**: Routine anomaly detection for ongoing operations

**Recommended Profile**: **Normal**

**Workflow**:
1. Set `detection_profile: "normal"`
2. Run analysis regularly (daily/weekly)
3. Review flagged anomalies
4. Investigate high-priority items
5. Track trends over time

### Scenario 3: Executive Reporting

**Goal**: Present only high-confidence anomalies to management

**Recommended Profile**: **Relaxed**

**Workflow**:
1. Set `detection_profile: "relaxed"`
2. Run analysis
3. Generate executive summary
4. Present top 10-20 critical anomalies
5. Provide clear, actionable recommendations

### Scenario 4: Compliance Audit

**Goal**: Identify clear violations for compliance reporting

**Recommended Profile**: **Relaxed** or **Custom**

**Configuration**:
```yaml
threshold_profiles:
  compliance:
    _base: "relaxed"
    
    # Very strict for logical inconsistencies
    logical:
      negative_value_tolerance: 0  # Zero tolerance
    
    # Relaxed for natural variation
    geographic:
      regional_z_score: 3.5
```


---

## Combining Profiles with Auto-Tuning

Profiles and auto-tuning work together:

### Option 1: Profile as Starting Point

Use profile as baseline, then auto-tune:

```yaml
detection_profile: "normal"  # Start with normal

auto_tuning:
  enabled: true
  target_false_positive_rate: 0.05
  # Auto-tuner optimizes from normal profile thresholds
```

### Option 2: Auto-Tune Each Profile

Generate optimized versions of each profile:

```bash
# Auto-tune strict profile
python main.py --profile strict --auto-tune

# Auto-tune normal profile
python main.py --profile normal --auto-tune

# Auto-tune relaxed profile
python main.py --profile relaxed --auto-tune
```

### Option 3: Profile-Specific Targets

Different FPR targets for different profiles:

```yaml
auto_tuning:
  enabled: true
  profile_targets:
    strict: 0.10   # 10% FPR acceptable
    normal: 0.05   # 5% FPR target
    relaxed: 0.02  # 2% FPR target
```

---

## Troubleshooting

### Problem: Too Many Anomalies

**Symptoms**: Thousands of anomalies, overwhelming to review

**Solution**: Switch to more relaxed profile

```yaml
# Change from:
detection_profile: "strict"

# To:
detection_profile: "normal"  # or "relaxed"
```

### Problem: Missing Known Issues

**Symptoms**: Known anomalies not being detected

**Solution**: Switch to stricter profile

```yaml
# Change from:
detection_profile: "relaxed"

# To:
detection_profile: "normal"  # or "strict"
```

### Problem: Inconsistent Results

**Symptoms**: Results vary significantly between runs

**Possible Causes**:
- Profile not specified (using default)
- Profile definitions changed
- Data changed significantly

**Solution**: Explicitly set profile and verify configuration

```yaml
# Explicitly set profile
detection_profile: "normal"

# Verify profile is loaded
# Check logs for: "Loaded detection profile: normal"
```


---

## Best Practices

### 1. Start with Normal Profile

Unless you have specific requirements, start with the normal profile:

```yaml
detection_profile: "normal"
```

It provides balanced detection suitable for most use cases.

### 2. Use Strict for Exploration

When first analyzing a new dataset:

```yaml
detection_profile: "strict"
```

This helps you understand the data and identify all potential issues.

### 3. Use Relaxed for Production

For production alerts and executive reports:

```yaml
detection_profile: "relaxed"
```

This minimizes false alarms and focuses on high-confidence anomalies.

### 4. Document Profile Choice

Always document why you chose a specific profile:

```yaml
# Using strict profile for initial data quality assessment
# Will switch to normal after baseline established
detection_profile: "strict"
```

### 5. Review Profile Periodically

Data characteristics change over time. Review your profile choice:
- Quarterly for stable data
- Monthly for dynamic data
- After major data source changes

---

## Profile Selection Decision Tree

```
Start
  │
  ├─ First time analyzing this data?
  │   └─ YES → Use STRICT profile
  │   └─ NO → Continue
  │
  ├─ For executive reporting?
  │   └─ YES → Use RELAXED profile
  │   └─ NO → Continue
  │
  ├─ Limited review capacity?
  │   └─ YES → Use RELAXED profile
  │   └─ NO → Continue
  │
  ├─ Need comprehensive coverage?
  │   └─ YES → Use STRICT profile
  │   └─ NO → Continue
  │
  └─ Default → Use NORMAL profile
```

---

## FAQ

**Q: Can I use different profiles for different detectors?**  
A: Not directly, but you can create a custom profile with mixed thresholds.

**Q: Will changing profiles affect historical comparisons?**  
A: Yes. Use the same profile for consistent comparisons over time.

**Q: Can I switch profiles mid-analysis?**  
A: Yes, but results won't be directly comparable. Document profile changes.

**Q: Which profile is fastest?**  
A: Relaxed profile is slightly faster (fewer anomalies to process), but difference is minimal.

**Q: Can I create multiple custom profiles?**  
A: Yes. Define as many custom profiles as needed in `config.yaml`.

---

## Related Documentation

- **Enhanced Detection Methodology**: `docs/enhanced_detection_methodology.md`
- **Auto-Tuning Guide**: `docs/auto_tuning_quick_reference.md`
- **Example Scripts**: `examples/threshold_profiles_demo.py`
- **Test Files**: `tests/test_detector_manager_profiles.py`

---

*Last Updated: 2025-11-01*
