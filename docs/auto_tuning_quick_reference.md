# Auto-Tuning Quick Reference

**Quick guide for using the auto-tuning feature**

---

## What is Auto-Tuning?

Auto-tuning automatically optimizes detection thresholds to minimize false positives while maintaining detection sensitivity.

**Benefits**:
- Eliminates manual threshold adjustment
- Adapts to your specific data patterns
- Reduces false positive rate to target level (default 5%)
- Provides recommended configuration

---

## Quick Start

### 1. Enable Auto-Tuning

Edit `config.yaml`:

```yaml
auto_tuning:
  enabled: true
  target_false_positive_rate: 0.05  # 5% target
  auto_apply: false  # Review recommendations first
```

### 2. Run Analysis

```bash
python main.py
```

### 3. Review Recommendations

Check the generated file:
```
output/recommended_thresholds_YYYYMMDD_HHMMSS.yaml
```

### 4. Apply Recommendations (Optional)

Copy recommended thresholds to your `config.yaml` or set `auto_apply: true`.

---

## Configuration Options

### Basic Configuration

```yaml
auto_tuning:
  enabled: true                      # Enable/disable auto-tuning
  target_false_positive_rate: 0.05   # Target FPR (5%)
  auto_apply: false                  # Auto-apply tuned thresholds
```

### Advanced Configuration

```yaml
auto_tuning:
  enabled: true
  target_false_positive_rate: 0.05
  min_anomalies_per_detector: 10      # Minimum anomalies required
  max_anomalies_per_detector: 1000    # Maximum anomalies allowed
  retuning_interval_days: 30          # Re-tune every 30 days
  auto_apply: false
```


---

## Common Use Cases

### Use Case 1: One-Time Optimization

**Goal**: Find optimal thresholds for your data

**Configuration**:
```yaml
auto_tuning:
  enabled: true
  auto_apply: false  # Review first
```

**Workflow**:
1. Run analysis: `python main.py`
2. Review `output/recommended_thresholds_*.yaml`
3. Copy recommended values to `config.yaml`
4. Disable auto-tuning: `enabled: false`

### Use Case 2: Automatic Application

**Goal**: Always use optimized thresholds

**Configuration**:
```yaml
auto_tuning:
  enabled: true
  auto_apply: true  # Apply automatically
```

**Workflow**:
1. Run analysis: `python main.py`
2. Thresholds automatically optimized and applied
3. Results use tuned thresholds

### Use Case 3: Periodic Re-Tuning

**Goal**: Adapt to changing data patterns over time

**Configuration**:
```yaml
auto_tuning:
  enabled: true
  retuning_interval_days: 30  # Re-tune monthly
  auto_apply: true
```

**Workflow**:
1. Initial run optimizes thresholds
2. Subsequent runs within 30 days use cached thresholds
3. After 30 days, automatically re-tunes
4. Tuning history stored for review

---

## Understanding the Output

### Recommended Thresholds File

```yaml
# output/recommended_thresholds_20251101_103000.yaml

tuning_metadata:
  timestamp: "2025-11-01 10:30:00"
  target_fpr: 0.05
  achieved_fpr: 0.048
  total_anomalies: 8234

recommended_thresholds:
  statistical:
    z_score: 3.2  # Optimized from 3.0
    iqr_multiplier: 1.6  # Optimized from 1.5
  
  geographic:
    regional_z_score: 2.7  # Optimized from 2.5
    capital_threshold: 3.8  # Optimized from 3.5
  
  # ... other thresholds

detector_statistics:
  StatisticalOutlierDetector:
    old_threshold: 3.0
    new_threshold: 3.2
    old_anomaly_count: 1456
    new_anomaly_count: 1123
    fpr_improvement: 0.023  # 2.3% reduction
```


### Tuning Report

The system also generates a human-readable report:

```
=== Auto-Tuning Report ===
Generated: 2025-11-01 10:30:00

Target False Positive Rate: 5.0%
Achieved False Positive Rate: 4.8%

Threshold Adjustments:
- StatisticalOutlierDetector: 3.0 → 3.2 (+6.7%)
- GeographicAnomalyDetector: 2.5 → 2.7 (+8.0%)
- TemporalAnomalyDetector: 100 → 110 (+10.0%)

Results:
- Total anomalies: 8,234 (was 12,456, -34%)
- False positive reduction: 23%
- Detection sensitivity: Maintained

Recommendations:
✓ Thresholds validated successfully
✓ All detectors within acceptable ranges
✓ Ready to apply to production
```

---

## Troubleshooting

### Problem: Auto-Tuning Takes Too Long

**Symptoms**: Auto-tuning runs for several minutes

**Causes**:
- Large dataset
- Many threshold candidates tested
- Complex detector logic

**Solutions**:
```yaml
auto_tuning:
  # Reduce search space
  threshold_search_step: 0.2  # Default 0.1
  
  # Limit iterations
  max_iterations: 20  # Default 50
```

### Problem: No Valid Thresholds Found

**Symptoms**: Auto-tuner reports "No valid thresholds found"

**Causes**:
- Target FPR too aggressive
- Min/max anomaly constraints too tight
- Insufficient data

**Solutions**:
```yaml
auto_tuning:
  # Relax target
  target_false_positive_rate: 0.10  # Increase from 0.05
  
  # Widen constraints
  min_anomalies_per_detector: 5     # Decrease from 10
  max_anomalies_per_detector: 2000  # Increase from 1000
```

### Problem: Thresholds Change Too Much

**Symptoms**: Recommended thresholds very different from current

**Causes**:
- Current thresholds far from optimal
- Data characteristics changed
- Initial thresholds were arbitrary

**Solutions**:
1. Review the tuning report to understand why
2. Apply changes gradually
3. Test on sample data first
4. Consider if data has genuinely changed


---

## Best Practices

### 1. Start with Review Mode

Always start with `auto_apply: false` to review recommendations before applying:

```yaml
auto_tuning:
  enabled: true
  auto_apply: false  # Review first
```

### 2. Validate on Sample Data

Before applying to production:
1. Run auto-tuning on sample data
2. Review anomaly counts and types
3. Manually verify some flagged anomalies
4. Ensure no critical anomalies missed

### 3. Monitor After Changes

After applying tuned thresholds:
- Compare anomaly counts with previous runs
- Review false positive rate
- Check if important anomalies still detected
- Adjust if needed

### 4. Keep Tuning History

The system maintains tuning history automatically:
```
output/tuning_history.json
```

Review this to understand threshold evolution over time.

### 5. Re-Tune Periodically

Data patterns change over time. Re-tune:
- Monthly for dynamic data
- Quarterly for stable data
- After major data source changes
- When false positive rate increases

---

## Advanced Features

### Custom FPR Estimation

Provide your own false positive labels:

```python
from src.auto_tuner import AutoTuner

tuner = AutoTuner(config)

# Provide labeled data
labeled_data = {
    'territory_id': [1, 2, 3, ...],
    'is_false_positive': [True, False, True, ...]
}

tuned_thresholds = tuner.optimize_with_labels(
    df=data,
    labels=labeled_data
)
```

### Detector-Specific Tuning

Tune only specific detectors:

```yaml
auto_tuning:
  enabled: true
  detectors_to_tune:
    - StatisticalOutlierDetector
    - GeographicAnomalyDetector
  # Other detectors use configured thresholds
```

### Constraint Customization

Set detector-specific constraints:

```yaml
auto_tuning:
  enabled: true
  detector_constraints:
    StatisticalOutlierDetector:
      min_anomalies: 50
      max_anomalies: 500
    GeographicAnomalyDetector:
      min_anomalies: 100
      max_anomalies: 2000
```

---

## FAQ

**Q: How long does auto-tuning take?**  
A: Typically 30-60 seconds for 3,000 municipalities. Scales linearly with data size.

**Q: Will auto-tuning miss important anomalies?**  
A: No. Auto-tuning optimizes for false positives while maintaining detection sensitivity. Validation ensures minimum anomaly counts.

**Q: Can I use auto-tuning with custom profiles?**  
A: Yes. Auto-tuning can start from any profile and optimize from there.

**Q: How often should I re-tune?**  
A: Monthly for dynamic data, quarterly for stable data. Monitor false positive rate to determine if re-tuning needed.

**Q: What if I don't have enough historical data?**  
A: Auto-tuning requires at least one full analysis run. Start with a configuration profile, then enable auto-tuning after first run.

**Q: Can I manually override auto-tuned thresholds?**  
A: Yes. Auto-tuned thresholds are recommendations. You can manually adjust any threshold in `config.yaml`.

---

## Related Documentation

- **Enhanced Detection Methodology**: `docs/enhanced_detection_methodology.md`
- **Configuration Profiles**: See "Configuration Profiles" section in methodology
- **Example Scripts**: `examples/auto_tuning_workflow_demo.py`
- **Test Files**: `tests/test_auto_tuner_fpr.py`

---

*Last Updated: 2025-11-01*
