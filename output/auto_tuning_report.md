# Auto-Tuning Report

**Tuning ID:** tuning_20251104_021825
**Timestamp:** 2025-11-04 02:18:25
**Optimization Strategy:** adaptive
**Target False Positive Rate:** 0.050 (5.0%)

## Executive Summary

The auto-tuning process optimized detection thresholds using the **adaptive** strategy to minimize false positives while maintaining detection capability.

### Key Metrics

- **Total Anomalies Before:** 1,181
- **Total Anomalies After:** 3,920
- **Anomaly Reduction:** -2,739 (-231.9%)
- **Average FPR Before:** 0.0162 (1.62%)
- **Average FPR After:** 0.0217 (2.17%)
- **FPR Reduction:** 97.8%

✅ **Target FPR achieved:** The optimized thresholds meet the target FPR of 0.050.

## Detector-Specific Results

The following sections detail the threshold optimizations for each detector.

### Statistical Detector

#### Threshold Changes

| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| `z_score` | 3.000 | 2.000 | -33.3% |
| `iqr_multiplier` | 1.500 | 1.000 | -33.3% |
| `percentile_lower` | 1.000 | 1.000 | +0.0% |
| `percentile_upper` | 99.000 | 99.000 | +0.0% |

#### Performance Metrics

- **Estimated FPR Before:** 0.0143 (1.43%)
- **Estimated FPR After:** 0.0481 (4.81%)
- **FPR Improvement:** 95.2%
- **Confidence Score:** 0.80/1.00

- **Estimated Anomalies Before:** 1,168
- **Estimated Anomalies After:** 3,917
- **Anomaly Reduction:** -2,749 (-235.4%)

#### Rationale

The **adaptive** strategy uses data-driven threshold optimization, analyzing actual distributions to find optimal detection points. Thresholds were decreased by 16.7%, indicating the original thresholds may have been too strict and missing true anomalies. For statistical detection, higher z-score thresholds reduce sensitivity to natural variation in the data, focusing on more extreme outliers. The optimization has **high confidence** (score: 0.80) based on sufficient data for analysis.

### Geographic Detector

#### Threshold Changes

| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| `regional_z_score` | 3.500 | 3.850 | +10.0% |
| `cluster_threshold` | 4.000 | 4.400 | +10.0% |

#### Performance Metrics

- **Estimated FPR Before:** 0.0005 (0.05%)
- **Estimated FPR After:** 0.0001 (0.01%)
- **FPR Improvement:** 100.0%
- **Confidence Score:** 0.70/1.00

- **Estimated Anomalies Before:** 13
- **Estimated Anomalies After:** 3
- **Anomaly Reduction:** 10 (76.9%)

#### Rationale

The **adaptive** strategy uses data-driven threshold optimization, analyzing actual distributions to find optimal detection points. Thresholds were moderately increased by 10.0% to reduce false positives while maintaining good detection capability. For geographic detection, adjusted thresholds account for natural regional variation, reducing false positives from legitimate urban-rural differences. The optimization has **moderate confidence** (score: 0.70). Results should be validated with actual detection runs.

### Temporal Detector

#### Threshold Changes

| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| `spike_threshold` | 100.000 | 110.000 | +10.0% |
| `drop_threshold` | -50.000 | -55.000 | -10.0% |
| `volatility_multiplier` | 2.000 | 2.100 | +5.0% |

#### Performance Metrics

- **Estimated FPR Before:** 0.0200 (2.00%)
- **Estimated FPR After:** 0.0160 (1.60%)
- **FPR Improvement:** 98.4%
- **Confidence Score:** 0.50/1.00

#### Rationale

The **adaptive** strategy uses data-driven threshold optimization, analyzing actual distributions to find optimal detection points. Thresholds required minimal adjustment, suggesting they were already well-calibrated. For temporal detection, threshold adjustments balance sensitivity to genuine changes against normal seasonal or cyclical variations. The optimization has **low confidence** (score: 0.50) due to limited data. Manual review and adjustment may be needed.

### Cross Source Detector

#### Threshold Changes

| Parameter | Original | Optimized | Change |
|-----------|----------|-----------|--------|
| `correlation_threshold` | 0.500 | 0.475 | -5.0% |
| `discrepancy_threshold` | 50.000 | 55.000 | +10.0% |

#### Performance Metrics

- **Estimated FPR Before:** 0.0300 (3.00%)
- **Estimated FPR After:** 0.0225 (2.25%)
- **FPR Improvement:** 97.8%
- **Confidence Score:** 0.60/1.00

#### Rationale

The **adaptive** strategy uses data-driven threshold optimization, analyzing actual distributions to find optimal detection points. Thresholds required minimal adjustment, suggesting they were already well-calibrated. For cross-source comparison, threshold adjustments account for expected discrepancies between different data sources while flagging significant inconsistencies. The optimization has **moderate confidence** (score: 0.60). Results should be validated with actual detection runs.

## Recommendations

- ✅ Target FPR achieved. The optimized thresholds are ready for production use.
- Limited anomaly reduction (-231.9%). Consider more aggressive threshold adjustments if false positives remain high.
- Low confidence in optimization for: temporal. Validate these detectors with actual detection runs and adjust manually if needed.
- Schedule next re-tuning for 2025-12-04 (30 days from last tuning).
- Monitor detection results over the next few runs to ensure thresholds are working as expected.
- Consider creating a custom threshold profile based on these optimized values for future use.

## Next Steps

1. **Review the optimized thresholds** in the exported configuration file
2. **Apply the thresholds** by updating your `config.yaml` with the optimized values
3. **Run detection** with the new thresholds and validate results
4. **Schedule re-tuning** in 30 days or when data patterns change

---

*Report generated by AutoTuner on 2025-11-04 02:18:25*