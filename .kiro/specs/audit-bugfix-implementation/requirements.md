# Requirements Document

## Introduction

This specification addresses critical bugs and design flaws identified in the СберИндекс Anomaly Detection System through an independent code audit. The audit revealed fundamental issues with data processing, statistical methods, and detector logic that result in 73-95% false positive rate. This feature will systematically fix these issues to achieve a realistic anomaly detection rate of 20-35% of territories with high-confidence anomalies.

## Glossary

- **System**: СберИндекс Anomaly Detection System
- **Temporal Structure**: Time-series data with date/period columns preserved for trend analysis
- **Connection Graph**: Network of 4.7M territory connections with distance metrics from connection.parquet
- **False Positive (FP)**: Legitimate pattern incorrectly flagged as anomaly
- **Cross-Source Comparison**: Comparison between СберИндекс and Росстат metrics
- **Z-Score Threshold**: Standard deviation multiplier for outlier detection
- **Skewed Distribution**: Statistical distribution with asymmetry (skewness > 2.0)
- **Legitimate Pattern**: Expected variation due to geography, tourism, or economic factors
- **Detector**: Component that identifies specific anomaly types (statistical, temporal, geographic, etc.)

## Requirements

### Requirement 1: Preserve Temporal Data Structure

**User Story:** As a data analyst, I want consumption data to retain temporal structure, so that I can detect time-based anomalies like spikes and seasonal patterns.

#### Acceptance Criteria

1. WHEN the System loads consumption.parquet, THE System SHALL preserve the date column in the pivoted DataFrame
2. WHEN the System pivots consumption data, THE System SHALL retain all 303,126 temporal records instead of aggregating to 2,571 records
3. WHEN TemporalAnomalyDetector analyzes data, THE System SHALL detect at least one temporal anomaly type (spike, drop, or volatility)
4. WHEN StatisticalOutlierDetector processes temporal data, THE System SHALL group by territory_id before calculating statistics
5. THE System SHALL create consumption column names with "consumption_" prefix to avoid naming conflicts

### Requirement 2: Utilize Connection Graph for Geographic Analysis

**User Story:** As a geographic analyst, I want the system to use real territory connections, so that neighbor comparisons reflect actual proximity rather than administrative boundaries.

#### Acceptance Criteria

1. THE System SHALL load connection.parquet containing 4.7M territory connections
2. WHEN GeographicAnomalyDetector identifies neighbors, THE System SHALL query the connection graph with configurable distance threshold
3. WHEN a territory has fewer than 3 neighbors within the distance threshold, THE System SHALL skip cluster analysis for that territory
4. WHEN connection graph is unavailable, THE System SHALL fall back to region-based clustering with a warning logged
5. THE System SHALL return neighbor lists containing territory IDs within the specified distance (default 50km)

### Requirement 3: Disable Invalid Cross-Source Comparisons

**User Story:** As a system administrator, I want to disable cross-source comparisons that compare incompatible metrics, so that false positives from invalid comparisons are eliminated.

#### Acceptance Criteria

1. THE System SHALL provide a configuration option to enable or disable CrossSourceComparator
2. WHEN CrossSourceComparator is disabled in configuration, THE System SHALL not initialize the detector
3. THE System SHALL set CrossSourceComparator to disabled by default in config.yaml
4. WHEN CrossSourceComparator is enabled, THE System SHALL log a warning about verifying metric pair validity
5. THE System SHALL reduce total anomaly count by approximately 2,414 when CrossSourceComparator is disabled

### Requirement 4: Apply Russia-Specific Statistical Thresholds

**User Story:** As a data scientist, I want statistical thresholds calibrated for Russia's extreme geographic heterogeneity, so that natural variation between Moscow and remote regions is not flagged as anomalous.

#### Acceptance Criteria

1. THE System SHALL use z-score threshold of 5.0 or higher for statistical outlier detection
2. THE System SHALL use IQR multiplier of 3.0 or higher for robust outlier detection
3. THE System SHALL use percentile thresholds of 0.1 and 99.9 for extreme outlier detection
4. THE System SHALL use regional z-score threshold of 6.0 or higher for geographic analysis
5. THE System SHALL use cluster threshold of 5.0 or higher for neighbor comparisons

### Requirement 5: Handle Skewed Distributions with Log Transform

**User Story:** As a statistician, I want the system to detect and transform highly skewed distributions, so that power-law metrics like population and consumption are analyzed correctly.

#### Acceptance Criteria

1. WHEN StatisticalOutlierDetector analyzes an indicator, THE System SHALL calculate the skewness coefficient
2. WHEN skewness absolute value exceeds 2.0, THE System SHALL apply log1p transformation before calculating z-scores
3. WHEN skewness absolute value is 2.0 or less, THE System SHALL use original values for z-score calculation
4. THE System SHALL log a debug message indicating when log-transform is applied and the skewness value
5. THE System SHALL clip negative values to zero before applying log transformation

### Requirement 6: Disable Auto-Tuning Without Ground Truth

**User Story:** As a system operator, I want auto-tuning disabled when no ground truth dataset exists, so that thresholds are not incorrectly optimized in the wrong direction.

#### Acceptance Criteria

1. THE System SHALL provide a configuration option to enable or disable auto-tuning
2. THE System SHALL set auto-tuning to disabled by default in config.yaml
3. WHEN auto-tuning is disabled, THE System SHALL use static threshold values from configuration
4. THE System SHALL not modify threshold values during execution when auto-tuning is disabled
5. THE System SHALL prevent threshold softening that increases false positive rate

### Requirement 7: Filter Legitimate Geographic Patterns

**User Story:** As a domain expert, I want known legitimate patterns (tourism zones, business districts, remote areas) to be filtered out, so that expected variations are not reported as anomalies.

#### Acceptance Criteria

1. THE System SHALL load legitimate pattern definitions from legitimate_patterns_config.yaml
2. WHEN LegitimatePatternFilter processes anomalies, THE System SHALL mark anomalies matching legitimate patterns with is_legitimate_pattern flag
3. THE System SHALL support at least 6 legitimate pattern categories (tourism, shift-work, industrial, border, transport, remote)
4. WHEN legitimate patterns are filtered, THE System SHALL log the count of reclassified anomalies
5. THE System SHALL remove anomalies flagged as legitimate patterns from the final output

### Requirement 8: Create Russia-Specific Detection Profile

**User Story:** As a configuration manager, I want a pre-configured detection profile optimized for Russian geographic heterogeneity, so that I can quickly apply appropriate thresholds.

#### Acceptance Criteria

1. THE System SHALL define a "custom_russia" threshold profile in config.yaml
2. THE System SHALL set "custom_russia" as the active detection_profile by default
3. WHEN "custom_russia" profile is active, THE System SHALL apply z_score of 5.0, IQR multiplier of 3.0, and regional z_score of 6.0
4. THE System SHALL disable CrossSourceComparator and impossible ratio checks in the "custom_russia" profile
5. THE System SHALL set temporal thresholds to spike_threshold of 200 and drop_threshold of -80 in the profile

### Requirement 9: Validate Detection Quality Metrics

**User Story:** As a quality assurance analyst, I want to monitor detection quality metrics after fixes, so that I can verify the system achieves target false positive reduction.

#### Acceptance Criteria

1. WHEN the System completes anomaly detection, THE System SHALL report total anomaly count between 3,000 and 5,000
2. WHEN the System completes detection, THE System SHALL report percentage of flagged territories between 20% and 35%
3. THE System SHALL detect at least one temporal anomaly when temporal data is available
4. THE System SHALL report zero cross-source anomalies when CrossSourceComparator is disabled
5. THE System SHALL reduce total anomalies by at least 70% compared to pre-fix baseline of 16,682

### Requirement 10: Maintain Backward Compatibility

**User Story:** As a system maintainer, I want existing configuration files and data formats to remain compatible, so that the bugfix does not break existing workflows.

#### Acceptance Criteria

1. WHEN connection.parquet is not found, THE System SHALL continue execution with region-based geographic analysis
2. WHEN legitimate_patterns_config.yaml is not found, THE System SHALL skip pattern filtering with a warning
3. THE System SHALL support both old and new configuration formats for threshold profiles
4. WHEN temporal data is not available, THE System SHALL skip temporal detection without errors
5. THE System SHALL maintain existing CSV and Excel export formats for anomaly results
