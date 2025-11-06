# Implementation Plan

## Overview

Данный план описывает пошаговую реализацию улучшений системы обнаружения аномалий. Задачи организованы по 4 фазам с инкрементальной доставкой ценности.

---

## Phase 1: Critical Fixes (Week 1-2)

- [ ] 1. Fix StatisticalOutlierDetector KeyError
- [x] 1.1 Analyze root cause of KeyError in z-score calculation

  - Review index handling in `detect_zscore_outliers()`
  - Identify where index mismatch occurs
  - _Requirements: 1.1_

-

- [x] 1.2 Implement safe index access in StatisticalOutlierDetector

  - Replace direct indexing with `.loc[]` and index validation
  - Add index existence checks before accessing values

  - Maintain original indices when working with subsets
  - _Requirements: 1.1_

- [x] 1.3 Apply same fix to IQR and percentile methods

  - Update `detect_iqr_outliers()` with safe indexing
  - Update `detect_percentile_outliers()` with safe indexing

  - _Requirements: 1.1_

- [x] 1.4 Add regression tests for StatisticalOutlierDetector

  - Test with missing indices
  - Test with filtered data

  - Test with edge cases (empty data, single value)
  - _Requirements: 1.1, 15.2_

- [ ] 2. Implement temporal data analysis
- [x] 2.1 Create TemporalMetadata dataclass

  - Define structure for temporal metadata
  - Include fields: has_temporal_data, temporal_columns, granularity, periods_per_territory, date_range
  - _Requirements: 2.1, 2.3_

- [x] 2.2 Implement temporal structure detection in DataLoader

  - Add `analyze_temporal_structure()` method
  - Detect temporal columns (date, period, year, month, quarter)
  - Determine granularity (daily, monthly, quarterly, yearly)
  - Count periods per territory
  - _Requirements: 2.1, 2.2_

- [x] 2.3 Implement duplicate detection logic

  - Add `detect_duplicates()` method
  - Distinguish temporal duplicates from data errors
  - Generate DuplicateReport with recommendations
  - _Requirements: 1.2, 2.2_

- [x] 2.4 Implement temporal data aggregation

  - Add `aggregate_temporal_data()` method
  - Support aggregation methods: latest, mean, median
  - Make aggregation method configurable
  - _Requirements: 1.4, 2.5_

- [ ]\* 2.5 Add tests for temporal analysis

  - Test temporal structure detection
  - Test duplicate detection (temporal vs errors)
  - Test aggregation methods
  - _Requirements: 2.1, 2.2, 15.1_

- [ ] 3. Implement improved source mapping
- [x] 3.1 Create explicit source mapping in DataLoader

  - Add `create_source_mapping()` method
  - Use column prefixes as primary method (consumption*, salary*, population\_)
  - Implement fallback heuristics for ambiguous names
  - _Requirements: 8.1, 8.2_

- [x] 3.2 Update detectors to use explicit source mapping

  - Replace `_determine_data_source()` with mapping lookup
  - Add source to anomaly metadata
  - Log warnings for ambiguous columns
  - _Requirements: 8.2, 8.3, 8.4_

- [ ]\* 3.3 Add tests for source mapping

  - Test prefix-based mapping
  - Test fallback heuristics
  - Test ambiguous column handling
  - _Requirements: 8.1, 8.2, 15.1_

- [ ] 4. Create DetectorManager component
- [x] 4.1 Implement DetectorManager class

  - Create `src/detector_manager.py`
  - Implement `run_all_detectors()` with error handling
  - Implement `run_detector_safe()` with try-catch
  - Add detector statistics tracking
  - _Requirements: 1.1, 13.2_

- [x] 4.2 Implement ThresholdManager class

  - Add threshold management logic
  - Support profile-based thresholds (strict/normal/relaxed)
  - _Requirements: 9.1, 9.2, 9.3_
    nd `load_profile()` methods
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 4.3 Integrate DetectorManager into main pipeline

  - Update `main.py` to use DetectorManager
  - Replace direct detector calls with manager
  - Ensure backward compatibility
  - _Requirements: 1.1, 14.4_

- [x] 4.4 Add tests for DetectorManager

  - Test error handling (detector failure)
  - Test statistics collection
  - Test threshold management
  - _Requirements: 13.2, 15.1_

- [ ] 5. Enhance logging and diagnostics
- [x] 5.1 Implement structured logging

  - Add context information to log messages
  - Use extra fields for structured data
  - Implement log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - _Requirements: 13.1, 13.3, 13.4_

- [x] 5.2 Add data quality warnings

  - Log duplicate territory_ids with count
  - Log missing value statistics
  - Log detector execution times
  - _Requirements: 1.3, 13.4, 13.5_

- [x] 5.3 Improve error messages

  - Add full stack traces with context
  - Include data shape and detector name in errors
  - Sanitize sensitive information
  - _Requirements: 13.1_

---

## Phase 2: Quality Improvements (Week 3-5)

- [ ] 6. Create DataPreprocessor component
- [x] 6.1 Implement DataPreprocessor class

  - Create `src/data_preprocessor.py`
  - Implement `preprocess()` method
  - Add configuration support
  - _Requirements: 7.1, 7.2_

- [x] 6.2 Implement MunicipalityClassifier

  - Create classification logic (capital/urban/rural)
  - Use population threshold (50,000) for urban
  - Use name patterns for classification
  - Add configurable capital cities list
  - _Requirements: 3.1, 3.2_

- [x] 6.3 Implement RobustStatsCalculator

  - Calculate median, MAD, IQR
  - Calculate percentiles (1st, 5th, 25th, 75th, 95th, 99th)
  - Detect skewness
  - _Requirements: 7.1, 7.2_

- [x] 6.4 Implement data normalization

  - Add per-capita normalization where applicable
  - Implement log transformation for skewed data (skewness > 2)
  - Implement winsorization (1st and 99th percentiles)
  - _Requirements: 7.3, 7.5_

- [ ]\* 6.5 Add tests for DataPreprocessor

  - Test municipality classification
  - Test robust statistics calculation
  - Test normalization methods
  - _Requirements: 7.1, 7.2, 15.1_

- [ ] 7. Enhance GeographicAnomalyDetector
- [x] 7.1 Implement type-aware comparison

  - Group by region AND municipality_type
  - Apply type-specific thresholds (capital: 3.5, urban: 2.5, rural: 2.0)
  - _Requirements: 3.2, 3.3_

- [x] 7.2 Replace mean/std with median/MAD

  - Update `detect_regional_outliers()` to use robust statistics
  - Calculate robust z-scores: (value - median) / (1.4826 \* MAD)
  - Update `detect_cluster_outliers()` similarly
  - _Requirements: 3.5, 7.1, 7.2_

- [x] 7.3 Update severity scoring for geographic anomalies

  - Reduce base severity for natural differences
  - Increase severity for same-type outliers
  - _Requirements: 3.5, 5.1_

- [ ]\* 7.4 Add tests for enhanced geographic detector

  - Test type-aware comparison
  - Test robust statistics usage
  - Verify reduced false positives
  - _Requirements: 3.1, 3.2, 15.1_

- [ ] 8. Implement priority scoring and ranking
- [x] 8.1 Add priority score calculation to ResultsAggregator

  - Implement `calculate_priority_score()` method
  - Apply type weights (logical: 1.5, cross_source: 1.2, geographic: 0.8)
  - Apply indicator weights (population: 1.3, consumption_total: 1.2)
  - _Requirements: 5.1, 5.2, 5.5_

- [x] 8.2 Implement anomaly grouping

  - Add `group_related_anomalies()` method
  - Group by territory_id
  - Identify patterns (multiple indicators, same indicator across detectors)
  - _Requirements: 4.5, 5.3_

- [x] 8.3 Implement root cause analysis

  - Add `identify_root_cause()` method
  - Detect patterns: missing data, duplicates, systematic discrepancies
  - Generate Russian-language root cause descriptions
  - _Requirements: 4.5_

- [x] 8.4 Update aggregation to sort by priority

  - Sort anomalies by priority_score descending
  - Add priority_score to anomaly records
  - _Requirements: 5.4_

- [ ]\* 8.5 Add tests for priority scoring

  - Test priority calculation
  - Test anomaly grouping
  - Test root cause identification
  - _Requirements: 5.1, 5.2, 15.1_

- [ ] 9. Implement management-friendly descriptions
- [ ] 9.1 Create DescriptionFormatter class

  - Implement `format_for_management()` method
  - Transform technical terms to business language
  - Use relative metrics (в X раз выше/ниже)
  - Add comparison context
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 9.2 Update anomaly record structure

  - Add description_management field
  - Add relative_deviation field
  - Add comparison_context field
  - Maintain backward compatibility
  - _Requirements: 4.1, 14.1, 14.2_

- [ ] 9.3 Integrate formatter into exporter

  - Call formatter for each anomaly
  - Add management descriptions to Excel export
  - Make feature configurable (use_management_descriptions)
  - _Requirements: 4.1, 14.4_

- [ ]\* 9.4 Add tests for description formatting

  - Test technical to business transformation
  - Test relative metrics generation
  - Test Russian language output
  - _Requirements: 4.1, 4.2, 15.1_

- [ ] 10. Create executive summary
- [x] 10.1 Implement ExecutiveSummaryGenerator class

  - Calculate summary statistics
  - Identify top 10 municipalities by risk
  - Generate key findings in Russian
  - Generate recommendations
  - _Requirements: 12.2, 12.3_

- [x] 10.2 Add executive summary sheet to Excel export

  - Create separate sheet with summary
  - Include top municipalities table
  - Add key findings as bullet points
  - Highlight critical anomalies in red
  - _Requirements: 12.1, 12.5_

- [x] 10.3 Create dashboard visualization

  - Implement `create_dashboard_summary()` method
  - Generate single-page dashboard with 4 charts
  - Add key metrics text boxes
  - _Requirements: 12.4_

- [ ]\* 10.4 Add tests for executive summary

  - Test summary statistics calculation
  - Test top municipalities selection
  - Test key findings generation
  - _Requirements: 12.2, 12.3, 15.1_

- [ ] 11. Improve missing value handling
- [x] 11.1 Add missingness analysis

  - Calculate missing percentage per indicator
  - Calculate missing percentage per municipality
  - _Requirements: 11.1, 11.3_

- [x] 11.2 Implement indicator filtering

  - Skip indicators with >50% missing values
  - Log warnings for skipped indicators
  - _Requirements: 11.2_

- [x] 11.3 Implement municipality flagging

  - Flag municipalities with >70% missing indicators
  - Add to logical consistency anomalies

  - _Requirements: 11.3, 11.5_

- [x] 11.4 Update statistics calculation

  - Use pairwise deletion for missing values

  - Document missing value handling in methodology
  - _Requirements: 11.4_

- [ ]\* 11.5 Add tests for missing value handling

  - Test indicator filtering
  - Test municipality flagging
  - Test statistics with missing values
  - _Requirements: 11.1, 11.2, 15.1_

- [ ] 12. Implement validation metrics
- [x] 12.1 Add detection metrics calculation

  - Count anomalies by type and severity
  - Calculate percentage of municipalities affected

  - Calculate anomaly rate per 1000 municipalities
  - _Requirements: 10.1, 10.2, 10.3_

- [x] 12.2 Add data quality metrics

  - Calculate data completeness score
  - Calculate consistency score
  - Include in validation report
  - _Requirements: 10.4_

- [x] 12.3 Implement anomaly count warnings

  - Check if anomaly count is in expected range
  - Log warning if count is too high/low
  - Suggest threshold adjustment
  - _Requirements: 10.5_

- [ ]\* 12.4 Add tests for validation metrics
  - Test metrics calculation
  - Test warning generation
  - Test expected range validation
  - _Requirements: 10.1, 10.2, 15.1_

---

## Phase 3: Auto-tuning (Week 6-7)

- [ ] 13. Implement threshold optimization
- [x] 13.1 Create AutoTuner class

  - Create `src/auto_tuner.py`
  - Implement threshold optimization algorithm
  - Support multiple optimization strategies
  - _Requirements: 6.1, 6.2_

- [x] 13.2 Implement false positive rate calculation

  - Analyze historical detection results
  - Calculate FPR for each detector
  - Identify optimal thresholds to minimize FPR
  - _Requirements: 6.2, 6.3_

- [x] 13.3 Implement threshold validation

  - Ensure at least 95% of normal municipalities not flagged
  - Validate threshold ranges
  - Check minimum/maximum anomaly counts per detector
  - _Requirements: 6.3, 6.4_

- [x] 13.4 Implement periodic re-tuning


  - Add scheduling logic for periodic re-evaluation
  - Make interval configurable (default: 30 days)
  - Store tuning history

  - _Requirements: 6.5_

- [ ]\* 13.5 Add tests for auto-tuner

  - Test threshold optimization
  - Test FPR calculation
  - Test validation logic
  - Test periodic re-tuning
  - _Requirements: 6.1, 6.2, 15.3_

- [ ] 14. Implement configuration profiles
- [x] 14.1 Create threshold profile definitions






  - Define strict profile (lower thresholds)
  - Define normal profile (current thresholds)
  - Define relaxed profile (higher thresholds)
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 14.2 Implement profile loading






  - Add profile selection to config.yaml
  - Implement profile merging with defaults
  - Validate profile completeness



  - _Requirements: 9.1, 9.4, 9.5_

- [x] 14.3 Integrate profiles with DetectorManager







  - Load profile on initialization
  - Apply profile thresholds to detectors
  - Support runtime profile switching
  - _Requirements: 9.1, 9.2_

- [x] 14.4 Add tests for configuration profiles






  - Test profile loading
  - Test profile merging
  - Test profile validation
  - _Requirements: 9.1, 9.4, 15.1_

- [ ] 15. Integrate auto-tuning into pipeline
- [x] 15.1 Add auto-tuning configuration






  - Add auto_tuning section to config.yaml
  - Make auto-tuning opt-in (enabled: false by default)
  - Add tuning parameters (target FPR, min/max anomalies)
  - _Requirements: 6.1, 14.4_

- [x] 15.2 Implement tuning workflow






  - Run auto-tuner before detection (if enabled)
  - Apply tuned thresholds to detectors
  - Log tuning results
  - _Requirements: 6.1, 6.4_


- [x] 15.3 Generate recommended configuration




  - Export tuned thresholds to file
  - Generate human-readable tuning report
  - Include tuning statistics and rationale
  - _Requirements: 6.4_
- [x] 15.4 Add integration tests for auto-tuning









- [ ] 15.4 Add integration tests for auto-tuning


  - Test full pipeline with auto-tuning enabled
  - Test threshold application
  - Test configuration export
  - _Requirements: 6.1, 15.1_

---

## Phase 4: Testing and Documentation (Continuous)

- [ ] 16. Update configuration schema
- [x] 16.1 Add new configuration sections






  - Add detection_profile
  - Add temporal settings
  - Add municipality_classification
  - Add threshold_profiles
  - Add auto_tuning
  - Add robust_statistics
  - Add priority_weights
  - _Requirements: 14.1, 14.2_

- [x] 16.2 Implement configuration validation





  - Validate schema on load
  - Check required fields
  - Validate value ranges
  - _Requirements: 14.1_

- [x] 16.3 Support old configuration format






  - Detect old vs new format
  - Auto-migrate old format to new
  - Log migration warnings
  - _Requirements: 14.3_

- [ ]\* 16.4 Add tests for configuration

  - Test new configuration loading
  - Test old configuration migration
  - Test validation logic
  - _Requirements: 14.3, 15.1_

- [ ] 17. Update documentation
- [x] 17.1 Update README.md





  - Document new features
  - Add configuration examples
  - Update usage instructions
  - _Requirements: 14.5_

- [x] 17.2 Update algorithms.md






  - Document robust statistics methods
  - Document type-aware comparison
  - Document priority scoring
  - _Requirements: 7.1, 7.2, 5.1_

- [ ] 17.3 Create migration guide

  - Document breaking changes (if any)
  - Provide migration steps
  - Include before/after examples
  - _Requirements: 14.5_

- [x] 17.4 Update methodology documentation






  - Document new detection methods
  - Explain auto-tuning process
  - Document configuration profiles
  - _Requirements: 6.1, 9.1_

- [ ] 18. Integration testing
- [x] 18.1 Test full pipeline with all phases








  - Run with temporal data
  - Run with duplicates
  - Run with all detectors enabled
  - Verify output compatibility
  - _Requirements: 14.1, 15.1_

- [x] 18.2 Test error scenarios






  - Test with detector failures
  - Test with invalid configuration
  - Test with corrupted data
  - _Requirements: 13.2, 15.1_

- [ ] 18.3 Test backward compatibility

  - Run with old configuration
  - Verify output format unchanged
  - Test with existing scripts
  - _Requirements: 14.1, 14.2, 14.3_

- [ ] 18.4 Performance testing

  - Measure execution time
  - Measure memory usage
  - Compare with baseline
  - _Requirements: 15.1_

- [ ] 19. Validation and acceptance
- [ ] 19.1 Run on production data

  - Execute full pipeline on actual СберИндекс data
  - Verify anomaly count reduction
  - Validate executive summary quality
  - _Requirements: 10.1, 12.1_

- [ ] 19.2 Review results with stakeholders

  - Present executive summary to management
  - Gather feedback on descriptions
  - Validate priority ranking
  - _Requirements: 4.1, 5.4, 12.3_

- [ ] 19.3 Measure improvement metrics

  - Compare anomaly counts (before/after)
  - Measure false positive reduction
  - Validate auto-tuning effectiveness
  - _Requirements: 6.2, 10.5_

- [ ] 19.4 Create final validation report
  - Document all improvements
  - Include metrics and statistics
  - Provide recommendations for future work
  - _Requirements: 10.1, 10.2, 10.3_

---

## Summary

**Total Tasks**: 19 top-level tasks, 75+ sub-tasks
**Estimated Duration**: 7 weeks
**Optional Tasks**: 19 (marked with \*)

**Phase Breakdown**:

- Phase 1 (Critical Fixes): 5 tasks, 18 sub-tasks
- Phase 2 (Quality): 7 tasks, 32 sub-tasks
- Phase 3 (Auto-tuning): 3 tasks, 13 sub-tasks
- Phase 4 (Testing): 4 tasks, 12 sub-tasks

**Key Deliverables**:

- ✅ All detectors working without errors
- ✅ 50-70% reduction in false positives
- ✅ Management-friendly reports
- ✅ Automatic threshold optimization
- ✅ Full backward compatibility
