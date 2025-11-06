# Implementation Plan

- [x] 1. Set up project structure and configuration

  - Create directory structure (src/, tests/, output/)
  - Create config.yaml with all detection thresholds
  - Create requirements.txt with all dependencies
  - Create main.py entry point script
  - Set up logging configuration
  - _Requirements: 10.3, 10.4_

- [x] 2. Implement data loading module

- [x] 2.1 Create DataLoader class with file loading methods

  - Implement load_sberindex_data() for connection, consumption, market_access parquet files
  - Implement load_rosstat_data() for population, migration, salary parquet files
  - Implement load_municipal_dict() for Excel dictionary
  - Add error handling for missing files with graceful degradation
  - _Requirements: 1.1, 1.5_

- [x] 2.2 Implement data merging and validation

  - Create merge_datasets() method to join all data by territory_id
  - Implement validate_data() to check for missing values and duplicates
  - Add data completeness scoring for each municipality
  - Log validation results and data quality metrics
  - _Requirements: 1.2, 1.3, 1.4_

- [x] 3. Implement statistical outlier detector

- [x] 3.1 Create BaseAnomalyDetector abstract class

  - Define detect() abstract method with standard return format
  - Create helper methods for severity score calculation
  - Implement anomaly record creation with all required fields
  - _Requirements: 2.5_

- [x] 3.2 Implement StatisticalOutlierDetector class

  - Create detect_zscore_outliers() method using scipy.stats
  - Implement detect_iqr_outliers() using quartile-based detection
  - Add detect_percentile_outliers() for top/bottom 1% identification
  - Calculate severity scores based on deviation magnitude
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Implement cross-source comparison detector
- [x] 4.1 Create CrossSourceComparator class

  - Implement calculate_correlations() between СберИндекс and Росстат indicators
  - Create detect_large_discrepancies() to find >50% differences
  - Add rank_by_discrepancy() to sort municipalities by deviation
  - Calculate percentage differences for each comparable indicator pair
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 5. Implement temporal anomaly detector

- [x] 5.1 Create TemporalAnomalyDetector class

  - Implement detect_sudden_spikes() for >100% growth or <-50% drops
  - Create detect_trend_reversals() to identify direction changes
  - Add detect_high_volatility() using standard deviation of changes
  - Calculate period-over-period growth rates for all indicators
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5.2 Add seasonal anomaly detection

  - Implement seasonal pattern comparison if temporal data available
  - _Requirements: 4.5_

- [ ] 6. Implement geographic anomaly detector
- [x] 6.1 Create GeographicAnomalyDetector class

  - Implement detect_regional_outliers() using regional z-scores
  - Create detect_cluster_outliers() to find municipalities differing from neighbors
  - Add separate analysis for urban vs rural municipalities
  - Calculate deviations from regional means for each indicator
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Implement logical consistency checker

- [x] 7.1 Create LogicalConsistencyChecker class

  - Implement detect_negative_values() for indicators that must be positive
  - Create detect_impossible_ratios() to find logically inconsistent values
  - Add detect_contradictory_indicators() for conflicting metrics

  - Identify duplicate or inconsistent municipality identifiers
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Implement results aggregation
- [x] 8.1 Create ResultsAggregator class

  - Implement combine_anomalies() to merge results from all detectors
  - Create calculate_municipality_scores() to compute total anomaly scores
  - Add rank_anomalies() to sort by severity
  - Implement categorize_anomalies() to group by type
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 9. Implement results export

- [x] 9.1 Create CSV exporter

  - Implement export_master_csv() with all required columns
  - Add timestamp to filename
  - Ensure proper encoding for Russian text
  - _Requirements: 8.1, 8.5_

- [x] 9.2 Create Excel exporter with multiple sheets

  - Implement export_summary_excel() with Overview sheet
  - Add sheets for each anomaly type (Statistical_Outliers, Temporal_Anomalies, etc.)
  - Create Top_Anomalous_Municipalities sheet with top 50
  - Add Data_Dictionary sheet explaining all columns
  - Include descriptive statistics for each anomaly type
  - _Requirements: 8.2, 8.3, 8.4, 8.5_

- [x] 9.3 Create visualization generator

  - Implement bar chart for anomaly distribution by type
  - Create horizontal bar chart for top 20 municipalities

  - Add heatmap for geographic distribution by region
  - Generate histogram for severity score distribution
  - Save all visualizations as PNG files
  - _Requirements: 9.2_

- [x] 9.4 Create documentation generator

  - Implement methodology document generation
  - Add example cases for each anomaly type
  - Create README with interpretation instructions
  - _Requirements: 9.1, 9.3, 9.5_

- [ ] 10. Implement main execution script
- [x] 10.1 Create main.py orchestration

  - Load configuration from config.yaml

  - Initialize all components (loader, detectors, aggregator, exporter)
  - Execute full pipeline: load → detect → aggregate → export
  - Add progress logging for each major step
  - Validate output files are created successfully
  - _Requirements: 10.4, 10.5_

- [x] 10.2 Add error handling and logging


  - Wrap each pipeline step in try-except blocks
  - Log errors with context and continue with available data
  - Create detailed execution log file
  - Add summary statistics at the end of execution
  - _Requirements: 10.4_

- [ ] 11. Create test suite
- [x] 11.1 Write unit tests for data loader

  - Test loading each parquet file type
  - Test handling of missing files
  - Test data validation logic
  - Test merge operations
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 11.2 Write unit tests for detectors

  - Create synthetic test data with known anomalies

  - Test each detection method independently
  - Verify severity score calculations
  - Test edge cases (empty data, all same values)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 4.1, 4.2, 5.1, 5.2, 6.1, 6.2_

- [ ] 11.3 Write integration tests




  - Test full pipeline execution on test dataset
  - Verify output file structure and content
  - Test performance on full dataset
  - _Requirements: 10.5_

- [ ] 12. Final validation and documentation
- [x] 12.1 Run full analysis on actual data






  - Execute main.py on all СберИндекс and Росстат files
  - Verify all output files are created in output/ directory
  - Check CSV and Excel files open correctly
  - Validate visualizations are generated
  - Review log file for any errors or warnings
  - _Requirements: 8.5, 10.5_

- [x] 12.2 Create project README





  - Document installation instructions
  - Explain how to run the analysis
  - Describe output files and their structure
  - Add examples of interpreting results
  - Include configuration options
  - _Requirements: 9.5_
