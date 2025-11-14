# Implementation Plan

- [x] 1. Fix consumption data temporal structure preservation
  - Modify `DataLoader.merge_datasets()` to preserve date column when pivoting consumption data
  - Add conditional logic to detect presence of date column
  - Use `pivot()` instead of `pivot_table()` with aggfunc when temporal data exists
  - Add column renaming with 'consumption_' prefix
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement connection graph loading and integration


  - [x] 2.1 Add `load_connection_data()` method to DataLoader





    - Create method to load connection.parquet file
    - Validate required columns (territory_id_x, territory_id_y, distance)
    - Return empty DataFrame on error for graceful degradation
    - Log connection statistics (count, unique territories, types)
    - _Requirements: 2.1_

  - [x] 2.2 Add neighbor detection to GeographicAnomalyDetector





    - Implement `_get_neighbors()` method using connection graph
    - Query connections where territory appears in either column
    - Filter by configurable distance threshold (default 50km)
    - Deduplicate and return neighbor territory IDs
    - _Requirements: 2.2, 2.5_

  - [x] 2.3 Modify `detect_cluster_outliers()` to use connection graph




    - Add connections parameter to method signature
    - Handle temporal data by using latest period for geographic analysis
    - Implement graph-based neighbor comparison logic
    - Add fallback to region-based clustering when connections unavailable
    - Enforce minimum 3 neighbors requirement
    - _Requirements: 2.2, 2.3, 2.4_

  - [x] 2.4 Update main.py to load and pass connection graph



    - Call `loader.load_connection_data()` after loading other data
    - Pass connections DataFrame to `detector_manager.run_all_detectors()`
    - Log connection graph availability
    - _Requirements: 2.1_

- [ ] 3. Disable CrossSourceComparator detector
  - [x] 3.1 Add detectors configuration section to config.yaml





    - Create detectors section with enabled flags for each detector type
    - Set cross_source.enabled to false by default
    - Set other detectors to enabled: true
    - _Requirements: 3.1, 3.3_

  - [x] 3.2 Modify DetectorManager to conditionally load detectors





    - Update `_initialize_detectors()` to check config.detectors section
    - Only initialize detectors where enabled flag is true
    - Log warning when CrossSourceComparator is enabled
    - Log info when detectors are disabled
    - _Requirements: 3.2, 3.4_

  - [x] 3.3 Update DetectorManager to pass connections to detectors





    - Add connections parameter to `run_all_detectors()` method
    - Add connections parameter to `run_detector_safe()` method
    - Pass connections to geographic detector when calling detect()
    - _Requirements: 2.1_

- [x] 4. Create and apply Russia-specific threshold profile




  - [x] 4.1 Add custom_russia profile to config.yaml


    - Create threshold_profiles.custom_russia section
    - Set statistical.z_score to 5.0 (increased from 3.0)
    - Set statistical.iqr_multiplier to 3.0 (increased from 1.5)
    - Set statistical.percentile_lower to 0.1 and percentile_upper to 99.9
    - Set geographic.regional_z_score to 6.0 (increased from 3.5)
    - Set geographic.cluster_threshold to 5.0 (increased from 4.0)
    - Set temporal.spike_threshold to 200 (increased from 100)
    - Set temporal.drop_threshold to -80 (decreased from -50)
    - Set logical.check_impossible_ratios to false
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 4.2 Set custom_russia as active detection_profile


    - Update detection_profile setting in config.yaml to "custom_russia"
    - _Requirements: 8.2_

- [ ] 5. Implement log transform for skewed distributions
  - [x] 5.1 Add skewness detection to StatisticalOutlierDetector





    - Calculate skewness coefficient for each indicator in `detect_zscore_outliers()`
    - Check if absolute skewness exceeds 2.0 threshold
    - Log debug message when skewness is high
    - _Requirements: 5.1, 5.4_

  - [x] 5.2 Apply log transformation for highly skewed indicators





    - Use np.log1p() for log(1+x) transformation when skewness > 2.0
    - Clip negative values to zero before transformation
    - Calculate z-scores on transformed values
    - Use original values for z-scores when skewness <= 2.0
    - _Requirements: 5.2, 5.3, 5.5_

  - [x] 5.3 Add temporal data handling to statistical detectors





    - Check for 'date' column in DataFrame
    - Use groupby('territory_id').last() to get latest period when temporal data exists
    - Apply statistical analysis on latest period data
    - _Requirements: 1.4_

- [x] 6. Disable auto-tuning without ground truth





  - Update auto_tuning.enabled to false in config.yaml
  - Add comment explaining why auto-tuning is disabled
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 7. Integrate legitimate pattern filter
  - [x] 7.1 Add filter application in main.py after aggregation





    - Import LegitimatePatternFilter after anomaly aggregation
    - Wrap filter application in try-except for graceful handling
    - Call filter_anomalies() on combined anomalies DataFrame
    - Count and log reclassified anomalies
    - Remove anomalies flagged as legitimate patterns
    - Handle ImportError and other exceptions gracefully
    - _Requirements: 7.2, 7.4, 7.5_

  - [ ]* 7.2 Expand legitimate_patterns_config.yaml with new categories
    - Add business_districts category with territories
    - Add university_cities category with territories
    - Add auto_whitelist_rules section with thresholds
    - _Requirements: 7.1, 7.3_

- [ ] 8. Update documentation
  - [x] 8.1 Update README.md with new features






    - Document connection graph usage
    - Explain custom_russia profile
    - Describe detector enable/disable flags
    - Add troubleshooting section for missing files
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 8.2 Add inline code comments
    - Comment temporal structure preservation logic in DataLoader
    - Comment connection graph query logic in GeographicAnomalyDetector
    - Comment log transform logic in StatisticalOutlierDetector
    - Comment conditional detector loading in DetectorManager

- [ ]-9. Validate implementation and metrics

  - [ ] 9.1 Run full pipeline and verify anomaly count reduction


    - Execute main.py with all fixes applied
    - Verify total anomalies between 3,000-5,000
    - Verify flagged territories between 20-35%
    - Compare with baseline of 16,682 anomalies
    - Confirm at least 70% reduction achieved
    - _Requirements: 9.1, 9.2, 9.5_

  - [ ]* 9.2 Verify temporal detection is working
    - Check that TemporalAnomalyDetector finds at least 1 anomaly
    - Verify temporal anomalies appear in output
    - Confirm date column preserved in consumption data
    - _Requirements: 1.3, 9.3_

  - [ ]* 9.3 Verify CrossSourceComparator is disabled
    - Check that CrossSourceComparator is not initialized
    - Verify 0 cross-source anomalies in output
    - Confirm total anomaly count reduced by ~2,414
    - _Requirements: 3.2, 9.4_

  - [ ]* 9.4 Verify connection graph is being used
    - Check logs for "Loaded connection graph" message
    - Verify GeographicAnomalyDetector uses graph-based neighbors
    - Confirm no "Connection file not found" warnings
    - _Requirements: 2.1, 2.2_

  - [ ]* 9.5 Verify legitimate pattern filter is working
    - Check logs for "Reclassified X anomalies as legitimate patterns"
    - Verify anomaly count reduced after filtering
    - Confirm legitimate patterns removed from output
    - _Requirements: 7.2, 7.4, 7.5_
