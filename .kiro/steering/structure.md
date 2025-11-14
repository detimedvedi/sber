# Project Structure

## Root Directory

```
├── main.py                              # Main entry point - orchestrates full pipeline
├── analyze_anomalies.py                 # Post-analysis script for result exploration
├── run_and_compare.py                   # Configuration comparison utility
├── config.yaml                          # Main configuration file
├── legitimate_patterns_config.yaml      # Optional legitimate pattern filters
├── requirements.txt                     # Python dependencies
├── README.md                            # Project documentation
├── DATA_GUIDE.md                        # Data acquisition and format guide
├── data_structure_summary.md            # Data structure documentation
```

## Source Code (`src/`)

Core modules implementing the anomaly detection system:

```
src/
├── __init__.py
├── data_loader.py              # Data loading and merging (СберИндекс, Rosstat, municipal dict)
├── data_preprocessor.py        # Data preprocessing and municipality classification
├── anomaly_detector.py         # Base detector class + all detector implementations
├── detector_manager.py         # Detector orchestration with error handling
├── results_aggregator.py       # Anomaly aggregation and ranking
├── exporter.py                 # Result export (CSV, Excel, visualizations, docs)
├── auto_tuner.py               # Automatic threshold optimization
├── config_validator.py         # Configuration validation
├── error_handler.py            # Centralized error handling
└── legitimate_pattern_filter.py # Filter for known legitimate patterns
```

### Key Module Responsibilities

- **data_loader.py**: Loads parquet/Excel files, merges datasets by territory_id, detects temporal structure, handles duplicates
- **anomaly_detector.py**: Contains `BaseAnomalyDetector` abstract class and concrete implementations:
  - `StatisticalOutlierDetector` (z-score, IQR, percentile methods)
  - `TemporalAnomalyDetector` (spikes, drops, volatility)
  - `GeographicAnomalyDetector` (regional deviations, connection graph-based)
  - `CrossSourceComparator` (СберИндекс vs Rosstat - currently disabled)
  - `LogicalConsistencyChecker` (contradictions, impossible values)
- **detector_manager.py**: Manages detector lifecycle, applies configuration profiles, tracks statistics, handles errors gracefully
- **exporter.py**: Generates all output files with timestamps

## Tests (`tests/`)

Comprehensive test suite covering all components:

```
tests/
├── __init__.py
├── test_data_loader.py                    # Data loading and merging tests
├── test_detectors.py                      # Detector functionality tests
├── test_detector_conditional_loading.py   # Detector enable/disable tests
├── test_detector_manager_profiles.py      # Profile switching tests
├── test_auto_tuner_fpr.py                 # Auto-tuning tests
├── test_config_validator.py               # Configuration validation tests
├── test_error_handler.py                  # Error handling tests
├── test_exporter.py                       # Export functionality tests
├── test_full_pipeline_integration.py      # End-to-end pipeline tests
└── [additional test files]
```

## Documentation (`docs/`)

Detailed methodology and usage documentation:

```
docs/
├── missing_value_methodology.md           # Missing value handling approach
├── enhanced_detection_methodology.md      # Detection algorithm details
├── config_migration_guide.md              # Configuration migration guide
├── auto_tuning_quick_reference.md         # Auto-tuning usage guide
├── configuration_profiles_quick_reference.md
├── indicator_filtering_usage.md
├── missingness_analysis_usage.md
├── municipality_flagging_usage.md
└── archive/                               # Archived documentation
```

## Data Directories

```
rosstat/                        # Rosstat data files (not in repo)
├── 2_bdmo_population.parquet
├── 3_bdmo_migration.parquet
└── 4_bdmo_salary.parquet

t_dict_municipal/               # Municipal dictionary (not in repo)
├── t_dict_municipal_districts.xlsx
└── t_dict_municipal_districts_poly.gpkg

output/                         # Generated results (created automatically)
├── anomalies_master_*.csv
├── anomalies_summary_*.xlsx
├── viz_*.png
├── methodology_*.md
├── example_cases_*.md
└── anomaly_detection.log
```

## Architecture Patterns

### Inheritance Hierarchy
- All detectors inherit from `BaseAnomalyDetector` abstract base class
- Provides common methods: `calculate_severity_score()`, `create_anomaly_record()`, `get_data_source()`
- Each detector implements `detect(df: pd.DataFrame) -> pd.DataFrame`

### Error Handling
- Centralized error handler in `error_handler.py`
- Graceful degradation: detector failures don't stop pipeline
- Detailed error context logging with structured data

### Configuration Management
- YAML-based configuration with validation
- Profile system for threshold presets (strict/normal/relaxed/custom_russia)
- Runtime profile switching supported via `DetectorManager`

### Data Flow
1. `main.py` → loads config
2. `DataLoader` → loads and merges all data sources
3. `DetectorManager` → runs all enabled detectors
4. `ResultsAggregator` → combines and ranks anomalies
5. `ResultsExporter` → generates all output files

### Naming Conventions
- Snake_case for functions and variables
- PascalCase for classes
- Descriptive names: `detect_zscore_outliers()`, `StatisticalOutlierDetector`
- Prefix indicators with source: `consumption_*`, `salary_*`, `population_*`
