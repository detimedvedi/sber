# Project Structure

## Directory Layout

```
sberindex-anomaly-detection/
├── src/                    # Core application code
├── tests/                  # Test suite
├── examples/               # Usage examples and demos
├── docs/                   # Documentation and implementation summaries
├── output/                 # Generated results (auto-created)
├── rosstat/                # Rosstat data files (Parquet)
├── t_dict_municipal/       # Municipal dictionary (Excel, GeoPackage)
├── config.yaml             # Main configuration
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Source Code Organization (`src/`)

### Core Modules

- **data_loader.py** - Loads and merges data from all sources (СберИндекс, Росстат, municipal dict)
- **anomaly_detector.py** - Base detector class and all detector implementations
- **detector_manager.py** - Orchestrates detector execution with error handling
- **data_preprocessor.py** - Data preprocessing and municipality classification
- **results_aggregator.py** - Aggregates and ranks anomaly results
- **exporter.py** - Exports results to CSV, Excel, visualizations
- **auto_tuner.py** - Automatic threshold optimization
- **config_validator.py** - Configuration validation
- **error_handler.py** - Centralized error handling
- **legitimate_pattern_filter.py** - Filters known legitimate patterns

### Detector Classes (in anomaly_detector.py)

- **BaseAnomalyDetector** - Abstract base class for all detectors
- **StatisticalOutlierDetector** - Z-score, IQR, percentile methods
- **TemporalAnomalyDetector** - Spikes, drops, volatility detection
- **GeographicAnomalyDetector** - Regional and connection-based analysis
- **CrossSourceComparator** - Compares СберИндекс vs Rosstat (disabled by default)
- **LogicalConsistencyChecker** - Validates data consistency

## Test Organization (`tests/`)

Tests mirror the source structure with `test_*.py` files:
- Unit tests for individual components
- Integration tests for full pipeline
- Detector-specific tests
- Configuration and validation tests

## Documentation (`docs/`)

- Implementation summaries for each task (task_*.md)
- Methodology documentation (missing_value_methodology.md, enhanced_detection_methodology.md)
- Configuration guides (config_migration_guide.md)
- Quick reference guides

## Examples (`examples/`)

Demonstration scripts for key features:
- Auto-tuning workflows
- Configuration management
- Profile loading
- Threshold validation

## Architecture Patterns

### Pipeline Pattern
`main.py` orchestrates a sequential pipeline:
1. Load configuration
2. Load and merge data
3. Run detectors (via DetectorManager)
4. Aggregate results
5. Export outputs

### Detector Pattern
All detectors inherit from `BaseAnomalyDetector` and implement:
- `detect(df)` - Main detection logic
- Return standardized anomaly records

### Error Handling
- Graceful degradation - one detector failure doesn't stop others
- Centralized error handler with context tracking
- Structured logging with extra fields

### Configuration Management
- YAML-based configuration
- Profile system (strict/normal/relaxed/custom_russia)
- Threshold manager for runtime adjustments
- Validation on load

## Data Flow

```
Parquet/Excel Files → DataLoader → Unified DataFrame
                                         ↓
                                  DetectorManager
                                         ↓
                    [Statistical, Temporal, Geographic, Logical]
                                         ↓
                                  ResultsAggregator
                                         ↓
                                    Exporter
                                         ↓
                        [CSV, Excel, PNG, Markdown]
```

## Naming Conventions

- **Files**: snake_case (e.g., `data_loader.py`)
- **Classes**: PascalCase (e.g., `DataLoader`, `StatisticalOutlierDetector`)
- **Functions/Methods**: snake_case (e.g., `load_data`, `detect_outliers`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `REQUIRED_THRESHOLDS`)
- **Config keys**: snake_case (e.g., `z_score`, `detection_profile`)

## Key Design Decisions

- **Temporal data preservation**: Date columns maintained throughout pipeline for trend analysis
- **Connection graph**: Real territorial connections (4.7M) used instead of administrative boundaries
- **Robust statistics**: Median/MAD preferred over mean/std for outlier resistance
- **Detector independence**: Each detector runs in isolation with error handling
- **Profile-based thresholds**: Easy switching between detection modes
- **Source mapping**: Explicit tracking of data source for each column
