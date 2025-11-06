# Project Structure

## Directory Layout

```
sberindex-anomaly-detection/
├── src/                          # Source code modules
├── tests/                        # Test suite
├── examples/                     # Usage examples and demos
├── docs/                         # Documentation
├── output/                       # Generated results (auto-created)
├── rosstat/                      # Росстат data files (Parquet)
├── t_dict_municipal/             # Municipal dictionary
├── config.yaml                   # Main configuration
├── main.py                       # Entry point
└── requirements.txt              # Python dependencies
```

## Source Code Organization (`src/`)

### Core Modules

- **data_loader.py** - Data loading, merging, temporal analysis, duplicate detection
- **anomaly_detector.py** - Base detector class and all detector implementations
- **detector_manager.py** - Centralized detector orchestration with error handling
- **data_preprocessor.py** - Data preprocessing and municipality classification
- **results_aggregator.py** - Anomaly aggregation, prioritization, and grouping
- **exporter.py** - Results export (CSV, Excel, visualizations, reports)

### Supporting Modules

- **auto_tuner.py** - Automatic threshold optimization
- **config_validator.py** - Configuration validation
- **error_handler.py** - Centralized error handling with graceful degradation

## Test Organization (`tests/`)

Tests follow naming convention `test_<module>.py` and mirror source structure:
- Unit tests for individual modules
- Integration tests for full pipeline
- Specific feature tests (auto-tuning, profiles, error handling)

## Examples (`examples/`)

Demonstration scripts showing:
- Auto-tuning workflows
- Configuration migration
- Profile management
- Threshold validation
- FPR calculation

## Documentation (`docs/`)

- Implementation summaries (task_*.md)
- Usage guides (methodology, filtering, analysis)
- Quick reference guides
- Migration guides

## Data Files

### Input Data (Parquet format)
- `connection.parquet`, `consumption.parquet`, `market_access.parquet` - СберИндекс data
- `rosstat/*.parquet` - Росстат data (population, migration, salary)
- `t_dict_municipal/*.xlsx` - Municipal dictionary

### Output Files (auto-generated in `output/`)
- `anomalies_master_*.csv` - Complete anomaly list
- `anomalies_summary_*.xlsx` - Multi-sheet Excel report
- `viz_*.png` - Visualizations
- `*.md` - Documentation and reports
- `anomaly_detection.log` - Execution log

## Architecture Patterns

### Detector Pattern
All detectors inherit from `BaseAnomalyDetector` abstract class and implement `detect()` method. Common functionality (severity scoring, record creation, source mapping) is provided by base class.

### Manager Pattern
`DetectorManager` orchestrates all detectors with error handling, ensuring one detector failure doesn't stop others. Tracks statistics and manages threshold profiles.

### Error Handling
Centralized error handler (`error_handler.py`) provides graceful degradation - system continues with available data/detectors when errors occur.

### Configuration Management
- YAML-based configuration with validation
- Profile system for quick mode switching
- Auto-tuning support for threshold optimization
- Backward compatibility with old config format
