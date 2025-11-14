# Technology Stack

## Language & Runtime

- **Python 3.8+** - Primary language
- **pip** - Package management

## Core Dependencies

### Data Processing
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `pyarrow>=14.0.0` - Parquet file reading
- `openpyxl>=3.1.0` - Excel file handling

### Statistical Analysis
- `scipy>=1.10.0` - Statistical methods (z-score, percentiles, distributions)

### Visualization
- `matplotlib>=3.7.0` - Plotting and charts
- `seaborn>=0.12.0` - Statistical visualizations

### Configuration
- `pyyaml>=6.0` - YAML config file parsing

### Development
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Test coverage
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.4.0` - Type checking

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Running
```bash
# Run full analysis pipeline
python main.py

# Run specific analysis scripts
python analyze_anomalies.py
python prepare_for_visualization.py
python run_and_compare.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration

- **config.yaml** - Main system configuration (thresholds, profiles, paths)
- **legitimate_patterns_config.yaml** - Legitimate pattern filters (optional)

## Data Formats

- **Input**: Parquet files (СберИндекс, Росстат), Excel (municipal dictionary)
- **Output**: CSV (master data), Excel (multi-sheet summaries), PNG (visualizations), Markdown (documentation)

## Logging

- Structured logging with context fields
- Log file: `output/anomaly_detection.log`
- Console output with progress indicators
