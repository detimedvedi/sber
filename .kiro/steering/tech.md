# Technology Stack

## Language & Runtime

- **Python 3.8+** (required)
- Virtual environment recommended for dependency isolation

## Core Dependencies

```
pandas>=2.0.0          # Data manipulation and analysis
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Statistical methods
pyarrow>=14.0.0        # Parquet file reading
pyyaml>=6.0            # YAML configuration parsing
openpyxl>=3.1.0        # Excel file handling
matplotlib>=3.7.0      # Plotting and visualization
seaborn>=0.12.0        # Statistical visualizations
```

## Development Dependencies

```
pytest>=7.4.0          # Testing framework
pytest-cov>=4.1.0      # Test coverage
black>=23.0.0          # Code formatting
flake8>=6.0.0          # Linting
mypy>=1.4.0            # Type checking
```

## Data Formats

- **Input**: Parquet files (СберИндекс, Rosstat), Excel (municipal dictionary)
- **Output**: CSV, Excel (XLSX with multiple sheets), PNG (visualizations), Markdown (documentation)

## Configuration

- **config.yaml**: Main configuration file with thresholds, profiles, and settings
- **legitimate_patterns_config.yaml**: Optional filter for known patterns
- Supports multiple detection profiles: strict, normal, relaxed, custom_russia

## Common Commands

### Installation
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis
```bash
# Full analysis with default config
python main.py

# Analyze results
python analyze_anomalies.py

# Run and compare configurations
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

## Architecture Notes

- **Modular detector system**: Each detector type (statistical, temporal, geographic, etc.) is a separate class inheriting from `BaseAnomalyDetector`
- **Error handling**: Centralized error handler with graceful degradation - failure of one detector doesn't stop others
- **Configuration-driven**: All thresholds and behavior controlled via YAML config
- **Detector management**: `DetectorManager` orchestrates all detectors with statistics tracking
- **Profile system**: Pre-configured threshold profiles for different detection modes
