# Technical Stack

## Language & Runtime

- Python 3.8+
- No build system required (interpreted language)

## Core Dependencies

- **pandas>=2.0.0** - Data processing and manipulation
- **numpy>=1.24.0** - Numerical computations
- **scipy>=1.10.0** - Statistical methods
- **matplotlib>=3.7.0** - Visualization
- **seaborn>=0.12.0** - Enhanced visualization
- **openpyxl>=3.1.0** - Excel file handling
- **pyarrow>=14.0.0** - Parquet file reading
- **pyyaml>=6.0** - Configuration management

## Development Dependencies

- **pytest>=7.4.0** - Testing framework
- **pytest-cov>=4.1.0** - Code coverage
- **black>=23.0.0** - Code formatting
- **flake8>=6.0.0** - Linting
- **mypy>=1.4.0** - Type checking

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python main.py
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

- Configuration file: `config.yaml` (YAML format)
- Supports multiple detection profiles (strict/normal/relaxed)
- Auto-tuning configuration for threshold optimization
- Validation performed on startup via `config_validator.py`
