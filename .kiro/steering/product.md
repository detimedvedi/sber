# Product Overview

СберИндекс Anomaly Detection System - A Python application for comprehensive analysis of СберИндекс open data to identify unusual patterns and deviations in municipal data.

## Purpose

Competition entry for СберИндекс: "О этот странный, странный мир" (About This Strange, Strange World). The system compares СберИндекс indicators with official Rosstat statistics and generates structured reports for visualization and analysis.

## Core Capabilities

- **Statistical Analysis**: Outlier detection using z-score, IQR, and percentile methods
- **Temporal Analysis**: Detection of spikes, drops, and anomalous volatility in time series
- **Geographic Analysis**: Identification of municipalities that differ from regional averages using real territorial connection graphs
- **Cross-Source Comparison**: Comparison of СберИндекс and Rosstat data (currently disabled - no valid overlapping metrics)
- **Logical Consistency**: Detection of contradictions and impossible values
- **Automated Reporting**: CSV, Excel with multiple sheets, visualizations, and documentation

## Key Features (2025)

- **Territorial Connection Graph**: Uses 4.7M real connections between territories for precise geographic analysis instead of administrative boundaries
- **Temporal Structure Preservation**: Maintains complete time series (303K+ records) for trend and seasonality detection
- **Custom Russia Profile**: Specialized thresholds for Russian geographic heterogeneity (Moscow vs Chukotka)
- **Detector Management**: Flexible enable/disable of individual detectors via configuration
- **Asymmetric Distribution Handling**: Automatic log-transformation for highly skewed indicators
- **Legitimate Pattern Filtering**: Exclusion of expected variations (tourism, business districts, remote territories)

## Data Sources

- **СберИндекс**: connection.parquet, consumption.parquet, market_access.parquet
- **Rosstat**: population, migration, salary data (parquet format)
- **Municipal Dictionary**: t_dict_municipal_districts.xlsx

## Output

Analysis completes in ~1-2 minutes, generating comprehensive reports in the `output/` directory including master CSV, summary Excel with multiple sheets, visualizations (PNG), and markdown documentation.
