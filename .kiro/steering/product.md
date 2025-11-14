# Product Overview

СберИндекс Anomaly Detection System - a Python application for detecting statistical anomalies in municipal data from СберИндекс and Росстат (Russian Federal State Statistics Service).

## Purpose

Competition entry for СберИндекс: "О этот странный, странный мир" (About this strange, strange world). The system identifies unusual patterns and outliers in Russian municipal data by comparing СберИндекс metrics with official Rosstat statistics.

## Core Capabilities

- **Statistical Analysis**: Detects outliers using z-score, IQR, and percentile methods
- **Temporal Analysis**: Identifies spikes, drops, and volatility in time series data
- **Geographic Analysis**: Finds municipalities that deviate from regional patterns using real territorial connection graphs
- **Cross-Source Comparison**: Compares СберИндекс and Rosstat data (disabled by default due to methodology differences)
- **Logical Consistency**: Detects contradictions and impossible values
- **Automated Reporting**: Generates CSV, Excel, visualizations, and documentation

## Key Features

- Robust statistics using median/MAD for outlier resistance
- Municipality classification (capital/urban/rural) with adaptive thresholds
- Configurable detection profiles (strict/normal/relaxed/custom_russia)
- Temporal data preservation for trend analysis
- Connection graph-based geographic analysis (4.7M territorial connections)
- Legitimate pattern filtering (tourism, business districts, remote territories)
- Executive summaries in Russian for management

## Data Sources

- **СберИндекс**: connection, consumption, market_access (Parquet)
- **Росстат**: population, migration, salary (Parquet)
- **Municipal Dictionary**: territory metadata (Excel)
