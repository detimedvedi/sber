# Product Overview

СберИндекс Anomaly Detection System - автоматическое обнаружение статистических аномалий в данных СберИндекса с использованием официальной статистики Росстата.

## Purpose

Python-приложение для комплексного анализа открытых данных СберИндекса с целью выявления необычных паттернов и отклонений в муниципальных данных. Система сравнивает показатели СберИндекса с официальной статистикой Росстата и генерирует структурированные отчеты.

## Core Capabilities

- Statistical analysis: Z-score, IQR, and percentile-based outlier detection
- Temporal analysis: Spike, drop, and volatility detection
- Geographic analysis: Regional comparison with municipality type awareness
- Cross-source comparison: СберИндекс vs Росстат data validation
- Logical consistency checking: Data quality and contradiction detection
- Auto-tuning: Automatic threshold optimization to minimize false positives
- Configuration profiles: Pre-configured detection modes (strict/normal/relaxed)

## Output

- CSV and Excel reports with multiple sheets
- Visualizations (PNG)
- Executive summaries in Russian for management
- Detailed methodology documentation
- Priority-ranked anomalies with root cause analysis
