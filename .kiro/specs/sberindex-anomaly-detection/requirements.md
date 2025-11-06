# Requirements Document

## Introduction

Данный проект направлен на анализ открытых данных СберИндекса с целью выявления статистических аномалий и отклонений. Система должна обнаруживать необычные паттерны в муниципальных данных, сравнивать их с официальной статистикой Росстата и подготавливать структурированные результаты для команды визуализации. Проект реализуется как разовый анализ с использованием Python и выдачей результатов в формате CSV/Excel.

## Glossary

- **Система анализа** (Analysis System): Python-приложение для обнаружения аномалий в данных
- **СберИндекс**: Набор данных о муниципалитетах, включающий показатели подключений (connection), потребления (consumption) и доступа к рынкам (market_access)
- **Росстат**: Официальная статистика, включающая данные о населении, миграции и зарплатах
- **Аномалия**: Статистическое отклонение, значительно отличающееся от ожидаемых значений
- **Муниципалитет**: Административно-территориальная единица (город, район, поселение)
- **Итоговые данные** (Output Data): Структурированные CSV/Excel файлы с обнаруженными аномалиями

## Requirements

### Requirement 1

**User Story:** Как аналитик данных, я хочу загрузить и объединить все доступные данные СберИндекса и Росстата, чтобы иметь полный датасет для анализа аномалий

#### Acceptance Criteria

1. THE Analysis System SHALL load all parquet files from the workspace (connection.parquet, consumption.parquet, market_access.parquet, rosstat/2_bdmo_population.parquet, rosstat/3_bdmo_migration.parquet, rosstat/4_bdmo_salary.parquet)

2. THE Analysis System SHALL merge datasets using municipal identifiers from t_dict_municipal directory

3. THE Analysis System SHALL validate data completeness and log missing values for each dataset

4. THE Analysis System SHALL create a unified dataframe with all indicators aligned by municipality and time period

5. WHEN data loading fails for any file, THEN THE Analysis System SHALL log the error and continue with available datasets

### Requirement 2

**User Story:** Как аналитик данных, я хочу обнаруживать статистические выбросы в данных СберИндекса, чтобы найти муниципалитеты с необычными показателями

#### Acceptance Criteria

1. THE Analysis System SHALL calculate z-scores for all numerical indicators in СберИндекс datasets

2. THE Analysis System SHALL identify outliers where z-score exceeds 3 standard deviations from the mean

3. THE Analysis System SHALL apply IQR (Interquartile Range) method to detect outliers for each indicator

4. THE Analysis System SHALL flag municipalities with values in the top 1% or bottom 1% percentile for each metric

5. THE Analysis System SHALL store detected outliers with municipality name, indicator name, actual value, expected range, and deviation magnitude

### Requirement 3

**User Story:** Как аналитик данных, я хочу сравнивать показатели СберИндекса с данными Росстата, чтобы выявить расхождения между источниками

#### Acceptance Criteria

1. THE Analysis System SHALL calculate correlation coefficients between comparable indicators from СберИндекс and Росстат

2. WHEN correlation between comparable indicators is below 0.5, THEN THE Analysis System SHALL flag this as a potential data quality issue

3. THE Analysis System SHALL identify municipalities where СберИндекс indicators deviate more than 50% from Росстат indicators

4. THE Analysis System SHALL calculate percentage differences between СберИндекс and Росстат for each municipality

5. THE Analysis System SHALL rank municipalities by the magnitude of discrepancies between data sources

### Requirement 4

**User Story:** Как аналитик данных, я хочу обнаруживать временные аномалии, чтобы найти резкие изменения показателей во времени

#### Acceptance Criteria

1. WHERE temporal data is available, THE Analysis System SHALL calculate period-over-period growth rates for each indicator

2. THE Analysis System SHALL identify sudden spikes where growth rate exceeds 100% or drops below -50% in a single period

3. THE Analysis System SHALL detect trend reversals where indicator direction changes significantly

4. THE Analysis System SHALL flag municipalities with volatility (standard deviation of changes) exceeding 2 times the median volatility

5. THE Analysis System SHALL identify seasonal anomalies by comparing values to historical patterns for the same period

### Requirement 5

**User Story:** Как аналитик данных, я хочу обнаруживать географические аномалии, чтобы найти муниципалитеты, отличающиеся от соседних территорий

#### Acceptance Criteria

1. WHERE geographic data is available, THE Analysis System SHALL identify municipalities that differ significantly from their regional average

2. THE Analysis System SHALL calculate regional z-scores for each indicator within federal districts or oblasts

3. THE Analysis System SHALL flag municipalities where indicators deviate more than 2 standard deviations from regional mean

4. THE Analysis System SHALL identify clusters of similar municipalities and detect outliers within each cluster

5. THE Analysis System SHALL compare urban vs rural municipalities separately to account for structural differences

### Requirement 6

**User Story:** Как аналитик данных, я хочу обнаруживать логические несоответствия, чтобы найти потенциальные ошибки в данных

#### Acceptance Criteria

1. THE Analysis System SHALL identify negative values where only positive values are logically possible

2. THE Analysis System SHALL detect impossible ratios (e.g., consumption exceeding total capacity)

3. THE Analysis System SHALL flag municipalities with missing data patterns that differ from typical patterns

4. THE Analysis System SHALL identify contradictory indicators (e.g., high consumption with low connection rates)

5. THE Analysis System SHALL detect duplicate or inconsistent municipality identifiers

### Requirement 7

**User Story:** Как аналитик данных, я хочу ранжировать и приоритизировать обнаруженные аномалии, чтобы сосредоточиться на наиболее значимых отклонениях

#### Acceptance Criteria

1. THE Analysis System SHALL assign severity scores to each anomaly based on deviation magnitude

2. THE Analysis System SHALL calculate an anomaly count for each municipality across all indicators

3. THE Analysis System SHALL rank municipalities by total anomaly score (sum of all severity scores)

4. THE Analysis System SHALL categorize anomalies by type (statistical outlier, temporal spike, geographic deviation, data quality issue, logical inconsistency)

5. THE Analysis System SHALL create a summary report showing top 50 most anomalous municipalities

### Requirement 8

**User Story:** Как аналитик данных, я хочу экспортировать результаты в CSV/Excel формате, чтобы передать готовые данные команде визуализации

#### Acceptance Criteria

1. THE Analysis System SHALL export a master anomalies table in CSV format with columns: municipality_id, municipality_name, indicator, anomaly_type, actual_value, expected_value, deviation, severity_score, data_source

2. THE Analysis System SHALL export a summary Excel file with multiple sheets: Overview, Statistical_Outliers, Temporal_Anomalies, Geographic_Anomalies, Data_Quality_Issues, Top_Anomalous_Municipalities

3. THE Analysis System SHALL include descriptive statistics for each anomaly type in the Excel summary

4. THE Analysis System SHALL add data dictionary sheet explaining all columns and anomaly types

5. THE Analysis System SHALL save all output files to an 'output' directory with timestamp in filename

### Requirement 9

**User Story:** Как аналитик данных, я хочу документировать методологию и находки, чтобы объяснить обнаруженные аномалии команде визуализации

#### Acceptance Criteria

1. THE Analysis System SHALL generate a methodology document describing all detection methods used

2. THE Analysis System SHALL create visualizations (saved as PNG files) showing distribution of anomalies by type and region

3. THE Analysis System SHALL include example cases for each anomaly type with potential explanations

4. THE Analysis System SHALL log all analysis steps with timestamps to a log file

5. THE Analysis System SHALL create a README file with instructions for interpreting the output files

### Requirement 10

**User Story:** Как аналитик данных, я хочу обеспечить воспроизводимость анализа, чтобы другие члены команды могли повторить результаты

#### Acceptance Criteria

1. THE Analysis System SHALL use fixed random seeds for any stochastic operations

2. THE Analysis System SHALL log all configuration parameters (thresholds, methods) used in the analysis

3. THE Analysis System SHALL include requirements.txt file with all Python dependencies and versions

4. THE Analysis System SHALL provide a main script that executes the entire analysis pipeline from start to finish

5. THE Analysis System SHALL validate that output files are created successfully before completing execution
