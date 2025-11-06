# Requirements Document

## Introduction

Данный документ описывает требования к улучшению системы обнаружения аномалий СберИндекс. Улучшения направлены на исправление критических ошибок, повышение качества детекции, уменьшение ложных срабатываний и автоматизацию настройки системы.

Проект разделен на 4 фазы для поэтапной реализации улучшений.

## Glossary

- **System**: Система обнаружения аномалий СберИндекс
- **Detector**: Модуль детектора аномалий (StatisticalOutlierDetector, GeographicAnomalyDetector и т.д.)
- **False Positive**: Ложное срабатывание - естественная вариация, ошибочно классифицированная как аномалия
- **Threshold**: Пороговое значение для определения аномалии
- **Auto-tuning**: Автоматический подбор оптимальных пороговых значений
- **Temporal Data**: Временные ряды данных с несколькими наблюдениями для одного муниципалитета
- **Severity Score**: Оценка серьезности аномалии (0-100)
- **Municipality Type**: Тип муниципалитета (городской/сельский)
- **Management Report**: Отчет для менеджмента с упрощенными описаниями

---

## Requirements

### Requirement 1: Исправление Критических Ошибок (Фаза 1)

**User Story:** Как аналитик данных, я хочу, чтобы все детекторы работали без ошибок, чтобы получать полный набор обнаруженных аномалий

#### Acceptance Criteria

1. WHEN THE System executes StatisticalOutlierDetector, THE System SHALL complete detection without KeyError exceptions
2. WHEN THE System processes data with duplicate territory_id values, THE System SHALL identify whether duplicates represent temporal data or data quality issues
3. WHEN THE System encounters duplicate territory_id values, THE System SHALL log a warning with the count and affected territories
4. WHEN THE System detects temporal data structure, THE System SHALL either aggregate data by territory or enable temporal analysis
5. WHERE temporal analysis is not required, THE System SHALL aggregate duplicate records using the most recent period or mean values

---

### Requirement 2: Анализ и Обработка Временных Данных (Фаза 1)

**User Story:** Как аналитик данных, я хочу понимать структуру временных данных в датасете, чтобы правильно их обрабатывать

#### Acceptance Criteria

1. WHEN THE System loads data, THE System SHALL analyze the presence of temporal dimensions (date, period, year, month columns)
2. WHEN THE System detects multiple records per territory_id, THE System SHALL determine if they represent different time periods
3. WHEN temporal structure is confirmed, THE System SHALL add metadata indicating temporal granularity (daily, monthly, quarterly, yearly)
4. WHERE temporal data exists, THE System SHALL provide configuration option to enable or disable temporal analysis
5. WHEN temporal analysis is disabled, THE System SHALL aggregate temporal data to single record per territory using configurable aggregation method (latest, mean, median)

---

### Requirement 3: Уменьшение Ложных Географических Аномалий (Фаза 2)

**User Story:** Как менеджер, я хочу видеть только значимые географические аномалии, чтобы не тратить время на естественные различия между городами и селами

#### Acceptance Criteria

1. WHEN THE GeographicAnomalyDetector analyzes municipalities, THE System SHALL classify each municipality by type (urban, rural, capital)
2. WHEN THE System compares municipalities, THE System SHALL only compare municipalities of the same type
3. WHEN THE System detects capital cities or major urban centers, THE System SHALL apply relaxed thresholds (higher z-score requirements)
4. WHEN THE System calculates regional outliers, THE System SHALL use robust statistics (median, MAD) instead of mean and standard deviation
5. WHERE a municipality is flagged as geographic anomaly, THE System SHALL verify the anomaly is not due to natural urban-rural differences

---

### Requirement 4: Улучшение Описаний Аномалий для Менеджмента (Фаза 2)

**User Story:** Как менеджер, я хочу получать понятные описания аномалий на русском языке с контекстом, чтобы быстро принимать решения

#### Acceptance Criteria

1. WHEN THE System generates anomaly description, THE System SHALL use business-friendly language without technical jargon
2. WHEN THE System describes deviation, THE System SHALL express it in relative terms (в X раз выше/ниже среднего)
3. WHEN THE System creates anomaly record, THE System SHALL include comparison context (сравнение с соседними муниципалитетами)
4. WHEN THE System exports results, THE System SHALL generate executive summary with top 10 most critical anomalies
5. WHERE multiple related anomalies exist for one municipality, THE System SHALL group them and identify root cause

---

### Requirement 5: Приоритизация и Ранжирование Аномалий (Фаза 2)

**User Story:** Как менеджер, я хочу видеть аномалии, отсортированные по важности, чтобы сначала обрабатывать критичные случаи

#### Acceptance Criteria

1. WHEN THE System calculates severity score, THE System SHALL apply type-based weighting (logical inconsistencies weighted higher than geographic outliers)
2. WHEN THE System ranks anomalies, THE System SHALL consider multiple factors: severity score, anomaly type, affected indicators count
3. WHEN THE System detects multiple anomalies for one municipality, THE System SHALL calculate aggregate municipality risk score
4. WHEN THE System exports results, THE System SHALL sort anomalies by priority score in descending order
5. WHERE anomaly affects critical indicator (population, total consumption), THE System SHALL increase priority score by 20%

---

### Requirement 6: Автоматический Подбор Порогов (Фаза 3)

**User Story:** Как администратор системы, я хочу, чтобы система автоматически подбирала оптимальные пороги, чтобы минимизировать ложные срабатывания

#### Acceptance Criteria

1. WHEN THE System starts auto-tuning process, THE System SHALL analyze historical anomaly detection results
2. WHEN THE System calculates optimal thresholds, THE System SHALL use statistical methods to minimize false positive rate
3. WHEN THE System determines threshold values, THE System SHALL ensure at least 95% of normal municipalities are not flagged
4. WHEN THE System completes auto-tuning, THE System SHALL generate recommended threshold configuration file
5. WHERE auto-tuning is enabled, THE System SHALL periodically re-evaluate thresholds based on new data (configurable interval)

---

### Requirement 7: Робастные Статистические Методы (Фаза 2)

**User Story:** Как аналитик данных, я хочу, чтобы статистические расчеты были устойчивы к выбросам, чтобы экстремальные значения не искажали результаты

#### Acceptance Criteria

1. WHEN THE System calculates central tendency, THE System SHALL use median instead of mean for skewed distributions
2. WHEN THE System calculates dispersion, THE System SHALL use MAD (Median Absolute Deviation) or IQR instead of standard deviation
3. WHEN THE System detects highly skewed indicator (skewness > 2), THE System SHALL apply log transformation before analysis
4. WHEN THE System compares values across municipalities, THE System SHALL normalize indicators to per-capita or percentage metrics where applicable
5. WHERE extreme outliers exist (>5 standard deviations), THE System SHALL apply winsorization to limit their influence on statistics

---

### Requirement 8: Улучшенное Определение Источника Данных (Фаза 1)

**User Story:** Как аналитик данных, я хочу точно знать источник каждого показателя, чтобы правильно интерпретировать аномалии

#### Acceptance Criteria

1. WHEN THE System loads data, THE System SHALL create explicit mapping between column names and data sources
2. WHEN THE System determines data source, THE System SHALL use column prefixes (consumption_, salary_, population_) as primary method
3. WHEN THE System encounters ambiguous column name, THE System SHALL log warning and use fallback heuristics
4. WHEN THE System exports anomalies, THE System SHALL include data source for each indicator in metadata
5. WHERE column naming convention is inconsistent, THE System SHALL provide configuration option for custom source mapping

---

### Requirement 9: Конфигурационные Профили (Фаза 3)

**User Story:** Как администратор системы, я хочу использовать предустановленные профили настроек, чтобы быстро переключаться между строгим и мягким режимами детекции

#### Acceptance Criteria

1. WHEN THE System initializes, THE System SHALL support multiple configuration profiles (strict, normal, relaxed)
2. WHEN strict profile is selected, THE System SHALL use lower thresholds to detect more anomalies
3. WHEN relaxed profile is selected, THE System SHALL use higher thresholds to reduce false positives
4. WHEN THE System loads profile, THE System SHALL validate all required parameters are present
5. WHERE custom profile is provided, THE System SHALL merge it with default profile for missing parameters

---

### Requirement 10: Расширенная Валидация Результатов (Фаза 2)

**User Story:** Как аналитик данных, я хочу получать метрики качества детекции, чтобы оценивать эффективность системы

#### Acceptance Criteria

1. WHEN THE System completes detection, THE System SHALL calculate and report total anomalies by type and severity
2. WHEN THE System generates report, THE System SHALL include percentage of municipalities affected
3. WHEN THE System detects anomalies, THE System SHALL calculate anomaly rate per 1000 municipalities for each detector
4. WHEN THE System exports results, THE System SHALL include data quality metrics (completeness, consistency scores)
5. WHERE anomaly count exceeds expected range, THE System SHALL log warning suggesting threshold adjustment

---

### Requirement 11: Улучшенная Обработка Пропущенных Значений (Фаза 2)

**User Story:** Как аналитик данных, я хочу, чтобы система корректно обрабатывала пропущенные значения, чтобы они не искажали результаты

#### Acceptance Criteria

1. WHEN THE System encounters missing values, THE System SHALL calculate missingness percentage per indicator
2. WHEN indicator has >50% missing values, THE System SHALL skip it from analysis and log warning
3. WHEN municipality has >70% missing indicators, THE System SHALL flag it as data quality issue
4. WHEN THE System calculates statistics, THE System SHALL use only non-missing values (pairwise deletion)
5. WHERE missing data pattern is unusual, THE LogicalConsistencyChecker SHALL detect it as anomaly

---

### Requirement 12: Экспорт для Менеджмента (Фаза 2)

**User Story:** Как менеджер, я хочу получать краткий executive summary в удобном формате, чтобы быстро понимать ситуацию

#### Acceptance Criteria

1. WHEN THE System exports results, THE System SHALL generate separate executive summary sheet in Excel
2. WHEN THE System creates summary, THE System SHALL include top 10 municipalities by risk score
3. WHEN THE System generates summary, THE System SHALL include key findings in bullet points (на русском языке)
4. WHEN THE System creates visualizations, THE System SHALL generate dashboard-style summary image
5. WHERE critical anomalies exist (severity > 90), THE System SHALL highlight them in red in summary

---

### Requirement 13: Логирование и Диагностика (Фаза 1)

**User Story:** Как разработчик, я хочу получать подробные логи выполнения, чтобы быстро диагностировать проблемы

#### Acceptance Criteria

1. WHEN THE System encounters error, THE System SHALL log full stack trace with context information
2. WHEN detector fails, THE System SHALL continue with other detectors and log failure details
3. WHEN THE System completes execution, THE System SHALL log summary statistics for each detector
4. WHEN THE System processes data, THE System SHALL log data quality warnings (duplicates, missing values, outliers)
5. WHERE performance is degraded, THE System SHALL log execution time for each major step

---

### Requirement 14: Обратная Совместимость (Все фазы)

**User Story:** Как пользователь системы, я хочу, чтобы улучшения не ломали существующие интеграции, чтобы продолжать использовать текущие скрипты

#### Acceptance Criteria

1. WHEN THE System exports results, THE System SHALL maintain current CSV and Excel file structure
2. WHEN new fields are added, THE System SHALL append them to existing structure without removing old fields
3. WHEN configuration format changes, THE System SHALL support both old and new formats with automatic migration
4. WHEN THE System introduces new features, THE System SHALL make them opt-in via configuration flags
5. WHERE breaking changes are necessary, THE System SHALL provide migration guide and deprecation warnings

---

### Requirement 15: Тестирование Улучшений (Все фазы)

**User Story:** Как разработчик, я хочу иметь автоматические тесты для всех улучшений, чтобы гарантировать их корректную работу

#### Acceptance Criteria

1. WHEN new detector logic is implemented, THE System SHALL include unit tests covering edge cases
2. WHEN bug fix is applied, THE System SHALL include regression test preventing bug recurrence
3. WHEN auto-tuning is implemented, THE System SHALL include tests validating threshold optimization
4. WHEN new aggregation method is added, THE System SHALL include tests comparing results with expected values
5. WHERE integration tests exist, THE System SHALL update them to cover new functionality
