# Design Document

## Overview

Данный документ описывает техническую архитектуру улучшений системы обнаружения аномалий СберИндекс. Улучшения реализуются в 4 фазы, каждая из которых добавляет новую функциональность при сохранении обратной совместимости.

### Design Goals

1. **Надежность**: Исправить все критические ошибки и обеспечить стабильную работу всех детекторов
2. **Качество**: Снизить количество ложных срабатываний с 70% до 20-30%
3. **Удобство**: Предоставить понятные отчеты для менеджмента
4. **Автоматизация**: Минимизировать ручную настройку через auto-tuning
5. **Совместимость**: Сохранить все существующие интерфейсы и форматы данных

### Phased Approach

- **Phase 1**: Critical Fixes (1-2 недели)
- **Phase 2**: Quality Improvements (2-3 недели)
- **Phase 3**: Auto-tuning (1-2 недели)
- **Phase 4**: Continuous (тестирование и документация)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Pipeline                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Enhanced Data Loader                       │
│  • Temporal Analysis                                         │
│  • Duplicate Detection                                       │
│  • Source Mapping                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Preprocessor (NEW)                     │
│  • Aggregation Strategy                                      │
│  • Municipality Classification                               │
│  • Robust Statistics Calculation                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Detector Manager (NEW)                    │
│  • Threshold Management                                      │
│  • Detector Orchestration                                    │
│  • Error Handling                                            │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Fixed Detectors  │      │  Auto-Tuner      │
        │ • Statistical    │      │  (Phase 3)       │
        │ • Geographic     │      └──────────────────┘
        │ • Temporal       │
        │ • Cross-Source   │
        │ • Logical        │
        └──────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Enhanced Aggregator     │
        │  • Priority Scoring      │
        │  • Grouping              │
        │  • Root Cause Analysis   │
        └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │  Enhanced Exporter       │
        │  • Management Reports    │
        │  • Executive Summary     │
        │  • Improved Descriptions │
        └──────────────────────────┘
```

---

## Components and Interfaces

### 1. Enhanced Data Loader

**Purpose**: Расширение существующего DataLoader для анализа временной структуры и улучшенного маппинга источников

**New Methods**:
```python
class EnhancedDataLoader(DataLoader):
    def analyze_temporal_structure(self, df: pd.DataFrame) -> TemporalMetadata
    def detect_duplicates(self, df: pd.DataFrame) -> DuplicateReport
    def create_source_mapping(self, columns: List[str]) -> Dict[str, str]
    def aggregate_temporal_data(self, df: pd.DataFrame, method: str) -> pd.DataFrame
```

**TemporalMetadata**:
```python
@dataclass
class TemporalMetadata:
    has_temporal_data: bool
    temporal_columns: List[str]
    granularity: str  # 'daily', 'monthly', 'quarterly', 'yearly', 'unknown'
    periods_per_territory: Dict[int, int]
    date_range: Tuple[datetime, datetime]
```

**DuplicateReport**:
```python
@dataclass
class DuplicateReport:
    duplicate_count: int
    affected_territories: List[int]
    is_temporal: bool
    recommendation: str  # 'aggregate', 'enable_temporal_analysis', 'investigate'
```

**Integration**: Расширяет существующий `src/data_loader.py` без breaking changes

---

### 2. Data Preprocessor (NEW)

**Purpose**: Новый компонент для предобработки данных перед детекцией

**Class Structure**:
```python
class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.municipality_classifier = MunicipalityClassifier()
        self.robust_stats_calculator = RobustStatsCalculator()
    
    def preprocess(self, df: pd.DataFrame) -> PreprocessedData
    def classify_municipalities(self, df: pd.DataFrame) -> pd.DataFrame
    def calculate_robust_statistics(self, df: pd.DataFrame) -> Dict[str, RobustStats]
    def normalize_indicators(self, df: pd.DataFrame) -> pd.DataFrame
```

**MunicipalityClassifier**:
```python
class MunicipalityClassifier:
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'municipality_type' column with values:
        - 'capital': Regional capitals and federal cities
        - 'urban': Cities with population > 50,000
        - 'rural': Other municipalities
        """
```

**RobustStatsCalculator**:
```python
class RobustStatsCalculator:
    def calculate_for_indicator(self, values: pd.Series) -> RobustStats:
        """
        Returns:
        - median: Robust central tendency
        - mad: Median Absolute Deviation
        - iqr: Interquartile Range
        - percentiles: 1st, 5th, 25th, 75th, 95th, 99th
        - skewness: Distribution skewness
        """
```

**File Location**: `src/data_preprocessor.py` (new file)

---

### 3. Detector Manager (NEW)

**Purpose**: Управление детекторами, обработка ошибок, управление порогами

**Class Structure**:
```python
class DetectorManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold_manager = ThresholdManager(config)
        self.detectors = self._initialize_detectors()
    
    def run_all_detectors(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Run all detectors with error handling"""
    
    def run_detector_safe(self, detector: BaseAnomalyDetector, df: pd.DataFrame) -> pd.DataFrame:
        """Run single detector with try-catch"""
    
    def get_detector_statistics(self) -> Dict[str, DetectorStats]:
        """Get execution statistics for each detector"""
```

**ThresholdManager**:
```python
class ThresholdManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profile = config.get('detection_profile', 'normal')
    
    def get_thresholds(self, detector_name: str) -> Dict[str, float]:
        """Get thresholds for specific detector based on profile"""
    
    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load predefined profile (strict/normal/relaxed)"""
    
    def apply_auto_tuned_thresholds(self, tuned_thresholds: Dict[str, float]):
        """Apply thresholds from auto-tuner"""
```

**File Location**: `src/detector_manager.py` (new file)

---

### 4. Fixed Statistical Outlier Detector

**Changes to Existing Code**:

**Problem**: KeyError при доступе к индексам после фильтрации

**Solution**: Использовать `.loc[]` вместо прямого индексирования

```python
# BEFORE (causes KeyError)
for idx in outlier_indices:
    actual_value = df.loc[idx, indicator]  # idx may not exist in df
    
# AFTER (fixed)
for idx in outlier_indices:
    if idx not in df.index:
        continue
    actual_value = df.loc[idx, indicator]
```

**Additional Fix**: Сохранять оригинальные индексы при работе с подвыборками

```python
def detect_zscore_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    anomalies = []
    
    for indicator in indicator_cols:
        values = df[indicator].dropna()
        if len(values) < 3:
            continue
        
        # Calculate z-scores maintaining original index
        z_scores = np.abs(stats.zscore(values))
        z_scores_series = pd.Series(z_scores, index=values.index)
        
        # Find outliers using boolean indexing
        outlier_mask = z_scores_series > threshold
        
        # Iterate over original dataframe indices
        for idx in df.index[df.index.isin(outlier_mask[outlier_mask].index)]:
            # Safe access to values
            ...
```

**File Location**: Modify `src/anomaly_detector.py` (StatisticalOutlierDetector class)

---

### 5. Enhanced Geographic Detector

**Changes to Existing Code**:

**New Feature**: Type-aware comparison

```python
def detect_regional_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
    anomalies = []
    
    # Group by region AND municipality type
    for (region, muni_type), group_df in df.groupby(['region_name', 'municipality_type']):
        
        for indicator in indicator_cols:
            # Use robust statistics
            median_val = group_df[indicator].median()
            mad_val = median_absolute_deviation(group_df[indicator])
            
            # Calculate robust z-score
            robust_z_scores = (group_df[indicator] - median_val) / (1.4826 * mad_val)
            
            # Apply type-specific thresholds
            threshold = self._get_threshold_for_type(muni_type)
            
            outlier_mask = np.abs(robust_z_scores) > threshold
            ...
```

**New Method**:
```python
def _get_threshold_for_type(self, muni_type: str) -> float:
    """
    Returns:
    - capital: 3.5 (very relaxed)
    - urban: 2.5 (relaxed)
    - rural: 2.0 (normal)
    """
    thresholds = {
        'capital': 3.5,
        'urban': 2.5,
        'rural': 2.0
    }
    return thresholds.get(muni_type, 2.0)
```

**File Location**: Modify `src/anomaly_detector.py` (GeographicAnomalyDetector class)

---

### 6. Enhanced Results Aggregator

**New Features**:

**Priority Scoring**:
```python
class EnhancedResultsAggregator(ResultsAggregator):
    
    def calculate_priority_score(self, anomaly: Dict[str, Any]) -> float:
        """
        Priority = base_severity * type_weight * indicator_weight
        
        Type weights:
        - logical_inconsistency: 1.5
        - cross_source_discrepancy: 1.2
        - temporal_anomaly: 1.1
        - statistical_outlier: 1.0
        - geographic_anomaly: 0.8
        
        Indicator weights:
        - population_*: 1.3
        - consumption_total: 1.2
        - salary_*: 1.1
        - other: 1.0
        """
```

**Anomaly Grouping**:
```python
def group_related_anomalies(self, anomalies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups anomalies by territory_id and identifies patterns:
    - Multiple indicators affected -> systemic issue
    - Same indicator across detectors -> high confidence
    - Related indicators (e.g., all salary_*) -> category issue
    """
```

**Root Cause Analysis**:
```python
def identify_root_cause(self, territory_anomalies: pd.DataFrame) -> str:
    """
    Analyzes anomalies for one territory and suggests root cause:
    - "Данные отсутствуют" (>70% missing)
    - "Дубликаты записей" (duplicate IDs)
    - "Систематическое расхождение источников" (multiple cross-source)
    - "Экстремальные значения" (multiple high severity)
    - "Неизвестная причина"
    """
```

**File Location**: Extend `src/results_aggregator.py`

---

### 7. Enhanced Exporter

**New Features**:

**Management-Friendly Descriptions**:
```python
class DescriptionFormatter:
    def format_for_management(self, anomaly: Dict[str, Any]) -> str:
        """
        Transforms technical description to business language:
        
        BEFORE: "salary_Финансы и страхование value 635238.74 is 6.49 
                 std deviations above unknown municipality average 117155.65"
        
        AFTER:  "Зарплата в секторе 'Финансы и страхование' в 5.4 раза выше 
                 средней по региону (635 тыс. руб. vs 117 тыс. руб.)"
        """
```

**Executive Summary Generator**:
```python
class ExecutiveSummaryGenerator:
    def generate(self, anomalies_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns:
        {
            'total_anomalies': int,
            'critical_count': int,
            'affected_municipalities': int,
            'top_10_municipalities': List[Dict],
            'key_findings': List[str],  # Bullet points in Russian
            'recommendations': List[str]
        }
        """
```

**Dashboard Visualization**:
```python
def create_dashboard_summary(self, anomalies_df: pd.DataFrame, output_path: str):
    """
    Creates single-page dashboard with:
    - Top 10 municipalities (bar chart)
    - Anomaly distribution by type (pie chart)
    - Severity distribution (histogram)
    - Geographic heatmap
    - Key metrics (text boxes)
    """
```

**File Location**: Extend `src/exporter.py`

---

## Data Models

### Enhanced Anomaly Record

**Extended Fields**:
```python
{
    # Existing fields
    'anomaly_id': str,
    'territory_id': int,
    'municipal_name': str,
    'region_name': str,
    'indicator': str,
    'anomaly_type': str,
    'actual_value': float,
    'expected_value': float,
    'deviation': float,
    'deviation_pct': float,
    'severity_score': float,
    'z_score': float,
    'data_source': str,
    'detection_method': str,
    'description': str,
    'potential_explanation': str,
    'detected_at': datetime,
    
    # NEW fields (Phase 2)
    'priority_score': float,  # Weighted severity
    'municipality_type': str,  # 'capital', 'urban', 'rural'
    'description_management': str,  # Business-friendly description
    'relative_deviation': str,  # "в 5.4 раза выше"
    'comparison_context': str,  # "выше 95% муниципалитетов региона"
    'anomaly_group_id': str,  # UUID for grouped anomalies
    'root_cause': str,  # Identified root cause
    'confidence': float,  # 0-1, based on multiple detectors agreement
}
```

### Configuration Schema

**Enhanced config.yaml**:
```yaml
# Detection profile (NEW)
detection_profile: "normal"  # strict, normal, relaxed

# Temporal data handling (NEW)
temporal:
  enabled: false
  aggregation_method: "latest"  # latest, mean, median
  auto_detect: true

# Municipality classification (NEW)
municipality_classification:
  enabled: true
  capital_cities:
    - "Москва"
    - "Санкт-Петербург"
    # ... other capitals
  urban_population_threshold: 50000

# Threshold profiles (NEW)
threshold_profiles:
  strict:
    statistical:
      z_score: 2.5
      iqr_multiplier: 1.2
    geographic:
      regional_z_score: 1.5
      cluster_threshold: 2.0
  
  normal:
    statistical:
      z_score: 3.0
      iqr_multiplier: 1.5
    geographic:
      regional_z_score: 2.5
      cluster_threshold: 2.5
  
  relaxed:
    statistical:
      z_score: 3.5
      iqr_multiplier: 2.0
    geographic:
      regional_z_score: 3.0
      cluster_threshold: 3.0

# Auto-tuning (NEW - Phase 3)
auto_tuning:
  enabled: false
  target_false_positive_rate: 0.05
  min_anomalies_per_detector: 10
  max_anomalies_per_detector: 1000
  retuning_interval_days: 30

# Robust statistics (NEW)
robust_statistics:
  enabled: true
  use_median: true
  use_mad: true
  winsorization_limits: [0.01, 0.99]
  log_transform_skewness_threshold: 2.0

# Priority weights (NEW)
priority_weights:
  anomaly_types:
    logical_inconsistency: 1.5
    cross_source_discrepancy: 1.2
    temporal_anomaly: 1.1
    statistical_outlier: 1.0
    geographic_anomaly: 0.8
  
  indicators:
    population: 1.3
    consumption_total: 1.2
    salary: 1.1
    default: 1.0

# Export settings (ENHANCED)
export:
  output_dir: "output"
  timestamp_format: "%Y%m%d_%H%M%S"
  top_n_municipalities: 50
  
  # NEW
  generate_executive_summary: true
  generate_dashboard: true
  use_management_descriptions: true
  highlight_critical_threshold: 90

# Existing settings remain unchanged for compatibility
thresholds:
  statistical:
    z_score: 3.0
    iqr_multiplier: 1.5
    percentile_lower: 1
    percentile_upper: 99
  # ... rest of existing config
```

---

## Error Handling

### Detector Error Handling Strategy

**Principle**: Fail gracefully - continue with other detectors if one fails

**Implementation**:
```python
class DetectorManager:
    def run_all_detectors(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        results = []
        
        for detector_name, detector in self.detectors.items():
            try:
                logger.info(f"Running {detector_name}...")
                anomalies = detector.detect(df)
                results.append(anomalies)
                logger.info(f"{detector_name} completed: {len(anomalies)} anomalies")
                
            except Exception as e:
                logger.error(
                    f"{detector_name} failed with error: {str(e)}",
                    exc_info=True,
                    extra={
                        'detector': detector_name,
                        'data_shape': df.shape,
                        'error_type': type(e).__name__
                    }
                )
                # Continue with other detectors
                continue
        
        return results
```

### Data Quality Warnings

**Warning Levels**:
- **INFO**: Normal operation (e.g., "Skipped indicator with 60% missing values")
- **WARNING**: Potential issue (e.g., "Found 438 duplicate territory_ids")
- **ERROR**: Detector failure (e.g., "StatisticalOutlierDetector failed: KeyError")
- **CRITICAL**: Pipeline failure (e.g., "Cannot load data files")

**Structured Logging**:
```python
logger.warning(
    "High percentage of geographic anomalies detected",
    extra={
        'total_anomalies': 12746,
        'geographic_anomalies': 8898,
        'percentage': 69.8,
        'recommendation': 'Consider adjusting geographic thresholds'
    }
)
```

---

## Testing Strategy

### Unit Tests

**Phase 1 Tests**:
```python
# test_enhanced_data_loader.py
def test_temporal_structure_detection()
def test_duplicate_detection_temporal()
def test_duplicate_detection_error()
def test_source_mapping_by_prefix()
def test_aggregation_latest()
def test_aggregation_mean()

# test_fixed_detectors.py
def test_statistical_detector_no_keyerror()
def test_statistical_detector_with_missing_indices()
def test_geographic_detector_with_types()
```

**Phase 2 Tests**:
```python
# test_data_preprocessor.py
def test_municipality_classification()
def test_robust_stats_calculation()
def test_winsorization()
def test_log_transformation()

# test_enhanced_aggregator.py
def test_priority_score_calculation()
def test_anomaly_grouping()
def test_root_cause_identification()

# test_enhanced_exporter.py
def test_management_description_formatting()
def test_executive_summary_generation()
def test_dashboard_creation()
```

**Phase 3 Tests**:
```python
# test_auto_tuner.py
def test_threshold_optimization()
def test_false_positive_rate_calculation()
def test_threshold_validation()
def test_periodic_retuning()
```

### Integration Tests

```python
# test_full_pipeline.py
def test_pipeline_with_temporal_data()
def test_pipeline_with_duplicates()
def test_pipeline_with_all_detectors()
def test_pipeline_with_detector_failure()
def test_pipeline_output_compatibility()
```

### Regression Tests

```python
# test_regression.py
def test_statistical_detector_keyerror_fixed()
def test_no_duplicate_warnings_for_temporal_data()
def test_geographic_anomalies_reduced()
```

---

## Performance Considerations

### Current Performance
- **Execution Time**: 15.5 seconds for 3,101 municipalities
- **Memory Usage**: ~500MB peak
- **Bottlenecks**: Geographic detector (nested loops)

### Optimization Strategies

**1. Caching Statistics**:
```python
class DataPreprocessor:
    def __init__(self):
        self._stats_cache = {}
    
    def get_robust_stats(self, indicator: str, group_key: str) -> RobustStats:
        cache_key = f"{indicator}_{group_key}"
        if cache_key not in self._stats_cache:
            self._stats_cache[cache_key] = self._calculate_stats(...)
        return self._stats_cache[cache_key]
```

**2. Vectorized Operations**:
```python
# Instead of loops, use pandas vectorized operations
df['robust_z_score'] = (df['value'] - df['median']) / df['mad']
outliers = df[df['robust_z_score'].abs() > threshold]
```

**3. Early Filtering**:
```python
# Skip indicators with too many missing values early
valid_indicators = [
    col for col in indicator_cols 
    if df[col].notna().sum() / len(df) > 0.5
]
```

**Expected Performance After Improvements**:
- **Execution Time**: 12-15 seconds (similar, due to added preprocessing)
- **Memory Usage**: ~600MB (slight increase due to caching)
- **Reliability**: 100% (no detector failures)

---

## Migration and Compatibility

### Backward Compatibility

**Guaranteed**:
- ✅ Existing CSV/Excel file structure unchanged
- ✅ All existing config.yaml parameters supported
- ✅ Existing API interfaces preserved
- ✅ Output file naming convention maintained

**New Optional Features**:
- New fields in anomaly records (appended at end)
- New config sections (ignored if not present)
- New output files (executive summary, dashboard)

### Migration Path

**For Users**:
1. Update code (no config changes required)
2. Run with existing config - works as before
3. Optionally enable new features via config
4. Optionally switch to new detection profile

**Config Migration**:
```python
# Old config still works
thresholds:
  statistical:
    z_score: 3.0

# New config adds options
detection_profile: "normal"  # Uses same thresholds as old config
```

### Deprecation Policy

**No Deprecations in This Release**:
- All existing functionality preserved
- New features are additive only
- Future deprecations will have 6-month warning period

---

## Phase-Specific Implementation Details

### Phase 1: Critical Fixes (Week 1-2)

**Files to Modify**:
- `src/data_loader.py` - Add temporal analysis
- `src/anomaly_detector.py` - Fix StatisticalOutlierDetector
- `src/detector_manager.py` - NEW file
- `tests/test_detectors.py` - Add regression tests

**Deliverables**:
- ✅ No KeyError in StatisticalOutlierDetector
- ✅ Duplicate detection and handling
- ✅ Improved logging
- ✅ All detectors run successfully

### Phase 2: Quality Improvements (Week 3-5)

**Files to Modify**:
- `src/data_preprocessor.py` - NEW file
- `src/anomaly_detector.py` - Enhance GeographicAnomalyDetector
- `src/results_aggregator.py` - Add priority scoring
- `src/exporter.py` - Add management reports
- `tests/test_preprocessor.py` - NEW file

**Deliverables**:
- ✅ Geographic anomalies reduced to ~30%
- ✅ Management-friendly descriptions
- ✅ Executive summary
- ✅ Priority-based ranking

### Phase 3: Auto-tuning (Week 6-7)

**Files to Create**:
- `src/auto_tuner.py` - NEW file
- `src/threshold_optimizer.py` - NEW file
- `tests/test_auto_tuner.py` - NEW file

**Deliverables**:
- ✅ Automatic threshold optimization
- ✅ Configuration profiles
- ✅ Periodic re-tuning capability

### Phase 4: Continuous

**Activities**:
- Documentation updates
- Integration testing
- Performance profiling
- User acceptance testing

---

## Monitoring and Observability

### Metrics to Track

**Detection Metrics**:
```python
{
    'total_anomalies': 12746,
    'anomalies_by_type': {
        'statistical_outlier': 0,
        'geographic_anomaly': 8898,
        'temporal_anomaly': 0,
        'cross_source_discrepancy': 2843,
        'logical_inconsistency': 3943
    },
    'anomalies_by_severity': {
        'critical': 5234,  # 90-100
        'high': 3421,      # 70-90
        'medium': 2891,    # 50-70
        'low': 1200        # 0-50
    },
    'municipalities_affected': 2634,
    'municipalities_affected_pct': 84.9,
    'avg_anomalies_per_municipality': 4.8
}
```

**Performance Metrics**:
```python
{
    'execution_time_seconds': 15.5,
    'detector_times': {
        'statistical': 0.0,  # Failed
        'geographic': 8.2,
        'temporal': 0.1,
        'cross_source': 3.4,
        'logical': 2.8
    },
    'memory_peak_mb': 512,
    'data_loaded_rows': 5243143
}
```

**Quality Metrics**:
```python
{
    'data_completeness': 0.8412,
    'duplicate_territories': 438,
    'missing_indicators_pct': 15.88,
    'detector_success_rate': 0.80,  # 4/5 detectors succeeded
    'estimated_false_positive_rate': 0.70  # Based on geographic anomalies
}
```

### Logging Strategy

**Log Levels**:
- **DEBUG**: Detailed execution flow
- **INFO**: Major steps and statistics
- **WARNING**: Data quality issues, high anomaly counts
- **ERROR**: Detector failures
- **CRITICAL**: Pipeline failures

**Log Format**:
```
2025-10-31 02:20:45 - data_loader - INFO - Loaded 3,101 municipalities with 36 indicators
2025-10-31 02:20:45 - data_loader - WARNING - Found 438 duplicate territory_ids
2025-10-31 02:20:46 - detector_manager - INFO - Running StatisticalOutlierDetector...
2025-10-31 02:20:46 - detector_manager - ERROR - StatisticalOutlierDetector failed: KeyError(2103)
2025-10-31 02:20:52 - results_aggregator - INFO - Aggregated 12,746 anomalies from 4 detectors
```

---

## Security Considerations

### Data Privacy
- No PII in logs or error messages
- Sanitize file paths in logs
- No sensitive data in exception messages

### Input Validation
- Validate config.yaml schema
- Check file paths for directory traversal
- Validate threshold ranges (0-100 for percentages, >0 for multipliers)

### Error Messages
- Don't expose internal paths
- Don't include data samples in production logs
- Sanitize SQL-like queries if any

---

## Future Enhancements (Post-Phase 3)

### Machine Learning Detectors
- Isolation Forest for multivariate outliers
- Autoencoder for pattern anomalies
- DBSCAN for spatial clustering
- Requires labeled data for validation

### Real-time Processing
- Streaming data support
- Incremental detection
- Online threshold adaptation

### Interactive Dashboard
- Web-based UI for exploring anomalies
- Drill-down capabilities
- Anomaly feedback mechanism

### API Integration
- REST API for programmatic access
- Webhook notifications for critical anomalies
- Integration with monitoring systems

---

## Conclusion

Данный design обеспечивает:

1. **Надежность**: Все критические ошибки исправлены, детекторы работают стабильно
2. **Качество**: Ложные срабатывания снижены с 70% до ~30% через type-aware анализ
3. **Удобство**: Понятные отчеты для менеджмента с executive summary
4. **Автоматизация**: Auto-tuning минимизирует ручную настройку
5. **Совместимость**: Полная обратная совместимость с существующим кодом

Поэтапная реализация позволяет получать ценность на каждом этапе, минимизируя риски.
