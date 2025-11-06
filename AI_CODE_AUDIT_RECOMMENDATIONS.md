# 🔧 РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ

## 🔴 КРИТИЧНО (Priority 1) - Исправить немедленно

### 1. Исправить обработку Consumption данных

**Файл:** `src/data_loader.py:219-238`

**Текущий код (НЕВЕРНО):**
```python
consumption_pivot = consumption_df.pivot_table(
    index='territory_id',
    columns='category',
    values='value',
    aggfunc='mean'  # ❌ Уничтожает временную структуру
).reset_index()
```

**Исправленный код (Option 1 - Latest period):**
```python
# Берём только последний период для каждой категории
consumption_latest = consumption_df.sort_values('date').groupby(
    ['territory_id', 'category']
).last().reset_index()

consumption_pivot = consumption_latest.pivot(
    index='territory_id',
    columns='category',
    values='value'
).reset_index()
```

**Исправленный код (Option 2 - Keep temporal structure):**
```python
# Сохраняем ВСЕ периоды для temporal analysis
consumption_pivot = consumption_df.pivot(
    index=['territory_id', 'date'],
    columns='category',
    values='value'
).reset_index()

# Затем в детекторах использовать groupby('territory_id') для анализа
```

**Эффект:** -25% false positives, включение temporal detection

---

### 2. Отключить CrossSourceComparator

**Файл:** `config.yaml`

**Добавить:**
```yaml
detectors:
  statistical: 
    enabled: true
  temporal:
    enabled: true
  geographic:
    enabled: true
  cross_source:
    enabled: false  # ❌ Нет перекрывающихся метрик между источниками
  logical:
    enabled: true
```

**Или в `src/detector_manager.py` закомментировать:**
```python
# detectors['cross_source'] = CrossSourceComparator(self.config)
```

**Эффект:** -15% false positives (-2,414 аномалий)

---

### 3. Загрузить и использовать граф связей

**Файл:** `src/data_loader.py`

**Добавить метод:**
```python
def load_connection_data(self) -> pd.DataFrame:
    """
    Load connection graph data.
    
    Returns:
        DataFrame with columns:
        - territory_id_x: Source territory
        - territory_id_y: Target territory
        - distance: Distance in km
        - type: Connection type (highway)
    """
    logger.info("Loading connection graph...")
    file_path = self.base_path / 'connection.parquet'
    
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded connection graph: {df.shape[0]} connections")
        return df
    except FileNotFoundError:
        logger.warning(f"Connection file not found: {file_path}")
        return pd.DataFrame()
```

**Файл:** `src/anomaly_detector.py` - GeographicAnomalyDetector

**Добавить метод:**
```python
def _get_neighbors(
    self, 
    territory_id: int, 
    connections: pd.DataFrame, 
    max_distance: float = 50.0
) -> List[int]:
    """
    Get neighboring territories from connection graph.
    
    Args:
        territory_id: Territory ID to find neighbors for
        connections: Connection graph DataFrame
        max_distance: Maximum distance in km to consider as neighbor
    
    Returns:
        List of neighbor territory IDs
    """
    # Find all connections involving this territory
    neighbors_x = connections[
        (connections['territory_id_x'] == territory_id) &
        (connections['distance'] <= max_distance)
    ]['territory_id_y'].tolist()
    
    neighbors_y = connections[
        (connections['territory_id_y'] == territory_id) &
        (connections['distance'] <= max_distance)
    ]['territory_id_x'].tolist()
    
    # Combine and deduplicate
    all_neighbors = list(set(neighbors_x + neighbors_y))
    
    return all_neighbors
```

**Модифицировать detect_cluster_outliers:**
```python
def detect_cluster_outliers(
    self, 
    df: pd.DataFrame, 
    connections: pd.DataFrame  # ✅ Добавить параметр
) -> List[Dict[str, Any]]:
    """
    Detect municipalities that differ from their neighbors.
    Uses real connection graph instead of administrative regions.
    """
    anomalies = []
    
    if connections.empty:
        self.logger.warning("No connection data - falling back to region-based clustering")
        return self._detect_cluster_outliers_legacy(df)
    
    # ... (остальной код с использованием _get_neighbors)
```

**Эффект:** -10% false positives, более точные geographic anomalies

---

## 🟡 ВАЖНО (Priority 2) - Исправить в ближайшее время

### 4. Ужесточить пороги для России

**Файл:** `config.yaml`

**Текущие значения:**
```yaml
thresholds:
  statistical:
    z_score: 3.0
    iqr_multiplier: 1.5
  geographic:
    regional_z_score: 3.5
    cluster_threshold: 4.0
```

**Рекомендуемые значения:**
```yaml
thresholds:
  statistical:
    z_score: 5.0          # 3.0 → 5.0 (Россия крайне неоднородна)
    iqr_multiplier: 3.0   # 1.5 → 3.0
    percentile_lower: 0.1  # 1 → 0.1 (только extreme outliers)
    percentile_upper: 99.9 # 99 → 99.9
  
  geographic:
    regional_z_score: 6.0  # 3.5 → 6.0 (учесть Москва vs Чукотка)
    cluster_threshold: 5.0 # 4.0 → 5.0
  
  cross_source:
    enabled: false  # Отключить
  
  logical:
    check_negative_values: true
    check_impossible_ratios: false  # Отключить (слишком много легитимных паттернов)
```

**Эффект:** -20% false positives

---

### 5. Отключить Auto-tuning

**Файл:** `config.yaml`

```yaml
auto_tuning:
  enabled: false  # ❌ Работает против целей
```

**Причина:** Auto-tuning СМЯГЧАЕТ пороги вместо ужесточения, так как:
- Нет ground truth dataset
- FPR расчёт неверен
- Оптимизирует в противоположном направлении

**Эффект:** Предотвращение дальнейшего роста false positives

---

### 6. Применить log-transform для скошенных распределений

**Файл:** `src/anomaly_detector.py` - StatisticalOutlierDetector

**Добавить перед расчётом z-scores:**
```python
def detect_zscore_outliers(self, df: pd.DataFrame, threshold: Optional[float] = None):
    ...
    for indicator in indicator_cols:
        values = df[indicator].dropna()
        
        # Check skewness
        skewness = values.skew()
        
        if abs(skewness) > 2.0:
            # Highly skewed - apply log transform
            self.logger.debug(f"Applying log transform to '{indicator}' (skewness={skewness:.2f})")
            values_transformed = np.log1p(values)  # log(1+x) to handle zeros
            mean_val = values_transformed.mean()
            std_val = values_transformed.std()
            z_scores = (values_transformed - mean_val) / std_val
        else:
            # Normal distribution - use original values
            mean_val = values.mean()
            std_val = values.std()
            z_scores = (values - mean_val) / std_val
```

**Эффект:** Правильная обработка power-law распределений (население, потребление)

---

### 7. Интегрировать legitimate pattern filter

**Файл:** `main.py`

**Добавить после агрегации аномалий (примерно строка 800):**
```python
# После combined_anomalies = aggregator.combine_anomalies(all_anomalies)

# Apply legitimate pattern filter
if len(combined_anomalies) > 0:
    try:
        from src.legitimate_pattern_filter import LegitimatePatternFilter
        
        logger.info("Applying legitimate pattern filter...")
        pattern_filter = LegitimatePatternFilter(config)
        
        # Filter anomalies
        filtered_anomalies = pattern_filter.filter_anomalies(combined_anomalies)
        
        # Count reclassified
        legitimate_count = (filtered_anomalies['is_legitimate_pattern'] == True).sum()
        logger.info(f"Reclassified {legitimate_count} anomalies as legitimate patterns")
        
        # Remove or flag legitimate patterns
        combined_anomalies = filtered_anomalies[
            filtered_anomalies['is_legitimate_pattern'] == False
        ]
        
        logger.info(f"After filtering: {len(combined_anomalies)} anomalies remain")
        
    except Exception as e:
        logger.warning(f"Failed to apply pattern filter: {e}")
        # Continue without filtering
```

**Эффект:** -8% false positives (легитимные паттерны: туризм, деловые центры)

---

## 🟢 ЖЕЛАТЕЛЬНО (Priority 3) - Улучшения

### 8. Добавить municipality whitelist

**Файл:** `config.yaml`

```yaml
whitelists:
  # Territories with known unique characteristics
  unique_municipalities:
    - territory_id: 42  # Норильск - экстремальный север
      reason: "Northernmost major city, extreme conditions"
    - territory_id: 123  # Билибинский район
      reason: "Remote Chukotka region, naturally different"
    - territory_id: 456  # Сочи
      reason: "Major tourist destination"
  
  # Automatically whitelist by category
  auto_whitelist:
    - type: "capital"  # All regional capitals
    - type: "tourism_zone"  # Tourist territories
    - type: "industrial"  # Major industrial centers
```

**Эффект:** -5% false positives (известные уникальные территории)

---

### 9. Улучшить Detection Metrics

**Файл:** `src/detector_manager.py`

**Добавить расчёт метрик:**
```python
def calculate_detection_metrics(self, anomalies_df: pd.DataFrame, df: pd.DataFrame):
    """
    Calculate detection quality metrics.
    
    Returns:
        - coverage: % territories with at least 1 anomaly
        - intensity: avg anomalies per flagged territory
        - diversity: distribution across anomaly types
        - concentration: % anomalies in top 10% territories
    """
    metrics = {}
    
    total_territories = df['territory_id'].nunique()
    flagged_territories = anomalies_df['territory_id'].nunique()
    
    metrics['coverage'] = flagged_territories / total_territories
    metrics['intensity'] = len(anomalies_df) / flagged_territories if flagged_territories > 0 else 0
    
    # Type diversity
    type_counts = anomalies_df['anomaly_type'].value_counts()
    metrics['diversity'] = len(type_counts) / 5  # 5 types total
    
    # Concentration (Gini-like)
    terr_counts = anomalies_df['territory_id'].value_counts()
    top10_pct = int(len(terr_counts) * 0.1)
    top10_anomalies = terr_counts.head(top10_pct).sum()
    metrics['concentration'] = top10_anomalies / len(anomalies_df)
    
    return metrics
```

**Эффект:** Лучший мониторинг качества детекции

---

## 📋 ПОШАГОВЫЙ ПЛАН ИСПРАВЛЕНИЯ

### Шаг 1: Critical Fixes (1-2 часа)

1. ✅ Исправить `data_loader.py` - consumption aggregation
2. ✅ Отключить CrossSourceComparator в config
3. ✅ Отключить Auto-tuning в config
4. ✅ Ужесточить пороги в config

**Запустить тест:**
```bash
python main.py
python analyze_anomalies.py
```

**Ожидаемый результат:** Аномалий 8,000-10,000 (~50% снижение)

---

### Шаг 2: Important Fixes (2-3 часа)

5. ✅ Добавить загрузку connection graph
6. ✅ Модифицировать GeographicAnomalyDetector
7. ✅ Применить log-transform в StatisticalOutlierDetector
8. ✅ Интегрировать legitimate pattern filter

**Запустить финальный тест:**
```bash
python main.py
python analyze_anomalies.py
```

**Ожидаемый результат:** Аномалий 3,000-4,000 (~75-80% снижение от original)

---

### Шаг 3: Optional Improvements (1-2 часа)

9. ✅ Добавить whitelist
10. ✅ Улучшить метрики
11. ✅ Добавить unit tests
12. ✅ Обновить документацию

---

## 🎯 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

### После ВСЕХ исправлений:

| Метрика | Сейчас | После | Улучшение |
|---------|--------|-------|-----------|
| **Всего аномалий** | 16,682 | ~3,500 | ↓79% |
| **Территорий помечено** | 103% (2,659) | ~30% (~770) | ↓70% |
| **Geographic anomalies** | 32.7% | ~18% | ↓45% |
| **Cross-source** | 14.5% | 0% | ↓100% |
| **Logical** | 24.1% | ~12% | ↓50% |
| **Statistical** | 28.8% | ~15% | ↓48% |
| **Temporal** | 0% | ~5% | NEW |

### Качественные улучшения:

✅ **Точность:** Аномалии реально аномальные (не естественная вариация)  
✅ **Полнота:** Temporal anomalies теперь обнаруживаются  
✅ **Релевантность:** Используется граф связей для geographic analysis  
✅ **Понятность:** Меньше ложных срабатываний = легче интерпретировать

---

## 📝 ЧЕКЛИСТ ДЛЯ ПРОВЕРКИ

После всех исправлений проверить:

- [ ] Consumption данные сохраняют date колонку
- [ ] TemporalAnomalyDetector находит >0 аномалий
- [ ] GeographicAnomalyDetector использует connection graph
- [ ] CrossSourceComparator отключен
- [ ] Auto-tuning отключен
- [ ] Пороги ужесточены (z_score >= 5.0)
- [ ] Log-transform применяется для skewed distributions
- [ ] Legitimate pattern filter интегрирован
- [ ] Всего аномалий < 5,000
- [ ] Территорий помечено < 35%
- [ ] Есть категория "legitimate_pattern" или "temporal_anomaly"

---

**Дата:** 6 ноября 2025  
**Автор:** Независимый аудитор  
**Статус:** ⏳ Готов к внедрению
