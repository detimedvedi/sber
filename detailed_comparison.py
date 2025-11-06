import pandas as pd
import numpy as np

print("=" * 80)
print("📊 ДЕТАЛЬНОЕ СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 80)

# Загрузить базовые результаты (до оптимизации)
print("\n📂 Загрузка базовых результатов (ДО оптимизации)...")
baseline = pd.read_csv('output/anomalies_master_20251101_054021.csv')
print(f"✓ Загружено: {len(baseline):,} аномалий")

# Загрузить новые результаты (после оптимизации)
print("\n📂 Загрузка новых результатов (ПОСЛЕ оптимизации)...")
current = pd.read_csv('output/anomalies_master_20251104_021837.csv')
print(f"✓ Загружено: {len(current):,} аномалий")

print("\n" + "=" * 80)
print("🔢 ОБЩИЕ МЕТРИКИ")
print("=" * 80)

# Основные метрики
metrics = {
    'Всего аномалий': (len(baseline), len(current)),
    'Уникальных территорий': (baseline['territory_id'].nunique(), current['territory_id'].nunique()),
    'Среднее на территорию': (
        len(baseline) / baseline['territory_id'].nunique(),
        len(current) / current['territory_id'].nunique()
    ),
    'Критических (>80)': (
        (baseline['severity_score'] > 80).sum(),
        (current['severity_score'] > 80).sum()
    ),
    '% критических': (
        (baseline['severity_score'] > 80).sum() / len(baseline) * 100,
        (current['severity_score'] > 80).sum() / len(current) * 100
    ),
}

for metric, (before, after) in metrics.items():
    change = after - before
    change_pct = (change / before * 100) if before != 0 else 0
    
    # Форматирование
    if metric in ['Среднее на территорию', '% критических']:
        before_str = f"{before:.2f}"
        after_str = f"{after:.2f}"
        change_str = f"{change:+.2f}"
    else:
        before_str = f"{int(before):,}"
        after_str = f"{int(after):,}"
        change_str = f"{int(change):+,}"
    
    # Иконка
    if change < 0:
        icon = "✅"
    elif change > 0:
        icon = "⚠️"
    else:
        icon = "➖"
    
    print(f"{metric:30s}: {before_str:>12s} → {after_str:>12s}  {icon} {change_str:>12s} ({change_pct:+6.1f}%)")

print("\n" + "=" * 80)
print("📋 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ АНОМАЛИЙ")
print("=" * 80)

baseline_types = baseline['anomaly_type'].value_counts()
current_types = current['anomaly_type'].value_counts()

all_types = set(baseline_types.index) | set(current_types.index)

for anom_type in sorted(all_types):
    before = baseline_types.get(anom_type, 0)
    after = current_types.get(anom_type, 0)
    change = after - before
    change_pct = (change / before * 100) if before != 0 else 0
    
    before_pct = before / len(baseline) * 100
    after_pct = after / len(current) * 100 if len(current) > 0 else 0
    
    if change < 0:
        icon = "✅"
    elif change > 0:
        icon = "⚠️"
    else:
        icon = "➖"
    
    print(f"{anom_type:35s}: {int(before):>6,} ({before_pct:4.1f}%) → {int(after):>6,} ({after_pct:4.1f}%)  {icon} {int(change):+6,} ({change_pct:+6.1f}%)")

print("\n" + "=" * 80)
print("🏆 ТОП-10 САМЫХ АНОМАЛЬНЫХ ТЕРРИТОРИЙ")
print("=" * 80)

print("\nДО оптимизации:")
baseline_top = baseline['municipal_name'].value_counts().head(10)
for i, (muni, count) in enumerate(baseline_top.items(), 1):
    print(f"{i:2d}. {muni:40s}: {count:3d} аномалий")

print("\nПОСЛЕ оптимизации:")
current_top = current['municipal_name'].value_counts().head(10)
for i, (muni, count) in enumerate(current_top.items(), 1):
    print(f"{i:2d}. {muni:40s}: {count:3d} аномалий")

print("\n" + "=" * 80)
print("📈 РАСПРЕДЕЛЕНИЕ ПО ИНДИКАТОРАМ (топ-10)")
print("=" * 80)

print("\nДО оптимизации:")
baseline_ind = baseline['indicator'].value_counts().head(10)
for indicator, count in baseline_ind.items():
    pct = count / len(baseline) * 100
    print(f"  {indicator:50s}: {count:4d} ({pct:4.1f}%)")

print("\nПОСЛЕ оптимизации:")
current_ind = current['indicator'].value_counts().head(10)
for indicator, count in current_ind.items():
    pct = count / len(current) * 100
    print(f"  {indicator:50s}: {count:4d} ({pct:4.1f}%)")

print("\n" + "=" * 80)
print("🎯 ОЦЕНКА ЭФФЕКТИВНОСТИ")
print("=" * 80)

reduction = (len(baseline) - len(current)) / len(baseline) * 100
territory_before = baseline['territory_id'].nunique() / 2571 * 100
territory_after = current['territory_id'].nunique() / 2571 * 100

print(f"\nСнижение аномалий: {reduction:.1f}%")
print(f"Территорий помечено: {territory_before:.1f}% → {territory_after:.1f}%")

# Оценка
if reduction > 70:
    grade = "⭐⭐⭐⭐⭐ ОТЛИЧНО"
    comment = "Значительное снижение ложных срабатываний!"
elif reduction > 40:
    grade = "⭐⭐⭐⭐ ХОРОШО"
    comment = "Заметное улучшение качества детекции"
elif reduction > 20:
    grade = "⭐⭐⭐ УДОВЛЕТВОРИТЕЛЬНО"
    comment = "Есть улучшение, но можно настроить жёстче"
elif reduction > 0:
    grade = "⭐⭐ СЛАБО"
    comment = "Малое улучшение, требуется дополнительная настройка"
else:
    grade = "⭐ ПЛОХО"
    comment = "Увеличение аномалий - проверьте конфигурацию!"

print(f"\n{grade}")
print(f"Комментарий: {comment}")

print("\n" + "=" * 80)
print("💡 РЕКОМЕНДАЦИИ")
print("=" * 80)

if reduction < 70:
    print("\n⚠️ ПРОБЛЕМА: Снижение меньше целевого (70%)")
    print("\nПочему так произошло:")
    print("  • Возможно, auto-tuning не сработал (требуется больше данных)")
    print("  • Temporal анализ может добавлять новые аномалии")
    print("  • Relaxed профиль недостаточно мягкий для данных")
    
    print("\nЧто делать:")
    print("  1. Ужесточить пороги вручную:")
    print("     geographic.regional_z_score: 3.5 → 4.5")
    print("     geographic.cluster_threshold: 4.0 → 5.0")
    print()
    print("  2. Отключить temporal анализ временно:")
    print("     temporal.enabled: true → false")
    print()
    print("  3. Проверить логи auto-tuning:")
    print("     cat output/anomaly_detection.log | grep 'auto.tuning'")

if territory_after > 50:
    print(f"\n⚠️ ПРОБЛЕМА: {territory_after:.0f}% территорий всё ещё помечены")
    print("  Целевой уровень: 15-20%")
    print("  Текущий уровень: слишком высокий")

# Анализ типов
geographic_pct = current_types.get('geographic_anomaly', 0) / len(current) * 100 if len(current) > 0 else 0
if geographic_pct > 30:
    print(f"\n⚠️ ПРОБЛЕМА: Geographic anomalies всё ещё {geographic_pct:.1f}%")
    print("  Целевой уровень: <25%")
    print("  Решение: Ужесточить geographic пороги")

# Позитивные моменты
print("\n✅ ПОЗИТИВНЫЕ МОМЕНТЫ:")
if reduction > 0:
    print(f"  • Аномалий стало меньше на {reduction:.1f}%")

critical_reduction = ((baseline['severity_score'] > 80).sum() - (current['severity_score'] > 80).sum()) / (baseline['severity_score'] > 80).sum() * 100
if critical_reduction > 0:
    print(f"  • Критических аномалий меньше на {critical_reduction:.1f}%")

if 'legitimate_pattern' in current_types.index:
    print(f"  • Фильтр легитимных паттернов работает: {current_types['legitimate_pattern']} переклассифицировано")

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)
