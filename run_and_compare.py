"""
Script to run anomaly detection with new settings and compare results

This script:
1. Runs the analysis with optimized configuration
2. Compares results with previous run (baseline)
3. Generates a comparison report
"""

import pandas as pd
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def load_latest_results(pattern="anomalies_master_*.csv"):
    """Load the most recent anomaly detection results."""
    output_dir = Path("output")
    files = sorted(output_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not files:
        print(f"❌ No files matching {pattern} found in output/")
        return None
    
    latest = files[0]
    print(f"📂 Loading: {latest.name}")
    return pd.read_csv(latest)


def analyze_results(df, label="Results"):
    """Analyze and print statistics about anomaly detection results."""
    if df is None or df.empty:
        print(f"⚠️ No data for {label}")
        return None
    
    stats = {
        'total_anomalies': len(df),
        'unique_territories': df['territory_id'].nunique(),
        'avg_per_territory': len(df) / df['territory_id'].nunique(),
        'critical_pct': (df['severity_score'] > 80).sum() / len(df) * 100,
        'type_distribution': df['anomaly_type'].value_counts().to_dict(),
        'top_municipality': df['municipal_name'].value_counts().iloc[0] if len(df) > 0 else 0,
        'top_muni_count': df['municipal_name'].value_counts().values[0] if len(df) > 0 else 0,
    }
    
    return stats


def print_comparison(before_stats, after_stats):
    """Print a detailed comparison of before/after results."""
    print("\n" + "=" * 80)
    print("📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ: ДО vs ПОСЛЕ ОПТИМИЗАЦИИ")
    print("=" * 80)
    
    if before_stats is None or after_stats is None:
        print("⚠️ Недостаточно данных для сравнения")
        return
    
    # Overall metrics
    print("\n🔢 ОБЩИЕ МЕТРИКИ")
    print("-" * 80)
    
    metrics = [
        ('Всего аномалий', 'total_anomalies'),
        ('Уникальных территорий', 'unique_territories'),
        ('Среднее на территорию', 'avg_per_territory'),
        ('% критических (>80)', 'critical_pct'),
    ]
    
    for label, key in metrics:
        before = before_stats[key]
        after = after_stats[key]
        change = after - before
        change_pct = (change / before * 100) if before != 0 else 0
        
        # Format based on type
        if key == 'avg_per_territory':
            before_str = f"{before:.1f}"
            after_str = f"{after:.1f}"
            change_str = f"{change:+.1f}"
        elif key == 'critical_pct':
            before_str = f"{before:.1f}%"
            after_str = f"{after:.1f}%"
            change_str = f"{change:+.1f}pp"
        else:
            before_str = f"{int(before):,}"
            after_str = f"{int(after):,}"
            change_str = f"{int(change):+,}"
        
        # Color coding
        if change < 0:
            symbol = "✅"
            color = "зелёный"
        elif change > 0:
            symbol = "⚠️"
            color = "красный"
        else:
            symbol = "➖"
            color = "жёлтый"
        
        print(f"{label:35s}: {before_str:>12s} → {after_str:>12s}  {symbol} {change_str:>12s} ({change_pct:+.1f}%)")
    
    # Type distribution
    print("\n📋 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ АНОМАЛИЙ")
    print("-" * 80)
    
    all_types = set(before_stats['type_distribution'].keys()) | set(after_stats['type_distribution'].keys())
    
    for anom_type in sorted(all_types):
        before = before_stats['type_distribution'].get(anom_type, 0)
        after = after_stats['type_distribution'].get(anom_type, 0)
        change = after - before
        change_pct = (change / before * 100) if before != 0 else 0
        
        before_pct = before / before_stats['total_anomalies'] * 100
        after_pct = after / after_stats['total_anomalies'] * 100
        
        if change < 0:
            symbol = "✅"
        elif change > 0:
            symbol = "⚠️" if anom_type != 'legitimate_pattern' else "ℹ️"
        else:
            symbol = "➖"
        
        print(f"{anom_type:35s}: {int(before):>6,} ({before_pct:4.1f}%) → {int(after):>6,} ({after_pct:4.1f}%)  {symbol} {int(change):+6,}")
    
    # Top municipalities
    print("\n🏆 САМАЯ АНОМАЛЬНАЯ ТЕРРИТОРИЯ")
    print("-" * 80)
    print(f"До:    {before_stats['top_municipality']} ({int(before_stats['top_muni_count'])} аномалий)")
    print(f"После: {after_stats['top_municipality']} ({int(after_stats['top_muni_count'])} аномалий)")
    
    # Assessment
    print("\n" + "=" * 80)
    print("📈 ОЦЕНКА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    anomaly_reduction = (before_stats['total_anomalies'] - after_stats['total_anomalies']) / before_stats['total_anomalies'] * 100
    
    if anomaly_reduction > 70:
        assessment = "✅ ОТЛИЧНО"
        comment = "Значительное снижение ложных срабатываний"
    elif anomaly_reduction > 40:
        assessment = "✅ ХОРОШО"
        comment = "Заметное улучшение качества детекции"
    elif anomaly_reduction > 20:
        assessment = "⚠️ УДОВЛЕТВОРИТЕЛЬНО"
        comment = "Есть улучшение, но можно настроить жёстче"
    elif anomaly_reduction > 0:
        assessment = "⚠️ НЕЗНАЧИТЕЛЬНО"
        comment = "Малое улучшение, рекомендуется дополнительная настройка"
    else:
        assessment = "❌ УХУДШЕНИЕ"
        comment = "Количество аномалий увеличилось! Проверьте конфигурацию"
    
    print(f"\nСнижение аномалий: {anomaly_reduction:.1f}%")
    print(f"Оценка: {assessment}")
    print(f"Комментарий: {comment}")
    
    # Recommendations
    print("\n💡 РЕКОМЕНДАЦИИ")
    print("-" * 80)
    
    if anomaly_reduction < 70:
        print("• Рассмотрите ужесточение порогов в config.yaml")
        print("• Проверьте, что auto_tuning.enabled = true")
    
    territory_pct = after_stats['unique_territories'] / 2571 * 100  # ~2571 всего территорий
    if territory_pct > 20:
        print(f"• {territory_pct:.1f}% территорий всё ещё помечены - можно снизить")
    
    if after_stats['critical_pct'] > 30:
        print(f"• {after_stats['critical_pct']:.1f}% критических аномалий - слишком много")
    
    legitimate_count = after_stats['type_distribution'].get('legitimate_pattern', 0)
    if legitimate_count > 0:
        print(f"✅ Фильтр легитимных паттернов работает! Отфильтровано: {legitimate_count}")
    else:
        print("⚠️ Фильтр легитимных паттернов не применён. См. CHANGES_APPLIED.md раздел 3")


def main():
    """Main execution function."""
    print("=" * 80)
    print("🚀 ЗАПУСК ОПТИМИЗИРОВАННОЙ ДЕТЕКЦИИ АНОМАЛИЙ")
    print("=" * 80)
    
    # Step 1: Load baseline (before optimization)
    print("\n📂 ШАГ 1: Загрузка базовых результатов (до оптимизации)")
    print("-" * 80)
    
    # Hardcode the baseline file from before optimization
    baseline_file = Path("output/anomalies_master_20251101_054021.csv")
    if baseline_file.exists():
        print(f"Используем базовый файл: {baseline_file.name}")
        baseline_df = pd.read_csv(baseline_file)
        baseline_stats = analyze_results(baseline_df, "Baseline (До)")
    else:
        print("⚠️ Базовый файл не найден. Будет только анализ новых результатов.")
        baseline_stats = None
    
    # Step 2: Run new analysis
    print("\n🔧 ШАГ 2: Запуск анализа с новыми настройками")
    print("-" * 80)
    print("Запускаем: python main.py")
    print("⏱️ Это может занять 1-2 минуты...")
    print()
    
    response = input("Запустить main.py сейчас? (y/n): ").lower()
    
    if response == 'y':
        try:
            # Run main.py
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=False,
                text=True,
                cwd=Path.cwd()
            )
            
            if result.returncode != 0:
                print(f"❌ Ошибка при запуске main.py (код: {result.returncode})")
                return
            
            print("\n✅ main.py завершён успешно")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return
    else:
        print("⏭️ Пропускаем запуск. Используем существующие результаты.")
    
    # Step 3: Load new results
    print("\n📂 ШАГ 3: Загрузка новых результатов")
    print("-" * 80)
    
    new_df = load_latest_results()
    if new_df is None:
        print("❌ Не удалось загрузить новые результаты")
        return
    
    new_stats = analyze_results(new_df, "После оптимизации")
    
    # Step 4: Compare
    if baseline_stats is not None and new_stats is not None:
        print_comparison(baseline_stats, new_stats)
    else:
        print("\n⚠️ Сравнение недоступно. Показываю только новые результаты:")
        print(f"\nВсего аномалий: {new_stats['total_anomalies']:,}")
        print(f"Уникальных территорий: {new_stats['unique_territories']:,}")
        print(f"Среднее на территорию: {new_stats['avg_per_territory']:.1f}")
        print(f"% критических: {new_stats['critical_pct']:.1f}%")
    
    # Step 5: Save report
    print("\n" + "=" * 80)
    print("💾 Сохранение отчёта...")
    print("=" * 80)
    
    report_path = Path("output") / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ СРАВНЕНИЯ РЕЗУЛЬТАТОВ ДЕТЕКЦИИ АНОМАЛИЙ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if baseline_stats:
            f.write("БАЗОВЫЕ РЕЗУЛЬТАТЫ (ДО ОПТИМИЗАЦИИ):\n")
            f.write(f"  Всего аномалий: {baseline_stats['total_anomalies']:,}\n")
            f.write(f"  Территорий: {baseline_stats['unique_territories']:,}\n\n")
        
        if new_stats:
            f.write("НОВЫЕ РЕЗУЛЬТАТЫ (ПОСЛЕ ОПТИМИЗАЦИИ):\n")
            f.write(f"  Всего аномалий: {new_stats['total_anomalies']:,}\n")
            f.write(f"  Территорий: {new_stats['unique_territories']:,}\n\n")
        
        if baseline_stats and new_stats:
            reduction = (baseline_stats['total_anomalies'] - new_stats['total_anomalies']) / baseline_stats['total_anomalies'] * 100
            f.write(f"СНИЖЕНИЕ АНОМАЛИЙ: {reduction:.1f}%\n")
    
    print(f"✅ Отчёт сохранён: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    print("\nСледующие шаги:")
    print("1. Просмотрите отчёт выше")
    print("2. Проверьте output/anomalies_master_*.csv")
    print("3. При необходимости настройте пороги в config.yaml")
    print("4. Интегрируйте фильтр (см. CHANGES_APPLIED.md раздел 3)")


if __name__ == "__main__":
    main()
