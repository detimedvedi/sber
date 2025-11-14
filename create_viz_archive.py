"""
Создание архива для передачи команде визуализации
Упаковывает все необходимые файлы в ZIP
"""

import zipfile
from pathlib import Path
from datetime import datetime

def create_viz_archive():
    """Создать ZIP архив с файлами для визуализации"""
    
    print("=" * 80)
    print("Создание архива для команды визуализации")
    print("=" * 80)
    
    output_dir = Path('output')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_name = f'viz_package_{timestamp}.zip'
    archive_path = output_dir / archive_name
    
    # Файлы для включения в архив
    files_to_include = [
        # Основные данные
        'VIZ_top50_strangest.csv',
        'VIZ_top20_municipalities.csv',
        'VIZ_temporal_anomalies.csv',
        'VIZ_geographic_contrasts.csv',
        'VIZ_regional_stats.csv',
        'VIZ_anomaly_type_stats.csv',
        'VIZ_summary_metrics.json',
        
        # Документация
        'VIZ_HANDOFF_GUIDE.md',
        'VIZ_QUICK_START.md',
        'ДЛЯ_КОМАНДЫ_ВИЗУАЛИЗАЦИИ.md',
        
        # Готовые визуализации (PNG)
        'viz_*_anomaly_type_distribution.png',
        'viz_*_geographic_heatmap.png',
        'viz_*_severity_distribution.png',
        'viz_*_top_municipalities.png',
        'dashboard_summary_*.png',
    ]
    
    print(f"\n📦 Создание архива: {archive_name}")
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        files_added = 0
        
        for pattern in files_to_include:
            # Поддержка wildcards
            if '*' in pattern:
                matching_files = list(output_dir.glob(pattern))
            else:
                matching_files = [output_dir / pattern]
            
            for file_path in matching_files:
                if file_path.exists() and file_path.is_file():
                    # Добавить файл в архив
                    arcname = file_path.name
                    zipf.write(file_path, arcname)
                    files_added += 1
                    print(f"  ✓ {arcname}")
    
    # Статистика
    archive_size = archive_path.stat().st_size / 1024  # KB
    
    print("\n" + "=" * 80)
    print("✅ Архив создан!")
    print("=" * 80)
    print(f"\n📁 Файл: {archive_path.name}")
    print(f"📊 Размер: {archive_size:.1f} KB")
    print(f"📦 Файлов в архиве: {files_added}")
    print(f"📍 Расположение: {archive_path.absolute()}")
    
    print("\n🚀 Готово к передаче команде визуализации!")
    print("\n💡 Инструкция:")
    print("   1. Отправить файл команде визуализации")
    print("   2. Распаковать архив")
    print("   3. Начать с VIZ_QUICK_START.md")
    
    return archive_path

if __name__ == '__main__':
    archive_path = create_viz_archive()
