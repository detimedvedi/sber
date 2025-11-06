import pandas as pd

# Загружаем последний файл с аномалиями
df = pd.read_csv('output/anomalies_master_20251104_021837.csv')

print("=" * 70)
print("АНАЛИЗ ОБНАРУЖЕННЫХ АНОМАЛИЙ")
print("=" * 70)

print(f"\nВсего аномалий обнаружено: {len(df)}")
print(f"Уникальных муниципалитетов затронуто: {df['territory_id'].nunique()}")

print("\n" + "=" * 70)
print("РАСПРЕДЕЛЕНИЕ ПО ТИПАМ АНОМАЛИЙ")
print("=" * 70)
type_counts = df['anomaly_type'].value_counts()
for anom_type, count in type_counts.items():
    pct = (count / len(df) * 100)
    print(f"{anom_type:35s}: {count:5d} ({pct:5.1f}%)")

print("\n" + "=" * 70)
print("СРЕДНЯЯ СЕРЬЕЗНОСТЬ ПО ТИПАМ")
print("=" * 70)
severity_by_type = df.groupby('anomaly_type')['severity_score'].agg(['mean', 'max', 'count'])
for idx, row in severity_by_type.iterrows():
    print(f"{idx:35s}: среднее={row['mean']:5.1f}, макс={row['max']:5.1f}, кол-во={int(row['count'])}")

print("\n" + "=" * 70)
print("ТОП-10 МУНИЦИПАЛИТЕТОВ С НАИБОЛЬШИМ ЧИСЛОМ АНОМАЛИЙ")
print("=" * 70)
top_muni = df.groupby(['municipal_name', 'region_name']).size().sort_values(ascending=False).head(10)
for (muni, region), count in top_muni.items():
    print(f"{muni:30s} ({region:20s}): {count:3d} аномалий")

print("\n" + "=" * 70)
print("РАСПРЕДЕЛЕНИЕ АНОМАЛИЙ ПО ИНДИКАТОРАМ (топ-15)")
print("=" * 70)
top_indicators = df['indicator'].value_counts().head(15)
for indicator, count in top_indicators.items():
    pct = (count / len(df) * 100)
    print(f"{indicator:40s}: {count:4d} ({pct:4.1f}%)")

print("\n" + "=" * 70)
print("КРИТИЧЕСКИЕ АНОМАЛИИ (severity > 80)")
print("=" * 70)
critical = df[df['severity_score'] > 80]
print(f"Критических аномалий: {len(critical)} ({len(critical)/len(df)*100:.1f}%)")
if len(critical) > 0:
    print("\nПримеры критических аномалий:")
    for idx, row in critical.head(5).iterrows():
        print(f"  - {row['municipal_name']} ({row['region_name']})")
        print(f"    Индикатор: {row['indicator']}")
        print(f"    Тип: {row['anomaly_type']}, Серьезность: {row['severity_score']:.1f}")
        print(f"    Описание: {row['description'][:80]}...")
        print()
