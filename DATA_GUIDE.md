# Data Requirements Guide

## Overview

This document describes all data files required to run the СберИндекс Anomaly Detection System. The system analyzes municipal data from two primary sources: СберИндекс and Росстат (Russian Federal State Statistics Service).

**Important:** All data files are excluded from Git via `.gitignore` for privacy and security reasons. You must obtain and place these files in the appropriate directories before running the analysis.

---

## Required Data Files

### 1. СберИндекс Data (Parquet Format)

Place these files in the **root directory** of the project:

#### `connection.parquet` ⭐ **RECOMMENDED**
- **Description**: Territorial connection graph with real distances between municipalities
- **Size**: ~4.7 million connections
- **Structure**:
  - `territory_id_x` (int) - Source territory ID
  - `territory_id_y` (int) - Target territory ID
  - `distance` (float) - Distance in kilometers
  - `type` (string) - Connection type (e.g., "highway", "railway")
- **Coverage**: 2,630 territories
- **Connection types**: 
  - Highway connections: ~3.3M records
  - Railway connections: ~1.5M records
- **Purpose**: Enables accurate geographic anomaly detection by comparing municipalities with their actual neighbors
- **Optional**: System will fall back to region-based analysis if not present

#### `consumption.parquet` ✅ **REQUIRED**
- **Description**: Consumption data by category over time
- **Size**: ~303,000 records (temporal data)
- **Structure**:
  - `date` (datetime) - Time period
  - `territory_id` (int) - Territory identifier
  - `category` (string) - Consumption category
  - `value` (float) - Consumption value
- **Categories** (6 total):
  - Продовольствие (Food)
  - Маркетплейсы (Marketplaces)
  - Все категории (All categories)
  - [3 additional categories]
- **Temporal structure**: 24-72 time periods per territory
- **Purpose**: Core consumption metrics for anomaly detection

#### `market_access.parquet` ✅ **REQUIRED**
- **Description**: Market accessibility metrics
- **Size**: ~2,600 records (one per territory)
- **Structure**:
  - `territory_id` (int) - Territory identifier
  - `market_access` (float) - Market accessibility score
- **Coverage**: One record per territory
- **Purpose**: Economic accessibility analysis

---

### 2. Росстат Data (Parquet Format)

Place these files in the **`rosstat/`** subdirectory:

#### `rosstat/2_bdmo_population.parquet` ✅ **REQUIRED**
- **Description**: Population statistics over time
- **Size**: ~700,000 records (temporal data)
- **Structure**:
  - `territory_id` (int) - Territory identifier
  - `date` (datetime) - Time period
  - `population` (int) - Population count
  - [Additional demographic fields]
- **Purpose**: Population-based anomaly detection and normalization

#### `rosstat/3_bdmo_migration.parquet` ✅ **REQUIRED**
- **Description**: Migration data over time
- **Size**: ~106,000 records (temporal data)
- **Structure**:
  - `territory_id` (int) - Territory identifier
  - `date` (datetime) - Time period
  - `migration_in` (int) - Incoming migration
  - `migration_out` (int) - Outgoing migration
  - [Additional migration fields]
- **Purpose**: Migration pattern analysis

#### `rosstat/4_bdmo_salary.parquet` ✅ **REQUIRED**
- **Description**: Salary data by industry over time
- **Size**: ~370,000 records (temporal data)
- **Structure**:
  - `territory_id` (int) - Territory identifier
  - `date` (datetime) - Time period
  - `industry` (string) - Industry sector
  - `salary` (float) - Average salary
- **Industries**: 21 different sectors
- **Purpose**: Economic indicator analysis

---

### 3. Municipal Dictionary (Excel Format)

Place these files in the **`t_dict_municipal/`** subdirectory:

#### `t_dict_municipal/t_dict_municipal_districts.xlsx` ✅ **REQUIRED**
- **Description**: Reference dictionary of all municipal territories
- **Size**: 3,101 municipalities
- **Structure**:
  - `territory_id` (int) - Unique territory identifier (primary key)
  - `municipal_district_name_short` (string) - Short municipality name
  - `municipal_district_name_full` (string) - Full municipality name
  - `region_name` (string) - Region name
  - `federal_district` (string) - Federal district
  - [Additional metadata fields]
- **Purpose**: Territory metadata and classification

#### `t_dict_municipal/t_dict_municipal_districts_poly.gpkg` ⚠️ **OPTIONAL**
- **Description**: Geographic boundaries (GeoPackage format)
- **Purpose**: Spatial visualization (not used by core analysis)
- **Note**: Can be omitted if only running statistical analysis

---

## Data Sources

### Where to Obtain Data

#### СберИндекс Data
- **Official source**: СберИндекс Open Data Platform
- **Competition**: "О этот странный, странный мир" (About this strange, strange world)
- **Access**: Contact СберИндекс or competition organizers for data access
- **Format**: Parquet files (optimized columnar format)

#### Росстат Data
- **Official source**: Federal State Statistics Service (Росстат)
- **Website**: https://rosstat.gov.ru/
- **Database**: БДМО (Municipal Statistics Database)
- **Format**: Available in various formats, converted to Parquet for this project
- **Access**: Public data, may require registration

#### Municipal Dictionary
- **Source**: Official administrative territorial division registry
- **Maintained by**: Ministry of Justice of the Russian Federation
- **Format**: Excel (XLSX)

---

## Expected Data Structure

### Unified Dataset After Loading

After loading and merging all data files, the system creates a unified dataset with:

- **Rows**: ~60,308 records (not 3,101!)
  - This is normal due to temporal structure (24-72 periods per territory)
  - Duplicate `territory_id` values are expected and correct
- **Columns**: 37 indicators
  - СберИндекс metrics: consumption categories, market access
  - Росстат metrics: population, migration, salary by industry
  - Metadata: territory names, regions, dates
- **Data completeness**: ~88.8% (some missing values are expected)

### Data Quality Expectations

**Normal characteristics:**
- **Sparsity**: 26 columns have some missing values
- **High sparsity**: 2 columns have >50% missing values (exotic industries)
- **Incomplete territories**: 286 municipalities with <50% data coverage
- **Heterogeneity**: Extreme variation (Moscow vs. remote Chukotka)

**These are not errors** - they reflect the real-world data landscape of Russia's diverse geography and economy.

---

## Data Placement Instructions

### Step-by-Step Setup

1. **Create directory structure** (if not exists):
```bash
mkdir -p rosstat
mkdir -p t_dict_municipal
```

2. **Place СберИндекс files** in root directory:
```
sberindex-anomaly-detection/
├── connection.parquet          # Optional but recommended
├── consumption.parquet         # Required
├── market_access.parquet       # Required
```

3. **Place Росстат files** in `rosstat/` subdirectory:
```
sberindex-anomaly-detection/
└── rosstat/
    ├── 2_bdmo_population.parquet    # Required
    ├── 3_bdmo_migration.parquet     # Required
    └── 4_bdmo_salary.parquet        # Required
```

4. **Place municipal dictionary** in `t_dict_municipal/` subdirectory:
```
sberindex-anomaly-detection/
└── t_dict_municipal/
    ├── t_dict_municipal_districts.xlsx       # Required
    └── t_dict_municipal_districts_poly.gpkg  # Optional
```

5. **Verify file placement**:
```bash
# Check if all required files exist
ls -lh *.parquet
ls -lh rosstat/*.parquet
ls -lh t_dict_municipal/*.xlsx
```

---

## Data Schemas

### СберИндекс Schemas

**connection.parquet:**
```
territory_id_x: int64
territory_id_y: int64
distance: float64
type: string
```

**consumption.parquet:**
```
date: datetime64[ns]
territory_id: int64
category: string
value: float64
```

**market_access.parquet:**
```
territory_id: int64
market_access: float64
```

### Росстат Schemas

**population.parquet:**
```
territory_id: int64
date: datetime64[ns]
population: int64
[additional demographic fields]
```

**migration.parquet:**
```
territory_id: int64
date: datetime64[ns]
migration_in: int64
migration_out: int64
[additional migration fields]
```

**salary.parquet:**
```
territory_id: int64
date: datetime64[ns]
industry: string
salary: float64
```

### Municipal Dictionary Schema

**t_dict_municipal_districts.xlsx:**
```
territory_id: int64 (primary key)
municipal_district_name_short: string
municipal_district_name_full: string
region_name: string
federal_district: string
[additional metadata fields]
```

---

## Data Privacy and Security

### Privacy Notice

⚠️ **IMPORTANT: Data Privacy Requirements**

1. **Never commit data files to Git**
   - All data files are automatically excluded via `.gitignore`
   - The `.gitignore` file includes patterns for `*.parquet`, `*.xlsx`, `*.gpkg`, and `*.csv`
   - Always verify that data files are not staged before committing

2. **Sensitive information**
   - Data files may contain personally identifiable information (PII)
   - Municipal-level data may be considered sensitive
   - Follow your organization's data handling policies

3. **Data sharing**
   - Do not share data files publicly without authorization
   - Do not include data files in project archives or distributions
   - Share only aggregated results and visualizations

4. **Local storage**
   - Store data files securely on your local machine
   - Use appropriate file permissions (read/write for owner only)
   - Consider encrypting sensitive data at rest

### Security Best Practices

1. **Access control**
   - Limit access to data files to authorized personnel only
   - Use secure channels for data transfer
   - Maintain audit logs of data access

2. **Data retention**
   - Delete data files when no longer needed
   - Follow data retention policies
   - Securely wipe files when deleting

3. **Output files**
   - Review output files before sharing
   - Ensure no sensitive data leaks into results
   - Aggregate data appropriately to protect privacy

4. **Version control**
   - Never override `.gitignore` to commit data files
   - Regularly check `git status` to verify no data files are tracked
   - Use `git rm --cached` to remove accidentally committed files

---

## Troubleshooting

### Common Issues

#### Issue: "File not found" errors

**Error message:**
```
Error loading СберИндекс data: [Errno 2] No such file or directory: 'connection.parquet'
```

**Solution:**
1. Verify file exists in the correct location
2. Check file name spelling (case-sensitive on Linux/Mac)
3. Ensure file has correct permissions (readable)

#### Issue: "Invalid Parquet file" errors

**Error message:**
```
ArrowInvalid: Parquet file is corrupted or invalid
```

**Solution:**
1. Re-download the file from the original source
2. Verify file integrity (check file size)
3. Ensure file was not corrupted during transfer

#### Issue: Missing connection.parquet

**Warning message:**
```
WARNING: Connection file not found - using region-based clustering
```

**Solution:**
- This is not an error - the system will continue with reduced accuracy
- Geographic anomaly detection will use regional averages instead of neighbor-based clustering
- To improve accuracy, obtain and place `connection.parquet` in the root directory

#### Issue: High percentage of missing values

**Warning message:**
```
WARNING: 286 municipalities have >70% missing data
```

**Solution:**
- This is expected for some remote territories
- The system handles missing values automatically using pairwise deletion
- Review `docs/missing_value_methodology.md` for details

### Validation

After placing all files, run the system to validate data loading:

```bash
python main.py
```

Expected output:
```
Loading configuration...
Step 1: Loading data...
  ✓ Loaded 3,101 municipalities with 36 indicators
  ✓ Data completeness: 84.1%
```

If you see errors, refer to the troubleshooting section above.

---

## Additional Resources

### Documentation

- **Missing Value Methodology**: `docs/missing_value_methodology.md`
- **Configuration Guide**: `docs/config_migration_guide.md`
- **Project README**: `README.md`

### Support

For questions about:
- **Data access**: Contact СберИндекс or competition organizers
- **System usage**: See `README.md` or open an issue
- **Data privacy**: Consult your organization's data governance team

---

## Summary

### Minimum Required Files (5 files)
1. ✅ `consumption.parquet`
2. ✅ `market_access.parquet`
3. ✅ `rosstat/2_bdmo_population.parquet`
4. ✅ `rosstat/3_bdmo_migration.parquet`
5. ✅ `rosstat/4_bdmo_salary.parquet`
6. ✅ `t_dict_municipal/t_dict_municipal_districts.xlsx`

### Recommended Files (1 file)
7. ⭐ `connection.parquet` - Significantly improves geographic analysis accuracy

### Optional Files (1 file)
8. ⚠️ `t_dict_municipal/t_dict_municipal_districts_poly.gpkg` - For spatial visualization only

---

**Last Updated**: November 2025  
**Version**: 1.0
