"""
Legitimate Pattern Filter Module

This module provides functionality to filter out legitimate patterns that should not
be flagged as logical inconsistencies. Examples include tourist zones, shift-work
settlements, and industrial centers with naturally high consumption-to-population ratios.
"""

import logging
from typing import Dict, List, Set, Any, Optional
import yaml
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class LegitimatePatternFilter:
    """
    Filter for identifying and excluding legitimate patterns from anomaly detection.
    
    Legitimate patterns include:
    - Tourist zones (high consumption, low permanent population)
    - Shift-work settlements (transient workers)
    - Industrial centers (corporate consumption)
    - Border territories (transit consumption)
    - Remote territories (extreme logistics)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the legitimate pattern filter.
        
        Args:
            config_path: Path to legitimate patterns configuration file.
                        If None, uses default 'legitimate_patterns_config.yaml'
        """
        self.config_path = config_path or "legitimate_patterns_config.yaml"
        self.config = self._load_config()
        self.whitelist = self._build_whitelist()
        self.legitimate_indicators = set(
            self.config.get('legitimate_indicator_patterns', [])
        )
        logger.info(f"Loaded {len(self.whitelist)} territories in whitelist")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.warning(
                f"Legitimate patterns config not found at {self.config_path}. "
                "Using empty whitelist."
            )
            return {
                'legitimate_patterns': {},
                'legitimate_indicator_patterns': [],
                'logical_consistency_thresholds': {}
            }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded legitimate patterns config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {
                'legitimate_patterns': {},
                'legitimate_indicator_patterns': [],
                'logical_consistency_thresholds': {}
            }
    
    def _build_whitelist(self) -> Dict[str, str]:
        """
        Build a whitelist dictionary mapping territory names to their categories.
        
        Returns:
            Dictionary: {territory_name: category_reason}
        """
        whitelist = {}
        patterns = self.config.get('legitimate_patterns', {})
        
        for category, category_data in patterns.items():
            reason = category_data.get('reason', 'Legitimate pattern')
            territories = category_data.get('territories', [])
            
            for territory in territories:
                whitelist[territory] = {
                    'category': category,
                    'reason': reason
                }
        
        return whitelist
    
    def is_legitimate_pattern(
        self,
        municipal_name: str,
        indicator: str
    ) -> Optional[Dict[str, str]]:
        """
        Check if a municipality-indicator combination is a legitimate pattern.
        
        Args:
            municipal_name: Name of the municipality
            indicator: Name of the indicator
            
        Returns:
            Dictionary with category and reason if legitimate, None otherwise
        """
        # Check if territory is in whitelist
        if municipal_name in self.whitelist:
            # Check if indicator is commonly legitimate
            if any(ind in indicator for ind in self.legitimate_indicators):
                return self.whitelist[municipal_name]
            # Even if not common indicator, still legitimate for this territory
            return self.whitelist[municipal_name]
        
        # Check if indicator is commonly legitimate (even for non-whitelisted territories)
        # Use more lenient threshold
        if any(ind in indicator for ind in self.legitimate_indicators):
            # Check if municipal name contains keywords
            remote_keywords = ['ский', 'ский район', 'улус', 'округ']
            if any(keyword in municipal_name for keyword in remote_keywords):
                return {
                    'category': 'likely_remote',
                    'reason': 'Potentially remote territory with legitimate pattern'
                }
        
        return None
    
    def filter_anomalies(
        self,
        anomalies_df: pd.DataFrame,
        anomaly_type: str = 'logical_inconsistency'
    ) -> pd.DataFrame:
        """
        Filter out legitimate patterns from detected anomalies.
        
        Args:
            anomalies_df: DataFrame with detected anomalies
            anomaly_type: Type of anomalies to filter (default: logical_inconsistency)
            
        Returns:
            DataFrame with legitimate patterns flagged via is_legitimate_pattern column
        """
        if anomalies_df.empty:
            return anomalies_df
        
        # Initialize is_legitimate_pattern column
        anomalies_df['is_legitimate_pattern'] = False
        anomalies_df['legitimate_pattern_category'] = None
        
        # Filter only specified anomaly type
        mask = anomalies_df['anomaly_type'] == anomaly_type
        filtered_count = 0
        
        # Check each anomaly
        for idx, row in anomalies_df[mask].iterrows():
            pattern_info = self.is_legitimate_pattern(
                row['municipal_name'],
                row['indicator']
            )
            
            if pattern_info:
                # Mark as legitimate pattern with flag
                anomalies_df.at[idx, 'is_legitimate_pattern'] = True
                anomalies_df.at[idx, 'legitimate_pattern_category'] = pattern_info['category']
                anomalies_df.at[idx, 'severity_score'] = 20.0  # Low severity
                anomalies_df.at[idx, 'description'] = (
                    f"[LEGITIMATE PATTERN - {pattern_info['category']}] "
                    f"{row['description']}"
                )
                anomalies_df.at[idx, 'potential_explanation'] = pattern_info['reason']
                filtered_count += 1
        
        if filtered_count > 0:
            logger.info(
                f"Reclassified {filtered_count} logical inconsistencies as "
                f"legitimate patterns"
            )
        
        return anomalies_df
    
    def get_whitelist_summary(self) -> pd.DataFrame:
        """
        Get a summary of whitelisted territories.
        
        Returns:
            DataFrame with columns: territory, category, reason
        """
        data = []
        for territory, info in self.whitelist.items():
            data.append({
                'territory': territory,
                'category': info['category'],
                'reason': info['reason']
            })
        
        return pd.DataFrame(data)
    
    def add_territory_to_whitelist(
        self,
        territory_name: str,
        category: str,
        reason: str
    ):
        """
        Add a territory to the whitelist at runtime.
        
        Args:
            territory_name: Name of the territory
            category: Category (e.g., 'tourist_zones', 'industrial_centers')
            reason: Explanation for why this is legitimate
        """
        self.whitelist[territory_name] = {
            'category': category,
            'reason': reason
        }
        logger.info(f"Added {territory_name} to whitelist as {category}")
    
    def save_whitelist(self, output_path: str = "output/whitelist_applied.yaml"):
        """
        Save current whitelist to YAML file for documentation.
        
        Args:
            output_path: Path to save the whitelist
        """
        # Reorganize whitelist by category
        by_category = {}
        for territory, info in self.whitelist.items():
            category = info['category']
            if category not in by_category:
                by_category[category] = {
                    'reason': info['reason'],
                    'territories': []
                }
            by_category[category]['territories'].append(territory)
        
        # Save to file
        output_data = {
            'legitimate_patterns': by_category,
            'total_territories': len(self.whitelist),
            'categories': list(by_category.keys())
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)
        
        logger.info(f"Saved whitelist to {output_path}")


def demo_usage():
    """Demonstrate usage of the LegitimatePatternFilter."""
    # Initialize filter
    filter = LegitimatePatternFilter()
    
    # Test some territories
    test_cases = [
        ("Сочи", "consumption_Общественное питание"),
        ("Норильск", "consumption_Продовольствие"),
        ("Москва", "consumption_Маркетплейсы"),
        ("Билибинский", "market_access vs population_total"),
    ]
    
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ФИЛЬТРА ЛЕГИТИМНЫХ ПАТТЕРНОВ")
    print("=" * 70)
    
    for territory, indicator in test_cases:
        result = filter.is_legitimate_pattern(territory, indicator)
        if result:
            print(f"\n✓ {territory} - {indicator}")
            print(f"  Категория: {result['category']}")
            print(f"  Причина: {result['reason']}")
        else:
            print(f"\n✗ {territory} - {indicator}")
            print(f"  Не в whitelist - будет помечен как аномалия")
    
    # Show whitelist summary
    print("\n" + "=" * 70)
    print("СТАТИСТИКА WHITELIST")
    print("=" * 70)
    summary = filter.get_whitelist_summary()
    print(f"Всего территорий в whitelist: {len(summary)}")
    print(f"\nРаспределение по категориям:")
    print(summary['category'].value_counts())


if __name__ == "__main__":
    demo_usage()
