"""Feature engineering utilities for crime prediction dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_project_root
from src.data.load_data import get_target_column


class FeatureEngineer:
    """Feature engineering utilities."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.target_col = get_target_column(config_path)
    
    def categorize_features(self, df: pd.DataFrame) -> dict:
        """
        Categorize features into groups based on naming patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with feature categories
        """
        categories = {
            'demographic': [],
            'socio_economic': [],
            'law_enforcement': [],
            'housing': [],
            'family': [],
            'other': []
        }
        
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        for col in feature_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['race', 'pctwhite', 'pctblack', 'pctasian', 'pcthisp', 'age']):
                categories['demographic'].append(col)
            elif any(x in col_lower for x in ['income', 'pov', 'employ', 'educ', 'wage', 'income', 'unemploy']):
                categories['socio_economic'].append(col)
            elif any(x in col_lower for x in ['polic', 'lemas', 'offic', 'drug', 'gang']):
                categories['law_enforcement'].append(col)
            elif any(x in col_lower for x in ['hous', 'rent', 'own']):
                categories['housing'].append(col)
            elif any(x in col_lower for x in ['fam', 'kid', 'teen', 'div', 'illeg', 'par']):
                categories['family'].append(col)
            else:
                categories['other'].append(col)
        
        return categories
    
    def create_interaction_features(
        self, 
        df: pd.DataFrame,
        feature_pairs: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """
        Create interaction features (multiplication of feature pairs).
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (feature1, feature2) tuples to create interactions.
                         If None, uses top correlated features with target.
            
        Returns:
            DataFrame with interaction features added
        """
        df_engineered = df.copy()
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        if feature_pairs is None:
            # Auto-select top features for interactions
            if self.target_col in df.columns:
                correlations = df[feature_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
                top_features = correlations.abs().nlargest(10).index.tolist()
                
                # Create interactions between top features
                feature_pairs = []
                for i, feat1 in enumerate(top_features[:5]):
                    for feat2 in top_features[i+1:6]:
                        feature_pairs.append((feat1, feat2))
        
        # Create interaction features
        interaction_count = 0
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_engineered[interaction_name] = df_engineered[feat1] * df_engineered[feat2]
                interaction_count += 1
        
        if interaction_count > 0:
            print(f"Created {interaction_count} interaction features")
        
        return df_engineered
    
    def create_ratio_features(
        self, 
        df: pd.DataFrame,
        ratio_pairs: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """
        Create ratio features (division of feature pairs).
        
        Args:
            df: Input DataFrame
            ratio_pairs: List of (numerator, denominator) tuples.
                        If None, uses domain knowledge to create meaningful ratios.
            
        Returns:
            DataFrame with ratio features added
        """
        df_engineered = df.copy()
        
        if ratio_pairs is None:
            # Create meaningful ratios based on domain knowledge
            ratio_pairs = []
            
            # Income ratios
            if 'medIncome' in df.columns and 'perCapInc' in df.columns:
                ratio_pairs.append(('medIncome', 'perCapInc'))
            
            # Poverty and income
            if 'PctPopUnderPov' in df.columns and 'medIncome' in df.columns:
                ratio_pairs.append(('PctPopUnderPov', 'medIncome'))
            
            # Family structure ratios
            if 'PctKids2Par' in df.columns and 'PctFam2Par' in df.columns:
                ratio_pairs.append(('PctKids2Par', 'PctFam2Par'))
        
        # Create ratio features
        ratio_count = 0
        for numerator, denominator in ratio_pairs:
            if numerator in df.columns and denominator in df.columns:
                # Avoid division by zero
                ratio_name = f"{numerator}_div_{denominator}"
                df_engineered[ratio_name] = df_engineered[numerator] / (
                    df_engineered[denominator] + 1e-8  # Small epsilon to avoid division by zero
                )
                ratio_count += 1
        
        if ratio_count > 0:
            print(f"Created {ratio_count} ratio features")
        
        return df_engineered
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features (squared, cubed, etc.).
        
        Args:
            df: Input DataFrame
            features: List of features to create polynomials for.
                     If None, uses top correlated features.
            degree: Polynomial degree (default 2 for squared features)
            
        Returns:
            DataFrame with polynomial features added
        """
        df_engineered = df.copy()
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        if features is None:
            # Auto-select top features
            if self.target_col in df.columns:
                correlations = df[feature_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
                features = correlations.abs().nlargest(5).index.tolist()
        
        # Create polynomial features
        poly_count = 0
        for feat in features:
            if feat in df.columns:
                for d in range(2, degree + 1):
                    poly_name = f"{feat}_pow{d}"
                    df_engineered[poly_name] = df_engineered[feat] ** d
                    poly_count += 1
        
        if poly_count > 0:
            print(f"Created {poly_count} polynomial features (degree {degree})")
        
        return df_engineered
    
    def create_binned_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_bins: int = 5
    ) -> pd.DataFrame:
        """
        Create binned/categorical features from continuous features.
        
        Args:
            df: Input DataFrame
            features: List of features to bin. If None, uses top correlated features.
            n_bins: Number of bins (default 5)
            
        Returns:
            DataFrame with binned features added
        """
        df_engineered = df.copy()
        feature_cols = [col for col in df.columns if col != self.target_col]
        
        if features is None:
            # Auto-select top features
            if self.target_col in df.columns:
                correlations = df[feature_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
                features = correlations.abs().nlargest(5).index.tolist()
        
        # Create binned features
        bin_count = 0
        for feat in features:
            if feat in df.columns:
                bin_name = f"{feat}_binned"
                df_engineered[bin_name] = pd.cut(
                    df_engineered[feat],
                    bins=n_bins,
                    labels=False,
                    duplicates='drop'
                )
                bin_count += 1
        
        if bin_count > 0:
            print(f"Created {bin_count} binned features ({n_bins} bins each)")
        
        return df_engineered
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        create_interactions: bool = False,
        create_ratios: bool = False,
        create_polynomials: bool = False,
        create_bins: bool = False
    ) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            df: Input DataFrame
            create_interactions: Whether to create interaction features
            create_ratios: Whether to create ratio features
            create_polynomials: Whether to create polynomial features
            create_bins: Whether to create binned features
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        print(f"Starting feature engineering...")
        print(f"Initial shape: {df_engineered.shape}")
        
        if create_interactions:
            print("\nCreating interaction features...")
            df_engineered = self.create_interaction_features(df_engineered)
        
        if create_ratios:
            print("\nCreating ratio features...")
            df_engineered = self.create_ratio_features(df_engineered)
        
        if create_polynomials:
            print("\nCreating polynomial features...")
            df_engineered = self.create_polynomial_features(df_engineered)
        
        if create_bins:
            print("\nCreating binned features...")
            df_engineered = self.create_binned_features(df_engineered)
        
        print(f"\nFinal shape: {df_engineered.shape}")
        print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
        
        return df_engineered


def main():
    """Main function for testing feature engineering."""
    import pandas as pd
    from src.data.preprocess import DataPreprocessor
    
    # Load processed data
    preprocessor = DataPreprocessor()
    train_df, _, _ = preprocessor.preprocess(save_processed=False)
    
    # Test feature engineering
    engineer = FeatureEngineer()
    
    # Categorize features
    categories = engineer.categorize_features(train_df)
    print("\nFeature Categories:")
    for category, features in categories.items():
        print(f"  {category}: {len(features)} features")
    
    # Example: Create interaction features
    print("\n" + "="*60)
    df_engineered = engineer.engineer_features(
        train_df,
        create_interactions=True,
        create_ratios=False,
        create_polynomials=False,
        create_bins=False
    )


if __name__ == "__main__":
    main()
