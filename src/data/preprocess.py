"""Data preprocessing pipeline for Communities and Crime dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_raw_data, get_target_column, get_non_predictive_columns
from src.utils.config import load_config, get_project_root, ensure_dir


class DataPreprocessor:
    """Preprocessing pipeline for crime prediction dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.project_root = get_project_root()
        self.target_col = get_target_column(config_path)
        self.non_predictive = get_non_predictive_columns(config_path)
        
        # Preprocessing parameters from config
        self.missing_threshold = self.config['data']['missing_value_threshold']
        self.correlation_threshold = self.config['data']['correlation_threshold']
        self.random_state = self.config['data']['random_state']
        
        # Features to drop (will be populated during preprocessing)
        self.features_to_drop = []
        
    def remove_non_predictive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-predictive columns."""
        cols_to_remove = [col for col in self.non_predictive if col in df.columns]
        df_cleaned = df.drop(columns=cols_to_remove)
        print(f"Removed {len(cols_to_remove)} non-predictive columns")
        return df_cleaned
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        drop_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values:
        1. Drop columns with missing percentage >= threshold
        2. Impute remaining missing values
        
        Args:
            df: Input DataFrame
            drop_threshold: Threshold for dropping columns (default 0.5)
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        # Calculate missing percentages
        missing_counts = df_cleaned.isnull().sum()
        missing_pct = (missing_counts / len(df_cleaned)) * 100
        
        # Identify columns to drop (>= threshold)
        cols_to_drop = missing_pct[missing_pct >= (drop_threshold * 100)].index.tolist()
        
        # Special case: OtherPerCap has only 1 missing value, don't drop it
        if 'OtherPerCap' in cols_to_drop:
            cols_to_drop.remove('OtherPerCap')
        
        # Drop high missing columns
        if cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            self.features_to_drop.extend(cols_to_drop)
            print(f"Dropped {len(cols_to_drop)} columns with ≥{drop_threshold*100}% missing values")
        
        # Impute OtherPerCap with mean (only 1 missing value)
        if 'OtherPerCap' in df_cleaned.columns and df_cleaned['OtherPerCap'].isnull().any():
            mean_value = df_cleaned['OtherPerCap'].mean()
            df_cleaned['OtherPerCap'].fillna(mean_value, inplace=True)
            print(f"Imputed OtherPerCap with mean value: {mean_value:.4f}")
        
        # Check for any remaining missing values
        remaining_missing = df_cleaned.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"Warning: {remaining_missing} missing values still remain")
            # Optionally impute with median for numeric columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(
                df_cleaned[numeric_cols].median()
            )
            print("Imputed remaining missing values with median")
        
        return df_cleaned
    
    def filter_by_correlation(
        self, 
        df: pd.DataFrame, 
        min_correlation: float = 0.1
    ) -> pd.DataFrame:
        """
        Filter features based on correlation with target variable.
        
        Args:
            df: Input DataFrame
            min_correlation: Minimum absolute correlation to keep (default 0.1)
            
        Returns:
            DataFrame with filtered features
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame")
        
        # Get numeric columns (exclude target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != self.target_col]
        
        # Calculate correlations with target
        correlations = df[feature_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
        
        # Identify features to drop (|correlation| < min_correlation)
        weak_features = correlations[correlations.abs() < min_correlation].index.tolist()
        
        if weak_features:
            df_cleaned = df.drop(columns=weak_features)
            self.features_to_drop.extend(weak_features)
            print(f"Dropped {len(weak_features)} features with |correlation| < {min_correlation}")
        else:
            df_cleaned = df.copy()
            print(f"No features dropped (all have |correlation| >= {min_correlation})")
        
        return df_cleaned
    
    def remove_highly_correlated_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Remove highly correlated feature pairs.
        Keeps one feature from each highly correlated pair.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold (default 0.95)
            
        Returns:
            DataFrame with highly correlated features removed
        """
        df_cleaned = df.copy()
        
        # Get numeric feature columns (exclude target)
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != self.target_col]
        
        # Calculate correlation matrix
        corr_matrix = df_cleaned[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        # Define which features to keep (based on EDA decisions)
        features_to_remove = {
            'numbUrban',  # Keep population
            'medFamInc',  # Keep medIncome
            'whitePerCap',  # Keep perCapInc
            'PctOccupMgmtProf',  # Keep PctBSorMore
            'MalePctDivorce',  # Keep TotalPctDiv
            'FemalePctDiv',  # Keep TotalPctDiv
            'PctFam2Par',  # Keep PctKids2Par
        }
        
        # Remove features
        cols_to_remove = [col for col in features_to_remove if col in df_cleaned.columns]
        
        if cols_to_remove:
            df_cleaned = df_cleaned.drop(columns=cols_to_remove)
            self.features_to_drop.extend(cols_to_remove)
            print(f"Removed {len(cols_to_remove)} highly correlated features")
            print(f"  Removed: {cols_to_remove}")
        
        return df_cleaned
    
    def split_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != self.target_col]
        X = df[feature_cols]
        y = df[self.target_col]
        
        # First split: train+val (85%) and test (15%)
        test_size = self.config['data']['test_size']
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=self.random_state
        )
        
        # Second split: train (80%) and val (15% of original)
        val_size = self.config['data']['val_size']
        # Adjust val_size for the train_val split
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=self.random_state
        )
        
        # Combine back into DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        print(f"\nData split:")
        print(f"  Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def preprocess(
        self, 
        df: Optional[pd.DataFrame] = None,
        save_processed: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame (loads from file if None)
            save_processed: Whether to save processed data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Load data if not provided
        if df is None:
            print("Loading raw data...")
            df = load_raw_data()
        
        print(f"\nStarting preprocessing pipeline...")
        print(f"Initial shape: {df.shape}")
        
        # Step 1: Remove non-predictive columns
        print("\n[Step 1] Removing non-predictive columns...")
        df = self.remove_non_predictive_columns(df)
        
        # Step 2: Handle missing values
        print("\n[Step 2] Handling missing values...")
        df = self.handle_missing_values(df, drop_threshold=self.missing_threshold)
        
        # Step 3: Filter by correlation
        print("\n[Step 3] Filtering features by correlation with target...")
        df = self.filter_by_correlation(df, min_correlation=0.1)
        
        # Step 4: Remove highly correlated features
        print("\n[Step 4] Removing highly correlated feature pairs...")
        df = self.remove_highly_correlated_features(df, threshold=self.correlation_threshold)
        
        print(f"\nFinal shape: {df.shape}")
        print(f"Total features dropped: {len(self.features_to_drop)}")
        
        # Step 5: Split data
        print("\n[Step 5] Splitting data into train/val/test...")
        train_df, val_df, test_df = self.split_data(df)
        
        # Save processed data
        if save_processed:
            self._save_processed_data(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _save_processed_data(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Save processed data to disk."""
        processed_dir = self.project_root / self.config['data']['processed_dir']
        ensure_dir(processed_dir)
        
        # Save as parquet (efficient) and CSV (readable)
        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        val_df.to_parquet(processed_dir / "val.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)
        
        train_df.to_csv(processed_dir / "train.csv", index=False)
        val_df.to_csv(processed_dir / "val.csv", index=False)
        test_df.to_csv(processed_dir / "test.csv", index=False)
        
        print(f"\nProcessed data saved to: {processed_dir}")
        print(f"  - train.parquet/csv ({len(train_df)} rows)")
        print(f"  - val.parquet/csv ({len(val_df)} rows)")
        print(f"  - test.parquet/csv ({len(test_df)} rows)")


def main():
    """Main function to run preprocessing."""
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.preprocess()
    
    print("\n✅ Preprocessing complete!")
    print(f"\nFeature columns: {len([c for c in train_df.columns if c != preprocessor.target_col])}")
    print(f"Target column: {preprocessor.target_col}")


if __name__ == "__main__":
    main()
