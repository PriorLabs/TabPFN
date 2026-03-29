"""
DataLoader class for centralized dataset management.

Provides a unified interface for loading, preprocessing, and managing datasets
across all baseline experiments.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from baseline_config import DATA_PATH, RANDOM_SEED


class DataLoader:
    """Centralized dataset loader for baseline experiments."""
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data_path = None
    
    def load_csv(self, filename: str, verbose: bool = True) -> pd.DataFrame:
        """
        Load CSV file from data directory.
        
        Parameters
        ----------
        filename : str
            Name of CSV file (e.g., 'eudirectlapse.csv')
        verbose : bool
            Print loading information
            
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        csv_path = DATA_PATH / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.data_path = csv_path
        
        if verbose:
            print(f"Loaded {filename}: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def get_target_column(self, target_options: list) -> Optional[str]:
        """
        Intelligently identify target column from list of candidates.
        
        Parameters
        ----------
        target_options : list
            List of candidate target column names
            
        Returns
        -------
        str or None
            Identified target column name, or None if not found
        """
        if self.df is None:
            raise ValueError("Must load data first with load_csv()")
        
        # Try exact matches first
        for col in target_options:
            if col in self.df.columns:
                return col
        
        # Try case-insensitive matches
        cols_lower = {col.lower(): col for col in self.df.columns}
        for opt in target_options:
            if opt.lower() in cols_lower:
                return cols_lower[opt.lower()]
        
        # Try partial matches
        for opt in target_options:
            matches = [col for col in self.df.columns if opt.lower() in col.lower()]
            if matches:
                return matches[0]
        
        return None
    
    def create_binary_target(
        self,
        source_col: str,
        target_name: str = "target",
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Create binary target from continuous or count column.
        
        Parameters
        ----------
        source_col : str
            Name of source column to convert
        target_name : str
            Name for new target column
        threshold : float
            Threshold for creating binary target (value > threshold = 1)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with new binary target column
        """
        if self.df is None:
            raise ValueError("Must load data first")
        
        if source_col not in self.df.columns:
            raise ValueError(f"Column '{source_col}' not found in data")
        
        self.df[target_name] = (self.df[source_col] > threshold).astype(int)
        
        counts = self.df[target_name].value_counts().to_dict()
        print(f"Created binary target '{target_name}':")
        print(f"  Class 0: {counts.get(0, 0)} samples")
        print(f"  Class 1: {counts.get(1, 0)} samples")
        print(f"  Imbalance ratio: {counts.get(1, 1) / counts.get(0, 1):.3f}")
        
        return self.df
    
    def sample_data(self, n_samples: Optional[int] = None, frac: Optional[float] = None) -> pd.DataFrame:
        """
        Randomly sample data for faster iteration.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of rows to sample. If provided, overrides frac.
        frac : float, optional
            Fraction of data to sample (0 < frac <= 1)
            
        Returns
        -------
        pd.DataFrame
            Sampled DataFrame
        """
        if self.df is None:
            raise ValueError("Must load data first")
        
        if n_samples is not None:
            if n_samples >= len(self.df):
                print(f"Sample size ({n_samples}) >= data size ({len(self.df)}), using all data")
                return self.df
            self.df = self.df.sample(n=n_samples, random_state=self.random_state).reset_index(drop=True)
            print(f"Sampled {n_samples} rows (from {len(self.df) + n_samples} total)")
        
        elif frac is not None:
            if not (0 < frac <= 1):
                raise ValueError("frac must be between 0 and 1")
            self.df = self.df.sample(frac=frac, random_state=self.random_state).reset_index(drop=True)
            print(f"Sampled {frac*100:.1f}% of data ({len(self.df)} rows)")
        
        return self.df
    
    def train_test_split(
        self,
        target_col: str,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        target_col : str
            Name of target column
        test_size : float
            Test set fraction (default: 0.2)
        stratify : bool
            Use stratified split to maintain class balance (default: True)
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("Must load data first")
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col].values
        
        stratify_arg = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_arg,
            random_state=self.random_state
        )
        
        print(f"Train/test split:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Get current train/test splits.
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.X_train is None:
            raise ValueError("Must call train_test_split() first")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def describe(self) -> Dict:
        """
        Get summary statistics about the data.
        
        Returns
        -------
        dict
            Summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded")
        
        summary = {
            'n_rows': len(self.df),
            'n_cols': len(self.df.columns),
            'n_numeric': len(self.df.select_dtypes(include=['number']).columns),
            'n_categorical': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'missing_pct': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'memory_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of DataLoader state."""
        status = []
        if self.df is not None:
            status.append(f"Data loaded: {self.df.shape}")
        if self.X_train is not None:
            status.append(f"Train/test split: {self.X_train.shape[0]}/{self.X_test.shape[0]}")
        
        status_str = ", ".join(status) if status else "No data loaded"
        return f"DataLoader({status_str})"


if __name__ == "__main__":
    print("DataLoader class loaded successfully")
    print("Usage: loader = DataLoader(); loader.load_csv('filename.csv')")
