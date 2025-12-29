import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_is_fitted
    from typing import List, Dict, Optional, Union

    class CardinalityAwareOrdinalEncoder(BaseEstimator, TransformerMixin):
        """
        Custom ordinal encoder that applies different strategies based on cardinality.
    
        Similar to the clustering preprocessing strategy but outputs ordinal encodings
        suitable for sklearn pipelines.
        """
    
        def __init__(
            self, 
            low_card_threshold: int = 5,
            medium_card_threshold: int = 20,
            min_frequency_pct: float = 0.005,  # 0.5%
            min_absolute_count: int = 3,
            high_card_top_n: int = 15,
            high_card_freq_pct: float = 0.01,  # 1%
            handle_unknown: str = 'use_encoded_value',
            unknown_value: int = -1,
            dtype: type = np.int64
        ):
            """
            Parameters:
            -----------
            low_card_threshold : int, default=5
                Categories with unique values <= this are kept as-is
            medium_card_threshold : int, default=20  
                Threshold between medium and high cardinality
            min_frequency_pct : float, default=0.005
                Minimum frequency percentage for medium cardinality categories
            min_absolute_count : int, default=3
                Minimum absolute count for medium cardinality categories
            high_card_top_n : int, default=15
                Number of top categories to keep for high cardinality
            high_card_freq_pct : float, default=0.01
                Minimum frequency percentage for high cardinality categories
            handle_unknown : str, default='use_encoded_value'
                How to handle unknown categories during transform
            unknown_value : int, default=-1
                Value to use for unknown categories
            dtype : type, default=np.int64
                Output data type
            """
            self.low_card_threshold = low_card_threshold
            self.medium_card_threshold = medium_card_threshold
            self.min_frequency_pct = min_frequency_pct
            self.min_absolute_count = min_absolute_count
            self.high_card_top_n = high_card_top_n
            self.high_card_freq_pct = high_card_freq_pct
            self.handle_unknown = handle_unknown
            self.unknown_value = unknown_value
            self.dtype = dtype
        
        def fit(self, X, y=None):
            """
            Fit the encoder to the training data.
        
            Parameters:
            -----------
            X : array-like of shape (n_samples, n_features)
                Training data
            y : ignored
        
            Returns:
            --------
            self : object
            """
            X = self._validate_input(X)
            n_samples = len(X)
        
            self.categories_ = {}
            self.strategies_ = {}
            self.encodings_ = {}
        
            for col_idx, col_name in enumerate(X.columns):
                n_unique = X[col_name].nunique()
                counts = X[col_name].value_counts()
            
                # Determine strategy based on cardinality
                if n_unique <= self.low_card_threshold:
                    # Low cardinality: keep all categories
                    keep_categories = counts.index.tolist()
                    self.strategies_[col_name] = 'low_cardinality'
                
                elif n_unique <= self.medium_card_threshold:
                    # Medium cardinality: conservative filtering
                    min_count = max(self.min_absolute_count, 
                                  self.min_frequency_pct * n_samples)
                    keep_categories = counts[counts >= min_count].index.tolist()
                    self.strategies_[col_name] = 'medium_cardinality'
                
                else:
                    # High cardinality: more aggressive filtering
                    top_n = counts.head(self.high_card_top_n).index
                    freq_threshold = counts[counts >= self.high_card_freq_pct * n_samples].index
                    keep_categories = list(set(top_n) | set(freq_threshold))
                    self.strategies_[col_name] = 'high_cardinality'
            
                # Store categories and create ordinal encoding
                self.categories_[col_name] = keep_categories
            
                # Create encoding mapping (category -> integer)
                # Reserve 0 for 'other' category if needed
                if len(keep_categories) < n_unique:
                    # We'll have an 'other' category
                    encoding_dict = {cat: i+1 for i, cat in enumerate(keep_categories)}
                    encoding_dict['__OTHER__'] = 0
                else:
                    encoding_dict = {cat: i for i, cat in enumerate(keep_categories)}
            
                self.encodings_[col_name] = encoding_dict
        
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = X.columns.tolist()
        
            return self
    
        def transform(self, X):
            """
            Transform the input data using fitted encodings.
        
            Parameters:
            -----------
            X : array-like of shape (n_samples, n_features)
                Data to transform
            
            Returns:
            --------
            X_transformed : ndarray of shape (n_samples, n_features)
                Transformed data with ordinal encodings
            """
            check_is_fitted(self)
            X = self._validate_input(X)
        
            X_transformed = np.empty((X.shape[0], X.shape[1]), dtype=self.dtype)
        
            for col_idx, col_name in enumerate(X.columns):
                col_data = X[col_name].copy()
                encoding_dict = self.encodings_[col_name]
            
                # Handle categories not seen during fit
                if '__OTHER__' in encoding_dict:
                    # Map unseen categories to 'other'
                    mask_known = col_data.isin(self.categories_[col_name])
                    col_data = col_data.where(mask_known, '__OTHER__')
            
                # Apply encoding
                if self.handle_unknown == 'use_encoded_value':
                    encoded_col = col_data.map(encoding_dict).fillna(self.unknown_value)
                else:
                    encoded_col = col_data.map(encoding_dict)
                    if encoded_col.isna().any():
                        raise ValueError(f"Unknown categories found in column {col_name}")
            
                X_transformed[:, col_idx] = encoded_col.astype(self.dtype)
        
            return X_transformed
    
        def inverse_transform(self, X):
            """
            Convert ordinal encodings back to original categories.
        
            Parameters:
            -----------
            X : array-like of shape (n_samples, n_features)
                Encoded data to inverse transform
            
            Returns:
            --------
            X_original : DataFrame
                Data with original categories
            """
            check_is_fitted(self)
        
            if hasattr(X, 'shape'):
                X = np.asarray(X)
            else:
                X = np.array(X)
            
            X_original = pd.DataFrame(index=range(X.shape[0]), 
                                    columns=self.feature_names_in_)
        
            for col_idx, col_name in enumerate(self.feature_names_in_):
                # Create reverse mapping (integer -> category)
                reverse_encoding = {v: k for k, v in self.encodings_[col_name].items()}
            
                # Apply inverse transform
                col_data = pd.Series(X[:, col_idx])
                X_original[col_name] = col_data.map(reverse_encoding)
            
                # Replace '__OTHER__' with 'other' for readability
                X_original[col_name] = X_original[col_name].replace('__OTHER__', 'other')
        
            return X_original
    
        def get_feature_names_out(self, input_features=None):
            """Get output feature names."""
            check_is_fitted(self)
            if input_features is None:
                return np.array(self.feature_names_in_)
            else:
                return np.array(input_features)
    
        def _validate_input(self, X):
            """Validate and convert input to DataFrame."""
            if not isinstance(X, pd.DataFrame):
                if hasattr(X, 'shape') and len(X.shape) == 2:
                    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    raise ValueError("Input must be a 2D array-like structure")
            return X
    
        def get_cardinality_info(self):
            """Get information about cardinality strategies used for each feature."""
            check_is_fitted(self)
            info = {}
            for col_name in self.feature_names_in_:
                info[col_name] = {
                    'strategy': self.strategies_[col_name],
                    'original_categories': len(self.categories_[col_name]),
                    'encoded_categories': len(self.encodings_[col_name]),
                    'has_other_category': '__OTHER__' in self.encodings_[col_name]
                }
            return info


    # Example usage and demonstration
    if __name__ == "__main__":
        # Create sample data with different cardinalities
        np.random.seed(42)
        n_samples = 1000
    
        data = {
            # Low cardinality (3 categories)
            'grade': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.4, 0.2]),
        
            # Medium cardinality (12 categories) 
            'city': np.random.choice([f'City_{i}' for i in range(12)], n_samples,
                                    p=[0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04, 0.05]),
        
            # High cardinality (50 categories)
            'product': np.random.choice([f'Product_{i}' for i in range(50)], n_samples)
        }
    
        df = pd.DataFrame(data)
    
        # Initialize and fit the encoder
        encoder = CardinalityAwareOrdinalEncoder()
        encoder.fit(df)
    
        # Transform the data
        df_encoded = encoder.transform(df)
        print("Encoded data shape:", df_encoded.shape)
        print("Encoded data sample:")
        print(df_encoded[:5])
    
        # Get cardinality information
        print("\nCardinality strategies:")
        for col, info in encoder.get_cardinality_info().items():
            print(f"{col}: {info}")
    
        # Inverse transform
        df_recovered = encoder.inverse_transform(df_encoded)
        print("\nRecovered data sample:")
        print(df_recovered.head())
    
        # Test with new data containing unknown categories
        new_data = pd.DataFrame({
            'grade': ['A', 'D'],  # 'D' is unknown
            'city': ['City_0', 'City_Unknown'],  # 'City_Unknown' is unknown
            'product': ['Product_0', 'Product_Unknown']  # 'Product_Unknown' is unknown
        })
    
        new_encoded = encoder.transform(new_data)
        print(f"\nNew data encoded:")
        print(new_encoded)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
