import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        # 修改 handle_unknown='ignore'，以处理未知类别
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.categorical_columns = ['manufacturer', 'model', 'gearbox_type', 'fuel_type']
        self.numerical_columns = ['year', 'engine_capacity', 'operating_hours', 'efficiency', 'registration_fees']
        
    def _fill_missing_values(self, df):
        df = df.copy()
        
        # Fill missing values for numerical variables
        for col in self.numerical_columns:
            df[col] = df[col].fillna(df[col].mean())
        
        # Fill missing values for categorical variables
        for col in self.categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df
    
    def _create_engineered_features(self, df):
        df = df.copy()
        
        # Age-related features
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        
        # Usage intensity
        df['hours_per_year'] = np.where(df['vehicle_age'] == 0, 0, 
                                      df['operating_hours'] / df['vehicle_age'].clip(lower=1))
        
        # Cost-related features
        df['registration_cost_per_capacity'] = np.where(df['engine_capacity'] == 0, 0,
                                                      df['registration_fees'] / df['engine_capacity'].clip(lower=1))
        
        # Add engineered features to numerical columns
        self.numerical_columns.extend(['vehicle_age', 'hours_per_year', 
                                     'registration_cost_per_capacity'])
        
        return df
    
    def fit_transform(self, train_df, test_df, validation_split=0.2, random_seed=42):
        """
        Fit preprocessor on training data, split into training and validation sets,
        and transform both training and test data.
        """
        # Create a copy of the DataFrames
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Drop 'id' column if exists
        if 'id' in train_df.columns:
            train_df = train_df.drop('id', axis=1)
        if 'id' in test_df.columns:
            test_df = test_df.drop('id', axis=1)
        
        # Fill missing values and create engineered features
        train_df = self._fill_missing_values(train_df)
        test_df = self._fill_missing_values(test_df)
        train_df = self._create_engineered_features(train_df)
        test_df = self._create_engineered_features(test_df)
        
        # Split train data into training and validation sets
        train_features = train_df.drop(columns=['price'])
        train_labels = train_df['price']
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_features,
            train_labels,
            test_size=validation_split,
            random_state=random_seed
        )
        
        # Process categorical features (One-Hot Encoding)
        self.one_hot_encoder.fit(train_features[self.categorical_columns])  # Fit on all categorical features
        X_train_cat = self.one_hot_encoder.transform(X_train[self.categorical_columns])
        X_val_cat = self.one_hot_encoder.transform(X_val[self.categorical_columns])
        X_test_cat = self.one_hot_encoder.transform(test_df[self.categorical_columns])  # 修正：忽略未知类别
        
        # Standardize numerical features (fit only on training data)
        self.scaler.fit(X_train[self.numerical_columns])  # Fit on training numerical columns
        X_train_num = self.scaler.transform(X_train[self.numerical_columns])
        X_val_num = self.scaler.transform(X_val[self.numerical_columns])
        X_test_num = self.scaler.transform(test_df[self.numerical_columns])
        
        # Combine categorical and numerical features
        X_train_combined = np.hstack([X_train_cat, X_train_num])
        X_val_combined = np.hstack([X_val_cat, X_val_num])
        X_test_combined = np.hstack([X_test_cat, X_test_num])
        
        return X_train_combined, X_val_combined, y_train.values, y_val.values, X_test_combined
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor
        """
        df = df.copy()
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
            
        df = self._fill_missing_values(df)
        df = self._create_engineered_features(df)
        
        # Transform categorical features using the fitted encoder
        cat_features = self.one_hot_encoder.transform(df[self.categorical_columns])  # 修正：忽略未知类别
        
        # Transform numerical features using the scaler fitted on training data
        num_features = self.scaler.transform(df[self.numerical_columns])
        
        return np.hstack([cat_features, num_features])