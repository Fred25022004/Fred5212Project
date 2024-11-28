import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
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
    
    def fit_transform(self, train_df, test_df):
        """
        Fit preprocessor on training data and transform both training and test data
        """
        # Create a copy of the DataFrames
        train_df = train_df.copy()
        test_df = test_df.copy()
        
        # Drop 'id' column if exists
        if 'id' in train_df.columns:
            train_df = train_df.drop('id', axis=1)
        if 'id' in test_df.columns:
            test_df = test_df.drop('id', axis=1)
        
        # Combine train and test for preprocessing
        combined_data = pd.concat([train_df, test_df], ignore_index=True)
        combined_data = self._fill_missing_values(combined_data)
        combined_data = self._create_engineered_features(combined_data)
        
        # Process categorical features (One-Hot Encoding)
        cat_features = self.one_hot_encoder.fit_transform(combined_data[self.categorical_columns])
        
        # Split combined data back into train and test
        combined_data_train = combined_data.iloc[:len(train_df), :]
        combined_data_test = combined_data.iloc[len(train_df):, :]
        
        # Standardize numerical features (fit only on training data)
        self.scaler.fit(combined_data_train[self.numerical_columns])  # Fit on training data
        num_features_train = self.scaler.transform(combined_data_train[self.numerical_columns])  # Transform training data
        num_features_test = self.scaler.transform(combined_data_test[self.numerical_columns])  # Transform test data
        
        # Combine categorical and numerical features
        X_train = np.hstack([cat_features[:len(train_df)], num_features_train])
        X_test = np.hstack([cat_features[len(train_df):], num_features_test])
        
        return X_train, X_test
    
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
        cat_features = self.one_hot_encoder.transform(df[self.categorical_columns])
        
        # Transform numerical features using the scaler fitted on training data
        num_features = self.scaler.transform(df[self.numerical_columns])
        
        return np.hstack([cat_features, num_features])