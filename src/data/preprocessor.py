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
            df[col] = df[col].interpolate(method='linear')
        
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
        
        # Efficiency-related features
        df['power_efficiency'] = np.where(df['engine_capacity'] == 0, 0,
                                        df['efficiency'] / df['engine_capacity'].clip(lower=1))
        
        # Cost-related features
        df['registration_cost_per_capacity'] = np.where(df['engine_capacity'] == 0, 0,
                                                      df['registration_fees'] / df['engine_capacity'].clip(lower=1))
        
        # Add engineered features to numerical columns
        self.numerical_columns.extend(['vehicle_age', 'hours_per_year', 
                                     'power_efficiency', 'registration_cost_per_capacity'])
        
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
        
        # Process categorical features
        cat_features = self.one_hot_encoder.fit_transform(combined_data[self.categorical_columns])
        
        # Process numerical features
        num_features = self.scaler.fit_transform(combined_data[self.numerical_columns])
        
        # Combine all features
        all_features = np.hstack([cat_features, num_features])
        
        # Split back into train and test
        X_train = all_features[:len(train_df)]
        X_test = all_features[len(train_df):]
        
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
        
        cat_features = self.one_hot_encoder.transform(df[self.categorical_columns])
        num_features = self.scaler.transform(df[self.numerical_columns])
        
        return np.hstack([cat_features, num_features])