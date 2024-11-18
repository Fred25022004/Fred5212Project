import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        
    def fit_transform(self, train_data, test_data):
        # Handle missing values
        num_cols = train_data.select_dtypes(include=[np.number]).columns
        num_cols = num_cols.drop('price')
        
        train_data[num_cols] = train_data[num_cols].interpolate(method='linear')
        test_data[num_cols] = test_data[num_cols].interpolate(method='linear')
        
        cat_cols = train_data.select_dtypes(include=[object]).columns
        for col in cat_cols:
            train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
            test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
            
        # Encode categorical features
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        encoded_features = self.encoder.fit_transform(
            combined_data[['manufacturer', 'model', 'gearbox_type', 'fuel_type']])
        
        encoded_train = encoded_features[:len(train_data)]
        encoded_test = encoded_features[len(train_data):]
        
        # Scale numerical features
        scaled_train = self.scaler.fit_transform(
            train_data[['year', 'engine_capacity', 'operating_hours', 'efficiency']])
        scaled_test = self.scaler.transform(
            test_data[['year', 'engine_capacity', 'operating_hours', 'efficiency']])
        
        # Combine features
        X_train = np.hstack([encoded_train, scaled_train])
        X_test = np.hstack([encoded_test, scaled_test])
        y_train = train_data['price'].values
        
        return X_train, X_test, y_train