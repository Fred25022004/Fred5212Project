import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_columns = ['manufacturer', 'model', 'gearbox_type', 'fuel_type']

    def fit_transform(self, df):
        df = df.copy()
        
        # Basic Encoding
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # Feature Engineering
        # Age-related features
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        
        # Usage intensity
        df['hours_per_year'] = df['operating_hours'] / df['vehicle_age']
        
        # Efficiency-related features
        df['power_efficiency'] = df['efficiency'] / df['engine_capacity']
        
        # Cost-related features
        df['registration_cost_per_capacity'] = df['registration_fees'] / df['engine_capacity']
        
        # Categorical interaction features
        df['manufacturer_model'] = df['manufacturer'].astype(str) + '_' + df['model'].astype(str)
        self.label_encoders['manufacturer_model'] = LabelEncoder()
        df['manufacturer_model'] = self.label_encoders['manufacturer_model'].fit_transform(df['manufacturer_model'])

        # Normalize numeric columns
        numeric_columns = ['year', 'operating_hours', 'registration_fees', 'efficiency', 
                         'engine_capacity', 'vehicle_age', 'hours_per_year', 
                         'power_efficiency', 'registration_cost_per_capacity',
                         'manufacturer_model']
        
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        
        return df

    def transform(self, df):
        df = df.copy()
        
        for col in self.categorical_columns:
            df[col] = self.label_encoders[col].transform(df[col])

        # Feature Engineering (same as above)
        current_year = 2024
        df['vehicle_age'] = current_year - df['year']
        df['hours_per_year'] = df['operating_hours'] / df['vehicle_age']
        df['power_efficiency'] = df['efficiency'] / df['engine_capacity']
        df['registration_cost_per_capacity'] = df['registration_fees'] / df['engine_capacity']
        
        df['manufacturer_model'] = df['manufacturer'].astype(str) + '_' + df['model'].astype(str)
        df['manufacturer_model'] = self.label_encoders['manufacturer_model'].transform(df['manufacturer_model'])

        # Normalize numeric columns
        numeric_columns = ['year', 'operating_hours', 'registration_fees', 'efficiency', 
                         'engine_capacity', 'vehicle_age', 'hours_per_year', 
                         'power_efficiency', 'registration_cost_per_capacity',
                         'manufacturer_model']
        
        df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        
        return df