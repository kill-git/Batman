from core.feature_engineering import (
    create_date_features,
    create_hour_features,
    create_lag_features,
    create_rolling_features
)
from core.data_preprocessing import preprocess_eCO2mix_data_engineered

def final_feature_cleaning_node(df):
    return preprocess_eCO2mix_data_engineered(df)

def create_features_node(df):
    df = create_date_features(df, date_column='Date')
    df = create_hour_features(df)
    df = create_lag_features(df, target_column='Consommation', lags=[1, 2, 24])
    df = create_rolling_features(df, target_column='Consommation', window=3)
    return df

