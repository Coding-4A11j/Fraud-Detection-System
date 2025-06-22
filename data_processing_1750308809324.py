import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_transaction_data(df):
    """
    Clean and preprocess transaction data
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            # Fill numeric columns with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            # Fill categorical columns with mode
            mode_val = df_clean[col].mode()
            df_clean[col] = df_clean[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle negative amounts
    if 'amount' in df_clean.columns:
        df_clean = df_clean[df_clean['amount'] >= 0]
    
    return df_clean

def extract_features(df):
    """
    Extract additional features from transaction data
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        DataFrame with additional features
    """
    df_features = df.copy()
    
    # Time-based features
    if 'timestamp' in df_features.columns:
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
    
    # Amount-based features
    if 'amount' in df_features.columns:
        df_features['amount_log'] = np.log(df_features['amount'] + 1)
        df_features['amount_zscore'] = (df_features['amount'] - df_features['amount'].mean()) / df_features['amount'].std()
        
        # Amount categories
        df_features['amount_category'] = pd.cut(
            df_features['amount'], 
            bins=[0, 50, 200, 500, 1000, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
    
    # User-based features
    if 'user_id' in df_features.columns:
        user_stats = df_features.groupby('user_id')['amount'].agg(['count', 'mean', 'std']).reset_index()
        user_stats.columns = ['user_id', 'user_transaction_count', 'user_avg_amount', 'user_amount_std']
        user_stats['user_amount_std'] = user_stats['user_amount_std'].fillna(0)
        
        df_features = df_features.merge(user_stats, on='user_id', how='left')
        
        # Deviation from user's normal behavior
        df_features['amount_deviation'] = abs(df_features['amount'] - df_features['user_avg_amount'])
    
    # Merchant-based features
    if 'merchant_category' in df_features.columns:
        category_stats = df_features.groupby('merchant_category')['amount'].agg(['mean', 'std']).reset_index()
        category_stats.columns = ['merchant_category', 'category_avg_amount', 'category_amount_std']
        category_stats['category_amount_std'] = category_stats['category_amount_std'].fillna(0)
        
        df_features = df_features.merge(category_stats, on='merchant_category', how='left')
    
    return df_features

def prepare_features_for_ml(df, target_column=None):
    """
    Prepare features for machine learning models
    
    Args:
        df: DataFrame with features
        target_column: Name of target column (for supervised learning)
        
    Returns:
        Tuple of (X, y, feature_names, encoders)
    """
    df_ml = df.copy()
    
    # Separate features and target
    if target_column and target_column in df_ml.columns:
        y = df_ml[target_column].values
        df_ml = df_ml.drop(columns=[target_column])
    else:
        y = None
    
    # Remove non-feature columns
    columns_to_remove = ['timestamp', 'user_id', 'merchant_name']
    df_ml = df_ml.drop(columns=[col for col in columns_to_remove if col in df_ml.columns])
    
    # Encode categorical variables
    encoders = {}
    categorical_columns = df_ml.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le
    
    # Get feature names
    feature_names = df_ml.columns.tolist()
    
    # Convert to numpy array
    X = df_ml.values
    
    return X, y, feature_names, encoders

def detect_anomalies_statistical(df, columns=None, threshold=3):
    """
    Detect anomalies using statistical methods (Z-score)
    
    Args:
        df: DataFrame with data
        columns: List of columns to check (default: all numeric columns)
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        Boolean array indicating anomalies
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    anomalies = np.zeros(len(df), dtype=bool)
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            anomalies |= (z_scores > threshold)
    
    return anomalies

def calculate_transaction_velocity(df, user_column='user_id', time_column='timestamp', window_minutes=60):
    """
    Calculate transaction velocity (number of transactions per time window)
    
    Args:
        df: DataFrame with transaction data
        user_column: Name of user ID column
        time_column: Name of timestamp column
        window_minutes: Time window in minutes
        
    Returns:
        DataFrame with velocity features
    """
    df_velocity = df.copy()
    df_velocity[time_column] = pd.to_datetime(df_velocity[time_column])
    
    # Sort by user and time
    df_velocity = df_velocity.sort_values([user_column, time_column])
    
    # Calculate velocity for each transaction
    velocities = []
    
    for idx, row in df_velocity.iterrows():
        user_id = row[user_column]
        timestamp = row[time_column]
        
        # Get transactions for this user within the time window
        window_start = timestamp - timedelta(minutes=window_minutes)
        user_transactions = df_velocity[
            (df_velocity[user_column] == user_id) & 
            (df_velocity[time_column] >= window_start) & 
            (df_velocity[time_column] < timestamp)
        ]
        
        velocities.append(len(user_transactions))
    
    df_velocity['transaction_velocity'] = velocities
    
    return df_velocity

def create_risk_features(df):
    """
    Create risk-based features for fraud detection
    
    Args:
        df: DataFrame with transaction data
        
    Returns:
        DataFrame with risk features
    """
    df_risk = df.copy()
    
    # High-risk time periods
    if 'timestamp' in df_risk.columns:
        df_risk['timestamp'] = pd.to_datetime(df_risk['timestamp'])
        df_risk['hour'] = df_risk['timestamp'].dt.hour
        
        # Late night/early morning transactions are riskier
        df_risk['is_risky_hour'] = ((df_risk['hour'] >= 23) | (df_risk['hour'] <= 5)).astype(int)
    
    # High-risk amounts
    if 'amount' in df_risk.columns:
        # Very high amounts are riskier
        amount_95th = df_risk['amount'].quantile(0.95)
        df_risk['is_high_amount'] = (df_risk['amount'] > amount_95th).astype(int)
        
        # Very low amounts can also be risky (testing)
        amount_5th = df_risk['amount'].quantile(0.05)
        df_risk['is_low_amount'] = (df_risk['amount'] < amount_5th).astype(int)
    
    # Sequence-based features
    if 'user_id' in df_risk.columns and 'timestamp' in df_risk.columns:
        df_risk = df_risk.sort_values(['user_id', 'timestamp'])
        
        # Time since last transaction for each user
        df_risk['time_since_last'] = df_risk.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60  # minutes
        df_risk['time_since_last'] = df_risk['time_since_last'].fillna(0)
        
        # Rapid successive transactions
        df_risk['is_rapid_transaction'] = (df_risk['time_since_last'] < 5).astype(int)  # less than 5 minutes
    
    # Location-based features (if location data exists)
    if 'location' in df_risk.columns:
        # Transactions from less common locations might be riskier
        location_counts = df_risk['location'].value_counts()
        df_risk['location_frequency'] = df_risk['location'].map(location_counts)
        df_risk['is_rare_location'] = (df_risk['location_frequency'] < 10).astype(int)
    
    return df_risk

def aggregate_user_features(df, user_column='user_id'):
    """
    Create aggregated features per user
    
    Args:
        df: DataFrame with transaction data
        user_column: Name of user ID column
        
    Returns:
        DataFrame with user-level features
    """
    user_features = df.groupby(user_column).agg({
        'amount': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'merchant_category': lambda x: x.nunique() if 'merchant_category' in df.columns else 0,
        'timestamp': lambda x: (x.max() - x.min()).days if 'timestamp' in df.columns else 0
    }).reset_index()
    
    # Flatten column names
    user_features.columns = [
        user_column, 'transaction_count', 'total_amount', 'avg_amount', 'amount_std',
        'min_amount', 'max_amount', 'unique_merchants', 'activity_days'
    ]
    
    # Fill NaN values
    user_features['amount_std'] = user_features['amount_std'].fillna(0)
    
    # Additional derived features
    user_features['amount_range'] = user_features['max_amount'] - user_features['min_amount']
    user_features['avg_transactions_per_day'] = user_features['transaction_count'] / (user_features['activity_days'] + 1)
    
    return user_features
