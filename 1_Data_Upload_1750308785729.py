import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Data Upload", page_icon="📁", layout="wide")

# Add 3D background CSS
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 400% 400%;
    animation: gradient 18s ease infinite;
    position: relative;
}

.main > div::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 25% 45%, rgba(120, 119, 198, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 75% 25%, rgba(255, 119, 198, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 45% 75%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
    z-index: -1;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(5px);
}
</style>
""", unsafe_allow_html=True)

st.title("📁 Data Upload")
st.markdown("Upload your transaction data for fraud detection analysis.")

# File upload section
st.subheader("Upload Transaction Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file containing transaction data",
    type=['csv'],
    help="Upload a CSV file with transaction data including columns like amount, timestamp, merchant, etc."
)

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Display file information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File Information:**")
            st.write(f"- Rows: {len(df):,}")
            st.write(f"- Columns: {len(df.columns)}")
            st.write(f"- File size: {uploaded_file.size:,} bytes")
        
        with col2:
            st.write("**Column Types:**")
            for col, dtype in df.dtypes.items():
                st.write(f"- {col}: {dtype}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Data validation
        st.subheader("Data Validation")
        
        # Check for required columns
        required_columns = ['amount', 'timestamp', 'merchant_category', 'user_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your data contains the following columns: amount, timestamp, merchant_category, user_id")
        else:
            st.success("✅ All required columns present")
            
            # Additional validation
            validation_issues = []
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                validation_issues.append(f"Missing values found: {missing_values.sum()} total")
            
            # Check amount column
            if 'amount' in df.columns:
                if df['amount'].dtype not in ['int64', 'float64']:
                    validation_issues.append("Amount column should be numeric")
                if (df['amount'] < 0).any():
                    validation_issues.append("Negative amounts found")
            
            # Check timestamp column
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'])
                except:
                    validation_issues.append("Timestamp column cannot be converted to datetime")
            
            if validation_issues:
                st.warning("⚠️ Data validation issues found:")
                for issue in validation_issues:
                    st.write(f"- {issue}")
            else:
                st.success("✅ Data validation passed")
        
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            handle_missing = st.selectbox(
                "Handle missing values:",
                ["Keep as is", "Drop rows with missing values", "Fill with mean/mode"]
            )
        
        with col2:
            normalize_amounts = st.checkbox("Normalize transaction amounts")
        
        # Apply preprocessing
        processed_df = df.copy()
        
        if handle_missing == "Drop rows with missing values":
            initial_rows = len(processed_df)
            processed_df = processed_df.dropna()
            st.info(f"Dropped {initial_rows - len(processed_df)} rows with missing values")
        elif handle_missing == "Fill with mean/mode":
            for col in processed_df.columns:
                if processed_df[col].dtype in ['int64', 'float64']:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                else:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown')
        
        if normalize_amounts and 'amount' in processed_df.columns:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            processed_df['amount_normalized'] = scaler.fit_transform(processed_df[['amount']])
            st.info("Added normalized amount column")
        
        # Load data button
        if st.button("Load Data for Analysis", type="primary"):
            st.session_state.data = processed_df
            st.success("✅ Data loaded successfully and ready for analysis!")
            st.balloons()
            
            # Show loaded data summary
            st.subheader("Loaded Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Transactions", len(processed_df))
            
            with col2:
                if 'amount' in processed_df.columns:
                    st.metric("Average Amount", f"${processed_df['amount'].mean():.2f}")
                else:
                    st.metric("Average Amount", "N/A")
            
            with col3:
                if 'timestamp' in processed_df.columns:
                    try:
                        timestamps = pd.to_datetime(processed_df['timestamp'])
                        date_range = (timestamps.max() - timestamps.min()).days
                        st.metric("Date Range (days)", date_range)
                    except:
                        st.metric("Date Range", "N/A")
                else:
                    st.metric("Date Range", "N/A")
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

else:
    # Show sample data format
    st.subheader("Expected Data Format")
    st.write("Your CSV file should contain the following columns:")
    
    sample_data = pd.DataFrame({
        'user_id': ['user_001', 'user_002', 'user_003'],
        'amount': [150.75, 89.20, 1200.00],
        'timestamp': ['2024-01-15 10:30:00', '2024-01-15 11:45:00', '2024-01-15 14:20:00'],
        'merchant_category': ['grocery', 'gas_station', 'electronics'],
        'merchant_name': ['SuperMart', 'Shell Station', 'TechStore'],
        'card_type': ['credit', 'debit', 'credit'],
        'location': ['New York', 'California', 'Texas']
    })
    
    st.dataframe(sample_data)
    
    # Provide sample data generation
    st.subheader("Generate Sample Data")
    st.write("For testing purposes, you can generate sample transaction data:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_transactions = st.number_input("Number of transactions", min_value=100, max_value=10000, value=1000)
    
    with col2:
        fraud_rate = st.slider("Fraud rate (%)", min_value=1, max_value=10, value=3)
    
    if st.button("Generate Sample Data"):
        # Generate sample transaction data
        np.random.seed(42)
        
        # Normal transactions
        normal_count = int(num_transactions * (100 - fraud_rate) / 100)
        fraud_count = num_transactions - normal_count
        
        # Generate normal transactions
        normal_amounts = np.random.lognormal(mean=4, sigma=1, size=normal_count)
        normal_amounts = np.clip(normal_amounts, 5, 1000)
        
        # Generate fraudulent transactions (typically higher amounts)
        fraud_amounts = np.random.lognormal(mean=6, sigma=1.5, size=fraud_count)
        fraud_amounts = np.clip(fraud_amounts, 500, 10000)
        
        # Combine amounts
        all_amounts = np.concatenate([normal_amounts, fraud_amounts])
        
        # Generate other features
        user_ids = [f"user_{i:04d}" for i in np.random.randint(1, 1000, num_transactions)]
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(
            seconds=np.random.randint(0, 30*24*3600)
        ) for _ in range(num_transactions)]
        
        merchant_categories = np.random.choice([
            'grocery', 'gas_station', 'restaurant', 'electronics', 'clothing', 
            'pharmacy', 'entertainment', 'travel', 'utilities', 'other'
        ], num_transactions)
        
        merchant_names = [f"Merchant_{i}" for i in np.random.randint(1, 200, num_transactions)]
        
        card_types = np.random.choice(['credit', 'debit'], num_transactions, p=[0.7, 0.3])
        
        locations = np.random.choice([
            'New York', 'California', 'Texas', 'Florida', 'Illinois', 
            'Pennsylvania', 'Ohio', 'Georgia', 'North Carolina', 'Michigan'
        ], num_transactions)
        
        # Create labels (1 for fraud, 0 for normal)
        labels = np.concatenate([np.zeros(normal_count), np.ones(fraud_count)])
        
        # Create DataFrame
        sample_df = pd.DataFrame({
            'user_id': user_ids,
            'amount': all_amounts,
            'timestamp': timestamps,
            'merchant_category': merchant_categories,
            'merchant_name': merchant_names,
            'card_type': card_types,
            'location': locations,
            'is_fraud': labels.astype(int)
        })
        
        # Shuffle the data
        sample_df = sample_df.sample(frac=1).reset_index(drop=True)
        
        # Store in session state
        st.session_state.data = sample_df
        
        st.success(f"✅ Generated {num_transactions} sample transactions with {fraud_rate}% fraud rate")
        st.dataframe(sample_df.head())
        
        # Download link for sample data
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Sample Data",
            data=csv_data,
            file_name="sample_transactions.csv",
            mime="text/csv"
        )

# Current data status
if st.session_state.data is not None:
    st.markdown("---")
    st.subheader("Current Data Status")
    st.success(f"✅ Data loaded: {len(st.session_state.data):,} transactions")
    
    # Show basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(st.session_state.data))
    
    with col2:
        if 'amount' in st.session_state.data.columns:
            st.metric("Avg Amount", f"${st.session_state.data['amount'].mean():.2f}")
    
    with col3:
        if 'is_fraud' in st.session_state.data.columns:
            fraud_count = st.session_state.data['is_fraud'].sum()
            st.metric("Known Fraud", f"{fraud_count} ({fraud_count/len(st.session_state.data)*100:.1f}%)")
    
    with col4:
        st.metric("Features", len(st.session_state.data.columns))
