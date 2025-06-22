import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Risk Analysis", page_icon="🎯", layout="wide")

# Add 3D background CSS
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #ffecd2, #fcb69f);
    background-size: 400% 400%;
    animation: gradient 25s ease infinite;
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
        radial-gradient(circle at 40% 35%, rgba(255, 154, 158, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 60% 65%, rgba(250, 208, 196, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 30% 70%, rgba(252, 182, 159, 0.2) 0%, transparent 50%);
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

st.title("🎯 Risk Analysis")
st.markdown("Detailed risk assessment and transaction analysis.")

# Check if data is available
if st.session_state.data is None:
    st.error("❌ No data available. Please upload transaction data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

df = st.session_state.data.copy()

# Add predictions and risk scores if available
if st.session_state.predictions is not None:
    df['predicted_fraud'] = st.session_state.predictions
    
    # Add risk scores from Logistic Regression if available
    if st.session_state.models and 'Logistic Regression' in st.session_state.models:
        if 'predictions_proba' in st.session_state.models['Logistic Regression']:
            df['risk_score'] = st.session_state.models['Logistic Regression']['predictions_proba']
        else:
            df['risk_score'] = df['predicted_fraud']  # Use binary prediction as fallback
    else:
        df['risk_score'] = df['predicted_fraud']  # Use binary prediction as fallback
else:
    st.warning("⚠️ No ML model predictions available. Some risk analysis features may be limited.")
    df['risk_score'] = 0  # Default risk score

# Risk categories
def categorize_risk(score):
    if score >= 0.8:
        return "Very High"
    elif score >= 0.6:
        return "High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.2:
        return "Low"
    else:
        return "Very Low"

if 'risk_score' in df.columns:
    df['risk_category'] = df['risk_score'].apply(categorize_risk)

# Sidebar filters
st.sidebar.header("Risk Analysis Filters")

# Risk level filter
if 'risk_category' in df.columns:
    risk_levels = df['risk_category'].unique()
    selected_risks = st.sidebar.multiselect(
        "Select risk levels:",
        options=risk_levels,
        default=risk_levels
    )
    
    if selected_risks:
        df = df[df['risk_category'].isin(selected_risks)]

# Amount filter
if 'amount' in df.columns:
    min_amount = float(df['amount'].min())
    max_amount = float(df['amount'].max())
    
    amount_filter = st.sidebar.slider(
        "Minimum transaction amount:",
        min_value=min_amount,
        max_value=max_amount,
        value=min_amount,
        step=10.0
    )
    
    df = df[df['amount'] >= amount_filter]

# User filter
if 'user_id' in df.columns:
    users = df['user_id'].unique()
    selected_user = st.sidebar.selectbox(
        "Select specific user (optional):",
        options=["All Users"] + list(users)
    )
    
    if selected_user != "All Users":
        df = df[df['user_id'] == selected_user]

st.sidebar.markdown(f"**Filtered records:** {len(df):,}")

# Main analysis
if len(df) == 0:
    st.error("No data matches the selected filters.")
    st.stop()

# Risk overview
st.subheader("Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if 'risk_score' in df.columns:
        avg_risk = df['risk_score'].mean()
        st.metric(
            label="Average Risk Score",
            value=f"{avg_risk:.3f}",
            delta=None
        )

with col2:
    if 'predicted_fraud' in df.columns:
        high_risk_count = (df['predicted_fraud'] == 1).sum()
        st.metric(
            label="High Risk Transactions",
            value=f"{high_risk_count:,}",
            delta=f"{high_risk_count/len(df)*100:.1f}%"
        )

with col3:
    if 'amount' in df.columns and 'predicted_fraud' in df.columns:
        high_risk_amount = df[df['predicted_fraud'] == 1]['amount'].sum()
        st.metric(
            label="High Risk Amount",
            value=f"${high_risk_amount:,.2f}",
            delta=None
        )

with col4:
    if 'user_id' in df.columns and 'predicted_fraud' in df.columns:
        high_risk_users = df[df['predicted_fraud'] == 1]['user_id'].nunique()
        st.metric(
            label="High Risk Users",
            value=f"{high_risk_users:,}",
            delta=None
        )

st.markdown("---")

# Risk distribution
col1, col2 = st.columns(2)

with col1:
    if 'risk_category' in df.columns:
        st.subheader("Risk Distribution")
        
        risk_counts = df['risk_category'].value_counts()
        
        fig_risk_dist = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Distribution of Risk Categories"
        )
        
        st.plotly_chart(fig_risk_dist, use_container_width=True)

with col2:
    if 'risk_score' in df.columns:
        st.subheader("Risk Score Distribution")
        
        fig_risk_hist = px.histogram(
            df, x='risk_score',
            title="Risk Score Distribution",
            nbins=20
        )
        
        st.plotly_chart(fig_risk_hist, use_container_width=True)

# Risk by features
st.subheader("Risk Analysis by Features")

col1, col2 = st.columns(2)

with col1:
    if 'merchant_category' in df.columns and 'risk_score' in df.columns:
        st.write("**Average Risk by Merchant Category**")
        
        risk_by_category = df.groupby('merchant_category')['risk_score'].agg(['mean', 'count']).reset_index()
        risk_by_category.columns = ['category', 'avg_risk', 'transaction_count']
        risk_by_category = risk_by_category.sort_values('avg_risk', ascending=False)
        
        fig_category_risk = px.bar(
            risk_by_category,
            x='avg_risk',
            y='category',
            orientation='h',
            title='Average Risk Score by Merchant Category'
        )
        
        st.plotly_chart(fig_category_risk, use_container_width=True)

with col2:
    if 'amount' in df.columns and 'risk_score' in df.columns:
        st.write("**Risk vs Transaction Amount**")
        
        fig_amount_risk = px.scatter(
            df.sample(min(1000, len(df))),  # Sample for performance
            x='amount',
            y='risk_score',
            title='Risk Score vs Transaction Amount',
            opacity=0.6
        )
        
        st.plotly_chart(fig_amount_risk, use_container_width=True)

# Time-based risk analysis
if 'timestamp' in df.columns and 'risk_score' in df.columns:
    st.subheader("Time-based Risk Analysis")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk by Hour of Day**")
        
        hourly_risk = df.groupby('hour')['risk_score'].mean().reset_index()
        
        fig_hourly = px.line(
            hourly_risk,
            x='hour',
            y='risk_score',
            title='Average Risk Score by Hour of Day'
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.write("**Risk by Day of Week**")
        
        daily_risk = df.groupby('day_of_week')['risk_score'].mean().reset_index()
        
        # Order days correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_risk['day_of_week'] = pd.Categorical(daily_risk['day_of_week'], categories=day_order, ordered=True)
        daily_risk = daily_risk.sort_values('day_of_week')
        
        fig_daily = px.bar(
            daily_risk,
            x='day_of_week',
            y='risk_score',
            title='Average Risk Score by Day of Week'
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)

# User risk profile
if 'user_id' in df.columns:
    st.subheader("User Risk Profiles")
    
    # Calculate user risk metrics
    user_risk = df.groupby('user_id').agg({
        'risk_score': ['mean', 'max', 'count'],
        'amount': ['sum', 'mean', 'max'],
        'predicted_fraud': 'sum' if 'predicted_fraud' in df.columns else 'count'
    }).reset_index()
    
    # Flatten column names
    user_risk.columns = ['user_id', 'avg_risk', 'max_risk', 'transaction_count', 
                        'total_amount', 'avg_amount', 'max_amount', 'fraud_count']
    
    # Calculate risk level
    user_risk['risk_level'] = user_risk['avg_risk'].apply(categorize_risk)
    
    # Sort by average risk
    user_risk = user_risk.sort_values('avg_risk', ascending=False)
    
    # Show top risky users
    st.write("**Top 20 Highest Risk Users**")
    
    display_columns = ['user_id', 'avg_risk', 'max_risk', 'transaction_count', 'total_amount', 'fraud_count', 'risk_level']
    st.dataframe(user_risk[display_columns].head(20), use_container_width=True)
    
    # User risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**User Risk Level Distribution**")
        
        user_risk_dist = user_risk['risk_level'].value_counts()
        
        fig_user_risk = px.pie(
            values=user_risk_dist.values,
            names=user_risk_dist.index,
            title="Distribution of User Risk Levels"
        )
        
        st.plotly_chart(fig_user_risk, use_container_width=True)
    
    with col2:
        st.write("**Risk vs Transaction Volume**")
        
        fig_volume_risk = px.scatter(
            user_risk,
            x='transaction_count',
            y='avg_risk',
            size='total_amount',
            title='User Risk vs Transaction Volume',
            hover_data=['user_id']
        )
        
        st.plotly_chart(fig_volume_risk, use_container_width=True)

# Transaction details
st.subheader("Transaction Details")

# Search and filter
col1, col2, col3 = st.columns(3)

with col1:
    search_user = st.text_input("Search by User ID:")

with col2:
    min_risk = st.number_input("Minimum Risk Score:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

with col3:
    sort_by = st.selectbox("Sort by:", ["Risk Score", "Amount", "Timestamp"])

# Filter transactions
filtered_df = df.copy()

if search_user and 'user_id' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['user_id'].str.contains(search_user, case=False, na=False)]

if 'risk_score' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['risk_score'] >= min_risk]

# Sort transactions
if sort_by == "Risk Score" and 'risk_score' in filtered_df.columns:
    filtered_df = filtered_df.sort_values('risk_score', ascending=False)
elif sort_by == "Amount" and 'amount' in filtered_df.columns:
    filtered_df = filtered_df.sort_values('amount', ascending=False)
elif sort_by == "Timestamp" and 'timestamp' in filtered_df.columns:
    filtered_df = filtered_df.sort_values('timestamp', ascending=False)

# Display transactions
if len(filtered_df) > 0:
    # Select columns to display (only include existing columns)
    display_cols = []
    
    if 'user_id' in filtered_df.columns:
        display_cols.append('user_id')
    if 'amount' in filtered_df.columns:
        display_cols.append('amount')
    if 'risk_score' in filtered_df.columns:
        display_cols.append('risk_score')
    if 'risk_category' in filtered_df.columns:
        display_cols.append('risk_category')
    if 'merchant_category' in filtered_df.columns:
        display_cols.append('merchant_category')
    if 'timestamp' in filtered_df.columns:
        display_cols.append('timestamp')
    if 'predicted_fraud' in filtered_df.columns:
        display_cols.append('predicted_fraud')
    
    if display_cols:
        st.dataframe(
            filtered_df[display_cols].head(50),
            use_container_width=True
        )
    else:
        st.dataframe(filtered_df.head(50), use_container_width=True)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export High Risk Transactions"):
            high_risk_df = filtered_df[filtered_df['risk_score'] >= 0.5] if 'risk_score' in filtered_df.columns else filtered_df
            csv_data = high_risk_df.to_csv(index=False)
            st.download_button(
                label="Download High Risk CSV",
                data=csv_data,
                file_name=f"high_risk_transactions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export All Filtered Transactions"):
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv_data,
                file_name=f"filtered_transactions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("No transactions match the current filters.")

# Risk scoring explanation
with st.expander("Risk Scoring Explanation"):
    st.markdown("""
    **Risk Score Calculation:**
    
    The risk scores are calculated using machine learning models trained on transaction patterns:
    
    - **Very High (0.8-1.0)**: Transactions with characteristics strongly indicating fraud
    - **High (0.6-0.8)**: Transactions with several suspicious patterns
    - **Medium (0.4-0.6)**: Transactions with some unusual characteristics
    - **Low (0.2-0.4)**: Transactions with minimal risk indicators
    - **Very Low (0.0-0.2)**: Transactions appearing normal
    
    **Factors Considered:**
    - Transaction amount (unusually high or low)
    - Merchant category patterns
    - Time of day and day of week
    - User transaction history
    - Geographic patterns (if location data available)
    """)
