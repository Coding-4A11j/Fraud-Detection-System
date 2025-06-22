import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Add 3D background CSS
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
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
        radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
        radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.3) 0%, transparent 50%);
    z-index: -1;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp > div:first-child {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

.metric-card {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.18);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Fraud Detection Dashboard")
st.markdown("Interactive dashboard showing fraud patterns and trends.")

# Check if data and models are available
if st.session_state.data is None:
    st.error("❌ No data available. Please upload transaction data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

df = st.session_state.data.copy()

# Add predictions to dataframe if available
if st.session_state.predictions is not None:
    df['predicted_fraud'] = st.session_state.predictions
else:
    st.warning("⚠️ No ML model predictions available. Train a model first for enhanced insights.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")

# Sidebar filters
st.sidebar.header("Dashboard Filters")

# Date range filter
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

# Amount range filter
if 'amount' in df.columns:
    min_amount = float(df['amount'].min())
    max_amount = float(df['amount'].max())
    
    amount_range = st.sidebar.slider(
        "Transaction amount range:",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount),
        step=10.0
    )
    
    df = df[(df['amount'] >= amount_range[0]) & (df['amount'] <= amount_range[1])]

# Category filter
if 'merchant_category' in df.columns:
    categories = df['merchant_category'].unique()
    selected_categories = st.sidebar.multiselect(
        "Select merchant categories:",
        options=categories,
        default=categories
    )
    
    if selected_categories:
        df = df[df['merchant_category'].isin(selected_categories)]

# Apply filters info
st.sidebar.markdown(f"**Filtered records:** {len(df):,}")

# Main dashboard
if len(df) == 0:
    st.error("No data matches the selected filters.")
    st.stop()

# Key metrics row
st.subheader("Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=f"{len(df):,}",
        delta=None
    )

with col2:
    if 'amount' in df.columns:
        avg_amount = df['amount'].mean()
        st.metric(
            label="Average Amount",
            value=f"${avg_amount:,.2f}",
            delta=None
        )

with col3:
    if 'predicted_fraud' in df.columns:
        fraud_count = df['predicted_fraud'].sum()
        fraud_rate = fraud_count / len(df) * 100
        st.metric(
            label="Predicted Fraud",
            value=f"{fraud_count:,}",
            delta=f"{fraud_rate:.1f}%"
        )
    elif 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        fraud_rate = fraud_count / len(df) * 100
        st.metric(
            label="Known Fraud",
            value=f"{fraud_count:,}",
            delta=f"{fraud_rate:.1f}%"
        )
    else:
        st.metric(
            label="Fraud Detection",
            value="N/A",
            delta="No model"
        )

with col4:
    if 'amount' in df.columns:
        total_amount = df['amount'].sum()
        st.metric(
            label="Total Volume",
            value=f"${total_amount:,.2f}",
            delta=None
        )

st.markdown("---")

# Main charts
col1, col2 = st.columns(2)

with col1:
    # Transaction volume over time
    if 'timestamp' in df.columns:
        st.subheader("Transaction Volume Over Time")
        
        # Group by date
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby('date').agg({
            'amount': ['count', 'sum']
        }).reset_index()
        
        daily_stats.columns = ['date', 'transaction_count', 'total_amount']
        
        # Create subplot
        fig_volume = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Count', 'Transaction Amount'),
            shared_xaxes=True
        )
        
        fig_volume.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['transaction_count'],
                      mode='lines+markers', name='Count'),
            row=1, col=1
        )
        
        fig_volume.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['total_amount'],
                      mode='lines+markers', name='Amount'),
            row=2, col=1
        )
        
        fig_volume.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_volume, use_container_width=True)

with col2:
    # Fraud detection over time
    if 'predicted_fraud' in df.columns and 'timestamp' in df.columns:
        st.subheader("Fraud Detection Over Time")
        
        fraud_daily = df.groupby('date').agg({
            'predicted_fraud': ['sum', 'count']
        }).reset_index()
        
        fraud_daily.columns = ['date', 'fraud_count', 'total_count']
        fraud_daily['fraud_rate'] = fraud_daily['fraud_count'] / fraud_daily['total_count'] * 100
        
        fig_fraud = go.Figure()
        
        fig_fraud.add_trace(go.Bar(
            x=fraud_daily['date'],
            y=fraud_daily['fraud_count'],
            name='Fraud Count',
            yaxis='y'
        ))
        
        fig_fraud.add_trace(go.Scatter(
            x=fraud_daily['date'],
            y=fraud_daily['fraud_rate'],
            mode='lines+markers',
            name='Fraud Rate (%)',
            yaxis='y2'
        ))
        
        fig_fraud.update_layout(
            title='Daily Fraud Detection',
            yaxis=dict(title='Fraud Count'),
            yaxis2=dict(title='Fraud Rate (%)', overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig_fraud, use_container_width=True)
    elif 'is_fraud' in df.columns and 'timestamp' in df.columns:
        st.subheader("Known Fraud Over Time")
        
        fraud_daily = df.groupby('date').agg({
            'is_fraud': ['sum', 'count']
        }).reset_index()
        
        fraud_daily.columns = ['date', 'fraud_count', 'total_count']
        fraud_daily['fraud_rate'] = fraud_daily['fraud_count'] / fraud_daily['total_count'] * 100
        
        fig_fraud = px.bar(fraud_daily, x='date', y='fraud_count', title='Daily Known Fraud')
        st.plotly_chart(fig_fraud, use_container_width=True)

# Second row of charts
col1, col2 = st.columns(2)

with col1:
    # Amount distribution
    if 'amount' in df.columns:
        st.subheader("Transaction Amount Distribution")
        
        # Create histogram with fraud overlay if available
        if 'predicted_fraud' in df.columns:
            fig_amount = px.histogram(
                df, x='amount', color='predicted_fraud',
                title='Amount Distribution by Fraud Prediction',
                labels={'predicted_fraud': 'Predicted Fraud'},
                nbins=50
            )
        elif 'is_fraud' in df.columns:
            fig_amount = px.histogram(
                df, x='amount', color='is_fraud',
                title='Amount Distribution by Fraud Label',
                labels={'is_fraud': 'Is Fraud'},
                nbins=50
            )
        else:
            fig_amount = px.histogram(df, x='amount', title='Amount Distribution', nbins=50)
        
        st.plotly_chart(fig_amount, use_container_width=True)

with col2:
    # Merchant category analysis
    if 'merchant_category' in df.columns:
        st.subheader("Transactions by Merchant Category")
        
        category_stats = df.groupby('merchant_category').agg({
            'amount': ['count', 'sum', 'mean']
        }).reset_index()
        
        category_stats.columns = ['category', 'count', 'total_amount', 'avg_amount']
        
        # Add fraud information if available
        if 'predicted_fraud' in df.columns:
            fraud_by_category = df.groupby('merchant_category')['predicted_fraud'].agg(['sum', 'mean']).reset_index()
            fraud_by_category.columns = ['category', 'fraud_count', 'fraud_rate']
            category_stats = category_stats.merge(fraud_by_category, on='category')
        
        fig_category = px.bar(
            category_stats.sort_values('count', ascending=True),
            x='count',
            y='category',
            orientation='h',
            title='Transaction Count by Category'
        )
        
        st.plotly_chart(fig_category, use_container_width=True)

# Fraud analysis section
fraud_col = None
if 'predicted_fraud' in df.columns:
    fraud_col = 'predicted_fraud'
elif 'is_fraud' in df.columns:
    fraud_col = 'is_fraud'

if fraud_col is not None:
    st.markdown("---")
    st.subheader("Fraud Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud by amount ranges
        if 'amount' in df.columns:
            st.write("**Fraud by Amount Ranges**")
            
            # Create amount bins
            df['amount_bin'] = pd.cut(df['amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            fraud_by_amount = df.groupby('amount_bin')[fraud_col].agg(['count', 'sum', 'mean']).reset_index()
            fraud_by_amount.columns = ['amount_range', 'total_transactions', 'fraud_count', 'fraud_rate']
            fraud_by_amount['fraud_rate'] = fraud_by_amount['fraud_rate'] * 100
            
            fig_fraud_amount = px.bar(
                fraud_by_amount,
                x='amount_range',
                y='fraud_rate',
                title='Fraud Rate by Amount Range'
            )
            
            st.plotly_chart(fig_fraud_amount, use_container_width=True)
    
    with col2:
        # Fraud by time of day
        if 'timestamp' in df.columns:
            st.write("**Fraud by Time of Day**")
            
            df['hour'] = df['timestamp'].dt.hour
            
            fraud_by_hour = df.groupby('hour')[fraud_col].agg(['count', 'sum', 'mean']).reset_index()
            fraud_by_hour.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']
            fraud_by_hour['fraud_rate'] = fraud_by_hour['fraud_rate'] * 100
            
            fig_fraud_hour = px.line(
                fraud_by_hour,
                x='hour',
                y='fraud_rate',
                title='Fraud Rate by Hour of Day'
            )
            
            st.plotly_chart(fig_fraud_hour, use_container_width=True)

# Risk heatmap
if fraud_col is not None and 'merchant_category' in df.columns and 'amount' in df.columns and fraud_col in df.columns:
    st.markdown("---")
    st.subheader("Risk Heatmap")
    
    # Create risk matrix
    risk_matrix = df.groupby(['merchant_category', pd.cut(df['amount'], bins=5)])[fraud_col].mean().reset_index()
    risk_matrix.columns = ['merchant_category', 'amount_range', 'fraud_rate']
    
    # Pivot for heatmap
    risk_pivot = risk_matrix.pivot(index='merchant_category', columns='amount_range', values='fraud_rate')
    
    fig_heatmap = px.imshow(
        risk_pivot,
        title='Fraud Risk Heatmap (Merchant Category vs Amount Range)',
        labels=dict(x="Amount Range", y="Merchant Category", color="Fraud Rate"),
        aspect="auto"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# High-risk transactions table
if fraud_col is not None:
    st.markdown("---")
    st.subheader("High-Risk Transactions")
    
    # Filter high-risk transactions
    if fraud_col == 'predicted_fraud':
        high_risk = df[df[fraud_col] == 1].copy()
    else:
        high_risk = df[df[fraud_col] == 1].copy()
    
    if len(high_risk) > 0:
        # Show top high-risk transactions
        display_cols = []
        
        # Only add columns that exist
        if 'user_id' in high_risk.columns:
            display_cols.append('user_id')
        if 'amount' in high_risk.columns:
            display_cols.append('amount')
        if 'merchant_category' in high_risk.columns:
            display_cols.append('merchant_category')
        if 'timestamp' in high_risk.columns:
            display_cols.append('timestamp')
        
        # Add risk scores if available from models
        if st.session_state.models and 'Logistic Regression' in st.session_state.models:
            if 'predictions_proba' in st.session_state.models['Logistic Regression']:
                high_risk['risk_score'] = st.session_state.models['Logistic Regression']['predictions_proba']
                display_cols.append('risk_score')
        
        if display_cols:
            st.dataframe(
                high_risk[display_cols].head(20),
                use_container_width=True
            )
        else:
            st.dataframe(high_risk.head(20), use_container_width=True)
        
        # Export high-risk transactions
        if st.button("Export High-Risk Transactions"):
            csv_data = high_risk.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"high_risk_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No high-risk transactions found with current filters.")

# Real-time monitoring section
st.markdown("---")
st.subheader("Real-time Monitoring")

# Auto-refresh option
auto_refresh = st.checkbox("Enable auto-refresh (5 seconds)")

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()

# Alert system
if fraud_col is not None:
    recent_fraud = df[df[fraud_col] == 1]
    if 'timestamp' in df.columns:
        recent_fraud = recent_fraud[recent_fraud['timestamp'] > (datetime.now() - timedelta(hours=1))]
    
    if len(recent_fraud) > 0:
        st.error(f"🚨 Alert: {len(recent_fraud)} high-risk transactions detected!")
        
        # Show recent alerts
        with st.expander("View Recent Alerts"):
            # Define display_cols for alerts
            alert_display_cols = []
            if 'user_id' in recent_fraud.columns:
                alert_display_cols.append('user_id')
            if 'amount' in recent_fraud.columns:
                alert_display_cols.append('amount')
            if 'merchant_category' in recent_fraud.columns:
                alert_display_cols.append('merchant_category')
            if 'timestamp' in recent_fraud.columns:
                alert_display_cols.append('timestamp')
            
            if alert_display_cols:
                st.dataframe(recent_fraud[alert_display_cols].head(10))
            else:
                st.dataframe(recent_fraud.head(10))
    else:
        st.success("✅ No high-risk transactions in the last hour")
