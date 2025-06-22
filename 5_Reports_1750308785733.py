import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Reports", page_icon="📈", layout="wide")

# Add 3D background CSS
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #a8edea, #fed6e3, #d299c2, #fef9d7);
    background-size: 400% 400%;
    animation: gradient 28s ease infinite;
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
        radial-gradient(circle at 20% 60%, rgba(168, 237, 234, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 80% 40%, rgba(254, 214, 227, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 50% 20%, rgba(210, 153, 194, 0.2) 0%, transparent 50%);
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

st.title("📈 Fraud Detection Reports")
st.markdown("Generate comprehensive fraud detection reports and analytics.")

# Check if data is available
if st.session_state.data is None:
    st.error("❌ No data available. Please upload transaction data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

df = st.session_state.data.copy()

# Add predictions if available
if st.session_state.predictions is not None:
    df['predicted_fraud'] = st.session_state.predictions
    
    # Add risk scores if available
    if st.session_state.models and 'Logistic Regression' in st.session_state.models:
        if 'predictions_proba' in st.session_state.models['Logistic Regression']:
            df['risk_score'] = st.session_state.models['Logistic Regression']['predictions_proba']

# Report configuration
st.subheader("Report Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    report_type = st.selectbox(
        "Select report type:",
        ["Executive Summary", "Detailed Analysis", "Model Performance", "User Analysis", "Time-based Analysis"]
    )

with col2:
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        
        report_period = st.date_input(
            "Report period:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(report_period) == 2:
            start_date, end_date = report_period
            df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

with col3:
    include_charts = st.checkbox("Include charts in report", value=True)
    include_tables = st.checkbox("Include detailed tables", value=True)

st.markdown("---")

# Generate report content based on type
if report_type == "Executive Summary":
    st.subheader("📊 Executive Summary Report")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        if 'amount' in df.columns:
            total_volume = df['amount'].sum()
            st.metric("Total Volume", f"${total_volume:,.2f}")
    
    with col3:
        if 'predicted_fraud' in df.columns:
            fraud_count = df['predicted_fraud'].sum()
            st.metric("Fraud Detected", f"{fraud_count:,}")
    
    with col4:
        if 'predicted_fraud' in df.columns and 'amount' in df.columns:
            fraud_amount = df[df['predicted_fraud'] == 1]['amount'].sum()
            st.metric("Fraud Amount", f"${fraud_amount:,.2f}")
    
    # Summary insights
    st.subheader("Key Insights")
    
    insights = []
    
    if 'predicted_fraud' in df.columns:
        fraud_rate = df['predicted_fraud'].mean() * 100
        insights.append(f"• Fraud detection rate: {fraud_rate:.2f}% of all transactions")
        
        if 'amount' in df.columns:
            avg_fraud_amount = df[df['predicted_fraud'] == 1]['amount'].mean()
            avg_normal_amount = df[df['predicted_fraud'] == 0]['amount'].mean()
            
            if avg_fraud_amount > avg_normal_amount:
                insights.append(f"• Fraudulent transactions are {avg_fraud_amount/avg_normal_amount:.1f}x larger on average")
            
            fraud_amount_pct = (df[df['predicted_fraud'] == 1]['amount'].sum() / df['amount'].sum()) * 100
            insights.append(f"• Fraudulent transactions represent {fraud_amount_pct:.1f}% of total transaction volume")
    
    if 'merchant_category' in df.columns and 'predicted_fraud' in df.columns:
        fraud_by_category = df.groupby('merchant_category')['predicted_fraud'].mean()
        highest_risk_category = fraud_by_category.idxmax()
        highest_risk_rate = fraud_by_category.max() * 100
        insights.append(f"• Highest risk category: {highest_risk_category} ({highest_risk_rate:.1f}% fraud rate)")
    
    if 'timestamp' in df.columns and 'predicted_fraud' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        fraud_by_hour = df.groupby('hour')['predicted_fraud'].mean()
        peak_fraud_hour = fraud_by_hour.idxmax()
        insights.append(f"• Peak fraud hour: {peak_fraud_hour}:00 ({fraud_by_hour.max()*100:.1f}% fraud rate)")
    
    for insight in insights:
        st.write(insight)
    
    # Charts for executive summary
    if include_charts:
        if 'predicted_fraud' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Fraud trend over time
                if 'timestamp' in df.columns:
                    df['date'] = df['timestamp'].dt.date
                    daily_fraud = df.groupby('date')['predicted_fraud'].agg(['sum', 'count']).reset_index()
                    daily_fraud.columns = ['date', 'fraud_count', 'total_count']
                    daily_fraud['fraud_rate'] = daily_fraud['fraud_count'] / daily_fraud['total_count'] * 100
                    
                    fig_trend = px.line(daily_fraud, x='date', y='fraud_rate', 
                                      title='Daily Fraud Rate Trend')
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # Fraud by category
                if 'merchant_category' in df.columns:
                    fraud_by_cat = df.groupby('merchant_category')['predicted_fraud'].agg(['sum', 'count']).reset_index()
                    fraud_by_cat.columns = ['category', 'fraud_count', 'total_count']
                    fraud_by_cat['fraud_rate'] = fraud_by_cat['fraud_count'] / fraud_by_cat['total_count'] * 100
                    
                    fig_category = px.bar(fraud_by_cat.sort_values('fraud_rate', ascending=False),
                                        x='fraud_rate', y='category', orientation='h',
                                        title='Fraud Rate by Category')
                    st.plotly_chart(fig_category, use_container_width=True)

elif report_type == "Detailed Analysis":
    st.subheader("🔍 Detailed Analysis Report")
    
    # Transaction patterns
    st.write("**Transaction Patterns Analysis**")
    
    if 'amount' in df.columns:
        # Amount distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Amount Statistics:**")
            amount_stats = df['amount'].describe()
            st.dataframe(amount_stats)
        
        with col2:
            if 'predicted_fraud' in df.columns:
                st.write("**Amount by Fraud Status:**")
                fraud_amount_stats = df.groupby('predicted_fraud')['amount'].describe()
                st.dataframe(fraud_amount_stats)
    
    # Temporal analysis
    if 'timestamp' in df.columns:
        st.write("**Temporal Analysis:**")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        if include_charts:
            # Hourly distribution
            hourly_dist = df.groupby('hour').size().reset_index(name='count')
            fig_hourly = px.bar(hourly_dist, x='hour', y='count', 
                              title='Transaction Distribution by Hour')
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Feature correlation analysis
    if include_tables:
        st.write("**Feature Correlation Analysis:**")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                               title='Feature Correlation Matrix',
                               aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)

elif report_type == "Model Performance":
    st.subheader("🤖 Model Performance Report")
    
    if not st.session_state.models:
        st.error("No trained models available. Please train models first.")
        if st.button("Go to Model Training"):
            st.switch_page("pages/2_Model_Training.py")
    else:
        # Model performance metrics
        for model_name, model_data in st.session_state.models.items():
            st.write(f"**{model_name} Performance:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'predictions' in model_data:
                    fraud_detected = model_data['predictions'].sum()
                    st.metric("Fraud Detected", f"{fraud_detected:,}")
            
            with col2:
                if 'predictions' in model_data:
                    fraud_rate = model_data['predictions'].mean() * 100
                    st.metric("Detection Rate", f"{fraud_rate:.2f}%")
            
            with col3:
                if 'test_predictions' in model_data and 'test_actual' in model_data:
                    accuracy = (model_data['test_predictions'] == model_data['test_actual']).mean()
                    st.metric("Test Accuracy", f"{accuracy:.3f}")
            
            # Feature importance
            if model_name == "Logistic Regression" and 'feature_names' in model_data:
                st.write("**Feature Importance:**")
                
                coefficients = model_data['model'].coef_[0]
                feature_importance = pd.DataFrame({
                    'feature': model_data['feature_names'],
                    'importance': np.abs(coefficients)
                }).sort_values('importance', ascending=False)
                
                if include_charts:
                    fig_importance = px.bar(feature_importance.head(10),
                                          x='importance', y='feature', orientation='h',
                                          title='Top 10 Feature Importance')
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                if include_tables:
                    st.dataframe(feature_importance)
            
            st.markdown("---")

elif report_type == "User Analysis":
    st.subheader("👥 User Analysis Report")
    
    if 'user_id' in df.columns:
        # User behavior analysis
        user_stats = df.groupby('user_id').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'predicted_fraud': 'sum' if 'predicted_fraud' in df.columns else 'count'
        }).reset_index()
        
        # Flatten column names
        user_stats.columns = ['user_id', 'transaction_count', 'total_amount', 'avg_amount', 'amount_std', 'fraud_count']
        
        # Calculate metrics
        user_stats['fraud_rate'] = user_stats['fraud_count'] / user_stats['transaction_count']
        
        # Top users by different metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Users by Transaction Volume:**")
            top_volume = user_stats.nlargest(10, 'transaction_count')[['user_id', 'transaction_count', 'total_amount']]
            st.dataframe(top_volume)
        
        with col2:
            if 'predicted_fraud' in df.columns:
                st.write("**Top Users by Fraud Count:**")
                top_fraud = user_stats.nlargest(10, 'fraud_count')[['user_id', 'fraud_count', 'fraud_rate']]
                st.dataframe(top_fraud)
        
        # User distribution charts
        if include_charts:
            # Transaction count distribution
            fig_user_dist = px.histogram(user_stats, x='transaction_count', 
                                       title='Distribution of User Transaction Counts',
                                       nbins=20)
            st.plotly_chart(fig_user_dist, use_container_width=True)
            
            # Amount vs fraud rate scatter
            if 'predicted_fraud' in df.columns:
                fig_scatter = px.scatter(user_stats, x='total_amount', y='fraud_rate',
                                       title='User Total Amount vs Fraud Rate',
                                       hover_data=['user_id'])
                st.plotly_chart(fig_scatter, use_container_width=True)

elif report_type == "Time-based Analysis":
    st.subheader("⏰ Time-based Analysis Report")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month
        
        # Daily trends
        daily_stats = df.groupby('date').agg({
            'amount': ['count', 'sum', 'mean'],
            'predicted_fraud': 'sum' if 'predicted_fraud' in df.columns else 'count'
        }).reset_index()
        
        daily_stats.columns = ['date', 'transaction_count', 'total_amount', 'avg_amount', 'fraud_count']
        
        if 'predicted_fraud' in df.columns:
            daily_stats['fraud_rate'] = daily_stats['fraud_count'] / daily_stats['transaction_count']
        
        # Summary statistics
        st.write("**Daily Statistics Summary:**")
        st.dataframe(daily_stats.describe())
        
        # Time-based charts
        if include_charts:
            # Daily transaction volume
            fig_daily = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Transaction Count', 'Daily Transaction Amount')
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_stats['date'], y=daily_stats['transaction_count'],
                          mode='lines+markers', name='Count'),
                row=1, col=1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_stats['date'], y=daily_stats['total_amount'],
                          mode='lines+markers', name='Amount'),
                row=2, col=1
            )
            
            fig_daily.update_layout(height=500, title='Daily Transaction Trends')
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Hourly patterns
            hourly_stats = df.groupby('hour').agg({
                'amount': ['count', 'mean'],
                'predicted_fraud': 'mean' if 'predicted_fraud' in df.columns else 'count'
            }).reset_index()
            
            hourly_stats.columns = ['hour', 'transaction_count', 'avg_amount', 'fraud_rate']
            
            fig_hourly = px.line(hourly_stats, x='hour', y='transaction_count',
                               title='Hourly Transaction Pattern')
            st.plotly_chart(fig_hourly, use_container_width=True)

# Export report
st.markdown("---")
st.subheader("Export Report")

col1, col2 = st.columns(2)

with col1:
    export_format = st.selectbox("Export format:", ["CSV", "JSON"])

with col2:
    report_name = st.text_input("Report name:", value=f"{report_type}_Report_{datetime.now().strftime('%Y%m%d')}")

if st.button("Generate Export", type="primary"):
    # Prepare data for export
    export_data = df.copy()
    
    # Add summary statistics
    summary_stats = {
        'report_type': report_type,
        'generated_at': datetime.now().isoformat(),
        'total_transactions': len(df),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "N/A"
    }
    
    if 'predicted_fraud' in df.columns:
        summary_stats['fraud_detected'] = int(df['predicted_fraud'].sum())
        summary_stats['fraud_rate'] = float(df['predicted_fraud'].mean())
    
    if 'amount' in df.columns:
        summary_stats['total_amount'] = float(df['amount'].sum())
        summary_stats['average_amount'] = float(df['amount'].mean())
    
    if export_format == "CSV":
        # Create CSV with summary at the top
        csv_buffer = io.StringIO()
        
        # Write summary
        csv_buffer.write("# FRAUD DETECTION REPORT SUMMARY\n")
        for key, value in summary_stats.items():
            csv_buffer.write(f"# {key}: {value}\n")
        csv_buffer.write("# \n")
        csv_buffer.write("# TRANSACTION DATA\n")
        
        # Write data
        export_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="Download CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"{report_name}.csv",
            mime="text/csv"
        )
    
    elif export_format == "JSON":
        # Create JSON report
        json_report = {
            'summary': summary_stats,
            'data': export_data.to_dict('records')
        }
        
        import json
        json_str = json.dumps(json_report, indent=2, default=str)
        
        st.download_button(
            label="Download JSON Report",
            data=json_str,
            file_name=f"{report_name}.json",
            mime="application/json"
        )
    
    st.success(f"✅ {report_type} report generated successfully!")

# Report scheduling (placeholder)
st.markdown("---")
st.subheader("Report Scheduling")

col1, col2 = st.columns(2)

with col1:
    schedule_frequency = st.selectbox("Schedule frequency:", ["Daily", "Weekly", "Monthly"])

with col2:
    email_recipients = st.text_input("Email recipients (comma-separated):")

if st.button("Schedule Report"):
    st.info("📧 Report scheduling would be implemented here. This feature requires email service integration.")

# Print/PDF option
st.markdown("---")
if st.button("Print Report"):
    st.info("🖨️ Use your browser's print function (Ctrl+P) to print or save as PDF.")
