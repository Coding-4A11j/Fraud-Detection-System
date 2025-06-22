import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for 3D animated background
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
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
        radial-gradient(circle at 30% 40%, rgba(120, 119, 198, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 70% 20%, rgba(255, 119, 198, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 50% 80%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
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

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Main title
st.title("🔍 Advanced Fraud Detection System")
st.markdown("*AI-powered financial crime detection and risk management platform*")

# System overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_transactions = len(st.session_state.data) if st.session_state.data is not None else 0
    st.metric(
        label="Total Transactions",
        value=f"{total_transactions:,}",
        delta=None
    )

with col2:
    if st.session_state.predictions is not None:
        fraud_count = sum(st.session_state.predictions)
        fraud_rate = (fraud_count / len(st.session_state.predictions) * 100) if len(st.session_state.predictions) > 0 else 0
        st.metric(
            label="Fraud Detected",
            value=fraud_count,
            delta=f"{fraud_rate:.1f}%"
        )
    else:
        st.metric(label="Fraud Detected", value=0, delta="0%")

with col3:
    model_count = len(st.session_state.models)
    st.metric(
        label="ML Models Active",
        value=model_count,
        delta="Ready" if model_count > 0 else "None"
    )

with col4:
    if st.session_state.data is not None and hasattr(st.session_state.data, 'columns') and 'amount' in st.session_state.data.columns:
        avg_amount = st.session_state.data['amount'].mean()
        st.metric(
            label="Avg Transaction",
            value=f"${avg_amount:.2f}",
            delta=None
        )
    else:
        st.metric(label="Avg Transaction", value="$0", delta=None)

st.markdown("---")

# Navigation instructions
st.subheader("System Navigation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Data Management**
    - Upload transaction data (CSV format)
    - Generate sample datasets for testing
    - Data preprocessing and feature engineering
    
    **🤖 Machine Learning**
    - Train fraud detection models
    - Isolation Forest (unsupervised)
    - Logistic Regression & Random Forest
    - Model evaluation and comparison
    """)

with col2:
    st.markdown("""
    **📈 Analytics & Monitoring**
    - Real-time fraud detection dashboard
    - Risk analysis and scoring
    - Transaction pattern analysis
    - Comprehensive reporting system
    
    **🔍 Investigation Tools**
    - Individual transaction review
    - User behavior analysis
    - Fraud trend identification
    """)

# Quick action buttons
st.subheader("Quick Actions")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📤 Data Upload", use_container_width=True):
        st.switch_page("pages/1_Data_Upload.py")

with col2:
    if st.button("🎯 Train Models", use_container_width=True):
        st.switch_page("pages/2_Model_Training.py")

with col3:
    if st.button("📊 Dashboard", use_container_width=True):
        st.switch_page("pages/3_Dashboard.py")

with col4:
    if st.button("⚠️ Risk Analysis", use_container_width=True):
        st.switch_page("pages/4_Risk_Analysis.py")

with col5:
    if st.button("📋 Reports", use_container_width=True):
        st.switch_page("pages/5_Reports.py")

# System status
st.markdown("---")
st.subheader("System Status")

if st.session_state.data is not None:
    st.success("✅ Transaction data loaded and ready for analysis")
    
    with st.expander("Data Summary", expanded=False):
        if hasattr(st.session_state.data, 'shape'):
            st.write(f"**Dataset Shape:** {st.session_state.data.shape[0]:,} transactions × {st.session_state.data.shape[1]} features")
            st.write(f"**Memory Usage:** {st.session_state.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            if hasattr(st.session_state.data, 'columns'):
                st.write("**Available Columns:**")
                for col in st.session_state.data.columns:
                    st.write(f"- {col}")
else:
    st.info("ℹ️ No transaction data loaded. Please navigate to **Data Upload** to begin.")

if st.session_state.models:
    st.success(f"✅ {len(st.session_state.models)} machine learning model(s) trained and ready")
    
    with st.expander("Model Details", expanded=False):
        for model_name, model_info in st.session_state.models.items():
            st.write(f"**{model_name}**")
            if isinstance(model_info, dict) and 'type' in model_info:
                st.write(f"  - Type: {model_info['type']}")
                st.write(f"  - Status: Ready")
            else:
                st.write(f"  - Status: Trained")
else:
    st.info("ℹ️ No ML models trained. Visit **Model Training** to build fraud detection models.")

# Recent activity log
st.markdown("---")
st.subheader("Recent Activity")

if 'activity_log' not in st.session_state:
    st.session_state.activity_log = []

if st.session_state.activity_log:
    for activity in st.session_state.activity_log[-5:]:  # Show last 5 activities
        st.write(f"• {activity}")
else:
    st.write("No recent activity")

# Footer with system information
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 30px;'>
    <h4>🔐 Advanced Fraud Detection System</h4>
    <p>Powered by machine learning algorithms for real-time financial crime detection</p>
    <p><strong>Features:</strong> Multi-model detection • Real-time scoring • Pattern analysis • Risk assessment</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with system information
with st.sidebar:
    st.header("🔍 System Control")
    
    # System metrics
    st.subheader("Live Metrics")
    
    import datetime
    current_time = datetime.datetime.now()
    st.write(f"**System Time:** {current_time.strftime('%H:%M:%S')}")
    st.write(f"**Date:** {current_time.strftime('%Y-%m-%d')}")
    
    if st.session_state.data is not None:
        st.write("**Data Status:** 🟢 Loaded")
        if hasattr(st.session_state.data, 'shape'):
            st.write(f"**Records:** {st.session_state.data.shape[0]:,}")
    else:
        st.write("**Data Status:** 🔴 No Data")
    
    st.write(f"**ML Models:** {len(st.session_state.models)}")
    
    # Quick settings
    st.markdown("---")
    st.subheader("Settings")
    
    # Alert thresholds
    st.write("**Alert Thresholds**")
    fraud_threshold = st.slider("Fraud Score Threshold", 0.0, 1.0, 0.5, 0.01)
    amount_threshold = st.number_input("High Amount Alert ($)", value=1000, step=100)
    
    # System actions
    st.markdown("---")
    st.subheader("System Actions")
    
    if st.button("🔄 Refresh System", use_container_width=True):
        st.rerun()
    
    if st.button("📊 Quick Stats", use_container_width=True):
        if st.session_state.data is not None:
            st.success("System operational")
        else:
            st.warning("Load data to see statistics")
    
    # Help section
    st.markdown("---")
    st.subheader("Help & Support")
    st.markdown("""
    **Getting Started:**
    1. Upload transaction data
    2. Train detection models
    3. Monitor dashboard
    4. Analyze risks
    5. Generate reports
    
    **Need Help?**
    - Check the documentation
    - Review system logs
    - Contact support team
    """)