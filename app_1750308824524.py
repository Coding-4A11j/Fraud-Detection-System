import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add 3D background effect to main app
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

.metric-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.18);
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

# Main page
st.title("🔍 Fraud Detection System")
st.markdown("---")

# Overview section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Transactions",
        value=len(st.session_state.data) if st.session_state.data is not None else 0,
        delta=None
    )

with col2:
    if st.session_state.predictions is not None:
        fraud_count = sum(st.session_state.predictions)
        st.metric(
            label="Flagged Transactions",
            value=fraud_count,
            delta=f"{(fraud_count/len(st.session_state.predictions)*100):.1f}%" if len(st.session_state.predictions) > 0 else "0%"
        )
    else:
        st.metric(label="Flagged Transactions", value=0, delta="0%")

with col3:
    st.metric(
        label="Models Trained",
        value=len(st.session_state.models),
        delta=None
    )

with col4:
    if st.session_state.data is not None:
        avg_amount = st.session_state.data['amount'].mean() if 'amount' in st.session_state.data.columns else 0
        st.metric(
            label="Avg Transaction Amount",
            value=f"${avg_amount:.2f}",
            delta=None
        )
    else:
        st.metric(label="Avg Transaction Amount", value="$0.00", delta=None)

st.markdown("---")

# Quick actions
st.subheader("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View Dashboard", use_container_width=True):
        st.switch_page("pages/3_Dashboard.py")

with col2:
    if st.button("🎯 Risk Analysis", use_container_width=True):
        st.switch_page("pages/4_Risk_Analysis.py")

with col3:
    if st.button("📈 Generate Report", use_container_width=True):
        st.switch_page("pages/5_Reports.py")

# Recent activity
st.subheader("System Status")

if st.session_state.data is not None:
    st.success("✅ Transaction data loaded successfully")
    
    # Show data summary
    with st.expander("Data Summary"):
        st.write(f"**Shape:** {st.session_state.data.shape}")
        st.write(f"**Columns:** {', '.join(st.session_state.data.columns)}")
        st.write("**Sample Data:**")
        st.dataframe(st.session_state.data.head())
else:
    st.info("ℹ️ No transaction data loaded. Please upload data to get started.")

if st.session_state.models:
    st.success(f"✅ {len(st.session_state.models)} ML model(s) trained and ready")
else:
    st.info("ℹ️ No ML models trained yet. Train models for fraud detection.")

# Instructions
st.markdown("---")
st.subheader("Getting Started")

st.markdown("""
1. **Upload Data**: Go to the Data Upload page to load your transaction data
2. **Train Models**: Use the Model Training page to build fraud detection models
3. **View Dashboard**: Analyze fraud patterns and trends in the Dashboard
4. **Risk Analysis**: Examine individual transactions and risk scores
5. **Generate Reports**: Create comprehensive fraud detection reports
""")

# Code download section
st.markdown("---")
st.subheader("Download Source Code")

if st.button("📥 Download Complete Project", type="primary"):
    import zipfile
    import io
    from pathlib import Path
    
    # Create in-memory zip file
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add main files
        for file_path in [
            'app.py', 
            'pyproject.toml',
            '.streamlit/config.toml'
        ]:
            if Path(file_path).exists():
                zipf.write(file_path)
        
        # Add pages
        pages_dir = Path('pages')
        if pages_dir.exists():
            for page_file in pages_dir.glob('*.py'):
                zipf.write(page_file)
        
        # Add utils
        utils_dir = Path('utils')
        if utils_dir.exists():
            for util_file in utils_dir.glob('*.py'):
                zipf.write(util_file)
        
        # Add README
        readme_content = """# Fraud Detection System

A comprehensive Streamlit-based fraud detection system with machine learning capabilities.

## Features
- Interactive data upload and preprocessing
- ML model training (Isolation Forest, Logistic Regression)
- Real-time fraud detection dashboard
- Risk analysis and user profiling
- Comprehensive reporting system
- Beautiful animated 3D backgrounds

## Installation
1. Install dependencies: `pip install streamlit pandas numpy scikit-learn plotly`
2. Run the app: `streamlit run app.py`

## Usage
1. Upload transaction data or generate sample data
2. Train ML models for fraud detection
3. Analyze results in the interactive dashboard
4. Generate detailed reports

## Project Structure
- `app.py` - Main application entry point
- `pages/` - Individual page components
- `utils/` - Utility functions for data processing, ML models, and visualization
"""
        zipf.writestr('README.md', readme_content)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="💾 Download fraud_detection_system.zip",
        data=zip_buffer.getvalue(),
        file_name="fraud_detection_system.zip",
        mime="application/zip"
    )
    
    st.success("Project files prepared for download!")

# Sidebar information
with st.sidebar:
    st.header("System Information")
    st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if st.session_state.data is not None:
        st.write(f"**Data Status:** Loaded")
        st.write(f"**Records:** {len(st.session_state.data):,}")
    else:
        st.write("**Data Status:** Not loaded")
    
    if st.session_state.models:
        st.write("**Available Models:**")
        for model_name in st.session_state.models.keys():
            st.write(f"- {model_name}")
    else:
        st.write("**Models:** None trained")
    
    st.markdown("---")
    st.subheader("Navigation")
    st.write("Use the pages in the sidebar to navigate through different sections of the fraud detection system.")
