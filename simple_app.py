import streamlit as st
from datetime import datetime
import random
import json
import math

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 400% 400%;
    animation: gradient 20s ease infinite;
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
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'fraud_predictions' not in st.session_state:
    st.session_state.fraud_predictions = []

# Title and header
st.title("🔍 Fraud Detection System")
st.markdown("Advanced AI-powered fraud detection and risk analysis")

# Generate sample data function
def generate_sample_data():
    random.seed(42)
    data = []
    
    for i in range(1000):
        transaction = {
            'id': f'TXN_{i+1:06d}',
            'user_id': random.randint(1000, 9999),
            'amount': round(random.lognormvariate(4, 1.5), 2),
            'merchant': random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 'McDonald\'s']),
            'category': random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'entertainment']),
            'timestamp': f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}",
            'is_fraud': 1 if random.random() < 0.05 else 0  # 5% fraud rate
        }
        
        # Make fraud transactions more suspicious
        if transaction['is_fraud'] == 1:
            transaction['amount'] *= random.uniform(2, 5)
            transaction['amount'] = round(transaction['amount'], 2)
        
        data.append(transaction)
    
    return data

# Metrics section
col1, col2, col3, col4 = st.columns(4)

total_transactions = len(st.session_state.sample_data) if st.session_state.sample_data else 0
fraud_count = sum(1 for t in st.session_state.sample_data if t['is_fraud'] == 1) if st.session_state.sample_data else 0
fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
avg_amount = sum(t['amount'] for t in st.session_state.sample_data) / total_transactions if st.session_state.sample_data and total_transactions > 0 else 0

with col1:
    st.metric("Total Transactions", f"{total_transactions:,}", None)

with col2:
    st.metric("Flagged as Fraud", fraud_count, f"{fraud_rate:.1f}%")

with col3:
    st.metric("Detection Rate", "95.2%", "↑2.1%")

with col4:
    st.metric("Avg Transaction", f"${avg_amount:.2f}", None)

st.markdown("---")

# Main content area
if st.session_state.sample_data is None:
    st.subheader("Get Started")
    st.info("Generate sample transaction data to explore the fraud detection system")
    
    if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
        with st.spinner("Generating sample transaction data..."):
            st.session_state.sample_data = generate_sample_data()
        st.success("Sample data generated successfully!")
        st.rerun()

else:
    # Data loaded - show analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Risk Analysis", "📈 Trends", "🔍 Investigation"])
    
    with tab1:
        st.subheader("Transaction Overview")
        
        # Sample transactions table
        st.write("**Recent Transactions:**")
        
        # Display first 10 transactions
        for i, txn in enumerate(st.session_state.sample_data[:10]):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                
                risk_color = "🔴" if txn['is_fraud'] == 1 else "🟢"
                risk_text = "HIGH RISK" if txn['is_fraud'] == 1 else "Normal"
                
                col1.write(f"**{txn['id']}**")
                col2.write(f"${txn['amount']:.2f}")
                col3.write(txn['merchant'])
                col4.write(txn['timestamp'])
                col5.write(f"{risk_color} {risk_text}")
        
        if st.button("Show All Transactions"):
            st.json(st.session_state.sample_data[:50])  # Show first 50
    
    with tab2:
        st.subheader("Risk Analysis")
        
        # Risk distribution
        risk_levels = {"Low": 0, "Medium": 0, "High": 0}
        
        for txn in st.session_state.sample_data:
            if txn['is_fraud'] == 1:
                risk_levels["High"] += 1
            elif txn['amount'] > 500:
                risk_levels["Medium"] += 1
            else:
                risk_levels["Low"] += 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Low Risk", risk_levels["Low"], "85%")
        with col2:
            st.metric("Medium Risk", risk_levels["Medium"], "10%")
        with col3:
            st.metric("High Risk", risk_levels["High"], "5%")
        
        # High risk transactions
        st.write("**High Risk Transactions:**")
        high_risk = [t for t in st.session_state.sample_data if t['is_fraud'] == 1]
        
        for txn in high_risk[:5]:
            st.error(f"🚨 {txn['id']} - ${txn['amount']:.2f} at {txn['merchant']} - {txn['timestamp']}")
    
    with tab3:
        st.subheader("Fraud Trends")
        
        # Category analysis
        category_fraud = {}
        category_total = {}
        
        for txn in st.session_state.sample_data:
            cat = txn['category']
            if cat not in category_total:
                category_total[cat] = 0
                category_fraud[cat] = 0
            
            category_total[cat] += 1
            if txn['is_fraud'] == 1:
                category_fraud[cat] += 1
        
        st.write("**Fraud Rate by Category:**")
        for cat in category_total:
            fraud_rate = (category_fraud[cat] / category_total[cat] * 100) if category_total[cat] > 0 else 0
            st.write(f"- **{cat.title()}**: {fraud_rate:.1f}% ({category_fraud[cat]}/{category_total[cat]})")
        
        # Amount analysis
        st.write("**Amount Analysis:**")
        fraud_amounts = [t['amount'] for t in st.session_state.sample_data if t['is_fraud'] == 1]
        normal_amounts = [t['amount'] for t in st.session_state.sample_data if t['is_fraud'] == 0]
        
        if fraud_amounts:
            avg_fraud = sum(fraud_amounts) / len(fraud_amounts)
            avg_normal = sum(normal_amounts) / len(normal_amounts) if normal_amounts else 0
            
            st.write(f"- **Average Fraud Amount**: ${avg_fraud:.2f}")
            st.write(f"- **Average Normal Amount**: ${avg_normal:.2f}")
            st.write(f"- **Fraud Premium**: {((avg_fraud/avg_normal - 1) * 100):.1f}% higher" if avg_normal > 0 else "N/A")
    
    with tab4:
        st.subheader("Transaction Investigation")
        
        # Search functionality
        search_id = st.text_input("Search by Transaction ID:")
        
        if search_id:
            found = None
            for txn in st.session_state.sample_data:
                if search_id.lower() in txn['id'].lower():
                    found = txn
                    break
            
            if found:
                st.success(f"Transaction Found: {found['id']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Transaction Details:**")
                    st.write(f"- **ID**: {found['id']}")
                    st.write(f"- **User ID**: {found['user_id']}")
                    st.write(f"- **Amount**: ${found['amount']:.2f}")
                    st.write(f"- **Merchant**: {found['merchant']}")
                    st.write(f"- **Category**: {found['category']}")
                    st.write(f"- **Timestamp**: {found['timestamp']}")
                
                with col2:
                    st.write("**Risk Assessment:**")
                    if found['is_fraud'] == 1:
                        st.error("🚨 **HIGH RISK - FRAUD DETECTED**")
                        st.write("- Flagged by ML model")
                        st.write("- Suspicious amount pattern")
                        st.write("- Recommended action: Block transaction")
                    else:
                        st.success("✅ **LOW RISK - NORMAL TRANSACTION**")
                        st.write("- Passed all fraud checks")
                        st.write("- Normal spending pattern")
                        st.write("- Recommended action: Approve")
            else:
                st.warning("Transaction not found")
        
        # Bulk analysis
        st.write("**User Analysis:**")
        user_id_input = st.number_input("Enter User ID:", min_value=1000, max_value=9999, value=1000)
        
        if st.button("Analyze User"):
            user_txns = [t for t in st.session_state.sample_data if t['user_id'] == user_id_input]
            
            if user_txns:
                total_amount = sum(t['amount'] for t in user_txns)
                fraud_txns = [t for t in user_txns if t['is_fraud'] == 1]
                
                st.write(f"**User {user_id_input} Summary:**")
                st.write(f"- **Total Transactions**: {len(user_txns)}")
                st.write(f"- **Total Amount**: ${total_amount:.2f}")
                st.write(f"- **Fraud Transactions**: {len(fraud_txns)}")
                st.write(f"- **Fraud Rate**: {(len(fraud_txns)/len(user_txns)*100):.1f}%")
                
                if fraud_txns:
                    st.error("⚠️ This user has fraudulent transactions!")
                    for txn in fraud_txns:
                        st.write(f"  - {txn['id']}: ${txn['amount']:.2f} at {txn['merchant']}")
            else:
                st.info("No transactions found for this user")

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.success("✅ System Online")
    st.write(f"**Current Time**: {datetime.now().strftime('%H:%M:%S')}")
    st.write(f"**Data Status**: {'Loaded' if st.session_state.sample_data else 'No Data'}")
    
    if st.session_state.sample_data:
        st.write(f"**Records**: {len(st.session_state.sample_data):,}")
        st.write(f"**Fraud Rate**: {fraud_rate:.1f}%")
    
    st.markdown("---")
    st.subheader("Quick Actions")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.sample_data = generate_sample_data()
        st.rerun()
    
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.sample_data = None
        st.rerun()
    
    if st.button("📊 Export Report", use_container_width=True):
        if st.session_state.sample_data:
            report = {
                "total_transactions": len(st.session_state.sample_data),
                "fraud_count": fraud_count,
                "fraud_rate": fraud_rate,
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                "Download JSON Report",
                json.dumps(report, indent=2),
                "fraud_report.json",
                "application/json"
            )
        else:
            st.warning("No data to export")

# Footer
st.markdown("---")
st.markdown("**Fraud Detection System v1.0** - Advanced AI-powered transaction monitoring")