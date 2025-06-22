import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

# Add 3D background CSS
st.markdown("""
<style>
.main > div {
    background: linear-gradient(-45deg, #43cea2, #185a9d, #667eea, #764ba2);
    background-size: 400% 400%;
    animation: gradient 22s ease infinite;
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
        radial-gradient(circle at 35% 30%, rgba(67, 206, 162, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 65% 70%, rgba(24, 90, 157, 0.2) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(102, 126, 234, 0.2) 0%, transparent 50%);
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

st.title("🤖 Model Training")
st.markdown("Train machine learning models for fraud detection.")

# Check if data is available
if st.session_state.data is None:
    st.error("❌ No data available. Please upload transaction data first.")
    if st.button("Go to Data Upload"):
        st.switch_page("pages/1_Data_Upload.py")
    st.stop()

# Data preparation section
st.subheader("Data Preparation")

df = st.session_state.data.copy()

# Show data info
col1, col2 = st.columns(2)

with col1:
    st.write("**Dataset Information:**")
    st.write(f"- Total records: {len(df):,}")
    st.write(f"- Features: {len(df.columns)}")
    
    if 'is_fraud' in df.columns:
        fraud_count = df['is_fraud'].sum()
        st.write(f"- Known fraud cases: {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
    else:
        st.write("- No fraud labels available (unsupervised learning)")

with col2:
    st.write("**Available Features:**")
    for col in df.columns:
        st.write(f"- {col} ({df[col].dtype})")

# Feature selection
st.subheader("Feature Selection")

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove timestamp and target columns from feature selection
if 'timestamp' in numeric_cols:
    numeric_cols.remove('timestamp')
if 'timestamp' in categorical_cols:
    categorical_cols.remove('timestamp')
if 'is_fraud' in numeric_cols:
    numeric_cols.remove('is_fraud')
if 'is_fraud' in categorical_cols:
    categorical_cols.remove('is_fraud')

col1, col2 = st.columns(2)

with col1:
    st.write("**Numeric Features:**")
    selected_numeric = st.multiselect(
        "Select numeric features to include:",
        numeric_cols,
        default=numeric_cols
    )

with col2:
    st.write("**Categorical Features:**")
    selected_categorical = st.multiselect(
        "Select categorical features to include:",
        categorical_cols,
        default=categorical_cols[:3] if len(categorical_cols) > 3 else categorical_cols
    )

# Prepare features
if selected_numeric or selected_categorical:
    # Prepare feature matrix
    X = pd.DataFrame()
    
    # Add numeric features
    if selected_numeric:
        X = pd.concat([X, df[selected_numeric]], axis=1)
    
    # Encode categorical features
    if selected_categorical:
        for col in selected_categorical:
            le = LabelEncoder()
            X[col] = le.fit_transform(df[col].astype(str))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    st.success(f"✅ Prepared {X_scaled.shape[1]} features for model training")
    
    # Model selection and training
    st.subheader("Model Selection")
    
    model_type = st.selectbox(
        "Select model type:",
        ["Isolation Forest (Unsupervised)", "Logistic Regression (Supervised)", "Both Models"]
    )
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Isolation Forest Parameters:**")
        contamination = st.slider("Contamination rate", 0.01, 0.20, 0.05, 0.01)
        n_estimators = st.slider("Number of estimators", 50, 200, 100, 10)
    
    with col2:
        if 'is_fraud' in df.columns and model_type in ["Logistic Regression (Supervised)", "Both Models"]:
            st.write("**Logistic Regression Parameters:**")
            test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
            regularization = st.selectbox("Regularization", ["l2", "l1", "elasticnet"])
        else:
            st.info("Supervised learning requires fraud labels (is_fraud column)")
    
    # Train models
    if st.button("Train Models", type="primary"):
        trained_models = {}
        results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train Isolation Forest
        if model_type in ["Isolation Forest (Unsupervised)", "Both Models"]:
            status_text.text("Training Isolation Forest...")
            progress_bar.progress(25)
            
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )
            
            iso_forest.fit(X_scaled)
            iso_predictions = iso_forest.predict(X_scaled)
            iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
            
            trained_models['Isolation Forest'] = {
                'model': iso_forest,
                'scaler': scaler,
                'predictions': iso_predictions,
                'feature_names': X.columns.tolist()
            }
            
            results['Isolation Forest'] = {
                'fraud_detected': iso_predictions.sum(),
                'fraud_rate': iso_predictions.mean() * 100
            }
            
            progress_bar.progress(50)
        
        # Train Logistic Regression
        if 'is_fraud' in df.columns and model_type in ["Logistic Regression (Supervised)", "Both Models"]:
            status_text.text("Training Logistic Regression...")
            progress_bar.progress(75)
            
            y = df['is_fraud'].values
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            log_reg = LogisticRegression(
                penalty=regularization,
                random_state=42,
                max_iter=1000
            )
            
            log_reg.fit(X_train, y_train)
            
            # Predictions
            y_pred = log_reg.predict(X_test)
            y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
            
            # Full dataset predictions
            full_predictions = log_reg.predict(X_scaled)
            full_predictions_proba = log_reg.predict_proba(X_scaled)[:, 1]
            
            trained_models['Logistic Regression'] = {
                'model': log_reg,
                'scaler': scaler,
                'predictions': full_predictions,
                'predictions_proba': full_predictions_proba,
                'feature_names': X.columns.tolist(),
                'test_predictions': y_pred,
                'test_predictions_proba': y_pred_proba,
                'test_actual': y_test
            }
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            results['Logistic Regression'] = {
                'fraud_detected': full_predictions.sum(),
                'fraud_rate': full_predictions.mean() * 100,
                'auc_score': auc_score,
                'test_accuracy': (y_pred == y_test).mean()
            }
        
        progress_bar.progress(100)
        status_text.text("Training completed!")
        
        # Store models in session state
        st.session_state.models = trained_models
        
        # Combine predictions for session state
        if len(trained_models) == 1:
            model_name = list(trained_models.keys())[0]
            st.session_state.predictions = trained_models[model_name]['predictions']
        else:
            # Use ensemble of both models
            iso_pred = trained_models['Isolation Forest']['predictions']
            lr_pred = trained_models['Logistic Regression']['predictions']
            ensemble_pred = ((iso_pred + lr_pred) >= 1).astype(int)
            st.session_state.predictions = ensemble_pred
        
        st.success("✅ Models trained successfully!")
        
        # Display results
        st.subheader("Training Results")
        
        for model_name, result in results.items():
            st.write(f"**{model_name}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Detected", f"{result['fraud_detected']:,}")
            
            with col2:
                st.metric("Fraud Rate", f"{result['fraud_rate']:.2f}%")
            
            with col3:
                if 'auc_score' in result:
                    st.metric("AUC Score", f"{result['auc_score']:.3f}")
                elif 'anomaly_score' in result:
                    st.metric("Avg Anomaly Score", f"{result['anomaly_score']:.3f}")
                else:
                    st.metric("Accuracy", f"{result.get('test_accuracy', 0):.3f}")
        
        st.markdown("---")
        
        # Model evaluation charts
        st.subheader("Model Evaluation")
        
        if 'Logistic Regression' in trained_models and 'is_fraud' in df.columns:
            model_data = trained_models['Logistic Regression']
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(model_data['test_actual'], model_data['test_predictions_proba'])
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {results["Logistic Regression"]["auc_score"]:.3f})'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            
            # Confusion Matrix
            cm = confusion_matrix(model_data['test_actual'], model_data['test_predictions'])
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto", 
                              title="Confusion Matrix",
                              labels=dict(x="Predicted", y="Actual"))
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_roc, use_container_width=True)
            with col2:
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(model_data['test_actual'], model_data['test_predictions'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # Feature importance (for Logistic Regression)
        if 'Logistic Regression' in trained_models:
            st.subheader("Feature Importance")
            model_data = trained_models['Logistic Regression']
            coefficients = model_data['model'].coef_[0]
            feature_importance = pd.DataFrame({
                'feature': model_data['feature_names'],
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance.head(10), 
                x='importance', 
                y='feature',
                orientation='h',
                title='Top 10 Feature Importance (Logistic Regression)'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.warning("⚠️ Please select at least one feature for model training.")

# Current models status
if st.session_state.models:
    st.markdown("---")
    st.subheader("Current Models Status")
    
    for model_name, model_data in st.session_state.models.items():
        with st.expander(f"{model_name} Details"):
            st.write(f"**Model Type:** {model_name}")
            st.write(f"**Features Used:** {len(model_data['feature_names'])}")
            st.write(f"**Feature Names:** {', '.join(model_data['feature_names'])}")
            
            if 'predictions' in model_data:
                fraud_count = model_data['predictions'].sum()
                st.write(f"**Fraud Detected:** {fraud_count:,} transactions")
                st.write(f"**Fraud Rate:** {fraud_count/len(model_data['predictions'])*100:.2f}%")
