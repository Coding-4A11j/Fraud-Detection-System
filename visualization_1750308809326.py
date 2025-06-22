import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def create_fraud_overview_chart(df, fraud_column='predicted_fraud'):
    """
    Create an overview chart showing fraud vs normal transactions
    
    Args:
        df: DataFrame with transaction data
        fraud_column: Column indicating fraud (0/1)
        
    Returns:
        Plotly figure
    """
    if fraud_column not in df.columns:
        return None
    
    fraud_counts = df[fraud_column].value_counts()
    labels = ['Normal', 'Fraud']
    
    fig = px.pie(
        values=fraud_counts.values,
        names=[labels[i] for i in fraud_counts.index],
        title='Transaction Distribution: Normal vs Fraud',
        color_discrete_map={'Normal': '#2E8B57', 'Fraud': '#DC143C'}
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True)
    
    return fig

def create_amount_distribution_chart(df, fraud_column='predicted_fraud', amount_column='amount'):
    """
    Create amount distribution chart with fraud overlay
    
    Args:
        df: DataFrame with transaction data
        fraud_column: Column indicating fraud
        amount_column: Column with transaction amounts
        
    Returns:
        Plotly figure
    """
    if amount_column not in df.columns:
        return None
    
    if fraud_column in df.columns:
        fig = px.histogram(
            df, 
            x=amount_column, 
            color=fraud_column,
            title='Transaction Amount Distribution by Fraud Status',
            labels={fraud_column: 'Fraud Status', amount_column: 'Transaction Amount'},
            nbins=50,
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'}
        )
        
        # Update legend labels
        fig.for_each_trace(lambda t: t.update(name='Normal' if t.name == '0' else 'Fraud'))
    else:
        fig = px.histogram(
            df, 
            x=amount_column,
            title='Transaction Amount Distribution',
            nbins=50
        )
    
    fig.update_layout(
        xaxis_title='Transaction Amount ($)',
        yaxis_title='Count',
        bargap=0.1
    )
    
    return fig

def create_time_series_chart(df, time_column='timestamp', fraud_column='predicted_fraud'):
    """
    Create time series chart showing fraud patterns over time
    
    Args:
        df: DataFrame with transaction data
        time_column: Column with timestamps
        fraud_column: Column indicating fraud
        
    Returns:
        Plotly figure
    """
    if time_column not in df.columns:
        return None
    
    df_time = df.copy()
    df_time[time_column] = pd.to_datetime(df_time[time_column])
    df_time['date'] = df_time[time_column].dt.date
    
    # Daily aggregation
    if fraud_column in df_time.columns:
        daily_stats = df_time.groupby('date').agg({
            fraud_column: ['sum', 'count']
        }).reset_index()
        
        daily_stats.columns = ['date', 'fraud_count', 'total_count']
        daily_stats['fraud_rate'] = daily_stats['fraud_count'] / daily_stats['total_count'] * 100
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            secondary_y=True,
            subplot_titles=['Daily Fraud Analysis']
        )
        
        # Add fraud count bars
        fig.add_trace(
            go.Bar(
                x=daily_stats['date'],
                y=daily_stats['fraud_count'],
                name='Fraud Count',
                marker_color='#DC143C'
            ),
            secondary_y=False,
        )
        
        # Add fraud rate line
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate (%)',
                line=dict(color='#FF6347', width=2)
            ),
            secondary_y=True,
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Fraud Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
        
    else:
        # Just show transaction count over time
        daily_counts = df_time.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title='Daily Transaction Count'
        )
    
    fig.update_xaxes(title_text="Date")
    fig.update_layout(title_text="Transaction Patterns Over Time")
    
    return fig

def create_risk_heatmap(df, x_column, y_column, fraud_column='predicted_fraud'):
    """
    Create risk heatmap showing fraud rates across two dimensions
    
    Args:
        df: DataFrame with transaction data
        x_column: Column for x-axis
        y_column: Column for y-axis
        fraud_column: Column indicating fraud
        
    Returns:
        Plotly figure
    """
    if fraud_column not in df.columns or x_column not in df.columns or y_column not in df.columns:
        return None
    
    # Create cross-tabulation
    risk_matrix = df.groupby([y_column, x_column])[fraud_column].agg(['mean', 'count']).reset_index()
    risk_matrix.columns = [y_column, x_column, 'fraud_rate', 'transaction_count']
    
    # Filter out cells with very few transactions
    risk_matrix = risk_matrix[risk_matrix['transaction_count'] >= 5]
    
    # Pivot for heatmap
    risk_pivot = risk_matrix.pivot(index=y_column, columns=x_column, values='fraud_rate')
    
    fig = px.imshow(
        risk_pivot,
        title=f'Fraud Risk Heatmap: {y_column.title()} vs {x_column.title()}',
        labels=dict(x=x_column.title(), y=y_column.title(), color="Fraud Rate"),
        aspect="auto",
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        xaxis_title=x_column.title(),
        yaxis_title=y_column.title()
    )
    
    return fig

def create_feature_importance_chart(feature_importance_dict, top_n=10):
    """
    Create feature importance chart
    
    Args:
        feature_importance_dict: Dictionary with feature names and importance scores
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    if not feature_importance_dict:
        return None
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(
        list(feature_importance_dict.items()),
        columns=['feature', 'importance']
    ).sort_values('importance', ascending=False).head(top_n)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_roc_curve(y_true, y_pred_proba):
    """
    Create ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='navy', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def create_confusion_matrix_chart(y_true, y_pred):
    """
    Create confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Plotly figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Create labels
    labels = ['Normal', 'Fraud']
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale='Blues'
    )
    
    fig.update_traces(texttemplate="%{text}", textfont_size=16)
    
    return fig

def create_hourly_pattern_chart(df, time_column='timestamp', fraud_column='predicted_fraud'):
    """
    Create hourly pattern chart showing fraud rates by hour of day
    
    Args:
        df: DataFrame with transaction data
        time_column: Column with timestamps
        fraud_column: Column indicating fraud
        
    Returns:
        Plotly figure
    """
    if time_column not in df.columns:
        return None
    
    df_hour = df.copy()
    df_hour[time_column] = pd.to_datetime(df_hour[time_column])
    df_hour['hour'] = df_hour[time_column].dt.hour
    
    if fraud_column in df_hour.columns:
        hourly_stats = df_hour.groupby('hour')[fraud_column].agg(['mean', 'count']).reset_index()
        hourly_stats.columns = ['hour', 'fraud_rate', 'transaction_count']
        hourly_stats['fraud_rate'] = hourly_stats['fraud_rate'] * 100
        
        # Create subplot with two y-axes
        fig = make_subplots(
            rows=1, cols=1,
            secondary_y=True,
            subplot_titles=['Hourly Transaction and Fraud Patterns']
        )
        
        # Add transaction count bars
        fig.add_trace(
            go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['transaction_count'],
                name='Transaction Count',
                marker_color='lightblue',
                opacity=0.7
            ),
            secondary_y=False,
        )
        
        # Add fraud rate line
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate (%)',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True,
        )
        
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
        
    else:
        # Just show transaction count by hour
        hourly_counts = df_hour.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            title='Hourly Transaction Distribution'
        )
    
    fig.update_xaxes(title_text="Hour of Day (0-23)")
    fig.update_layout(title_text="Hourly Transaction Patterns")
    
    return fig

def create_risk_score_distribution(df, risk_score_column='risk_score'):
    """
    Create risk score distribution chart
    
    Args:
        df: DataFrame with transaction data
        risk_score_column: Column with risk scores
        
    Returns:
        Plotly figure
    """
    if risk_score_column not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x=risk_score_column,
        title='Risk Score Distribution',
        nbins=30,
        labels={risk_score_column: 'Risk Score', 'count': 'Number of Transactions'}
    )
    
    # Add vertical lines for risk thresholds
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk Threshold")
    fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold")
    
    fig.update_layout(
        xaxis_title='Risk Score',
        yaxis_title='Count',
        showlegend=False
    )
    
    return fig

def create_merchant_analysis_chart(df, merchant_column='merchant_category', fraud_column='predicted_fraud'):
    """
    Create merchant category analysis chart
    
    Args:
        df: DataFrame with transaction data
        merchant_column: Column with merchant categories
        fraud_column: Column indicating fraud
        
    Returns:
        Plotly figure
    """
    if merchant_column not in df.columns:
        return None
    
    if fraud_column in df.columns:
        merchant_stats = df.groupby(merchant_column)[fraud_column].agg(['mean', 'count']).reset_index()
        merchant_stats.columns = ['category', 'fraud_rate', 'transaction_count']
        merchant_stats['fraud_rate'] = merchant_stats['fraud_rate'] * 100
        
        # Sort by fraud rate
        merchant_stats = merchant_stats.sort_values('fraud_rate', ascending=True)
        
        fig = px.bar(
            merchant_stats,
            x='fraud_rate',
            y='category',
            orientation='h',
            title='Fraud Rate by Merchant Category',
            labels={'fraud_rate': 'Fraud Rate (%)', 'category': 'Merchant Category'},
            color='fraud_rate',
            color_continuous_scale='Reds'
        )
        
    else:
        # Just show transaction count by category
        merchant_counts = df[merchant_column].value_counts().reset_index()
        merchant_counts.columns = ['category', 'count']
        
        fig = px.bar(
            merchant_counts,
            x='count',
            y='category',
            orientation='h',
            title='Transaction Count by Merchant Category'
        )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_user_risk_profile_chart(user_stats_df, top_n=20):
    """
    Create user risk profile chart
    
    Args:
        user_stats_df: DataFrame with user statistics
        top_n: Number of top risky users to show
        
    Returns:
        Plotly figure
    """
    if 'avg_risk' not in user_stats_df.columns:
        return None
    
    # Get top risky users
    top_users = user_stats_df.nlargest(top_n, 'avg_risk')
    
    fig = px.bar(
        top_users,
        x='avg_risk',
        y='user_id',
        orientation='h',
        title=f'Top {top_n} Highest Risk Users',
        labels={'avg_risk': 'Average Risk Score', 'user_id': 'User ID'},
        color='avg_risk',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=max(400, top_n * 20)  # Adjust height based on number of users
    )
    
    return fig

def create_amount_vs_risk_scatter(df, amount_column='amount', risk_column='risk_score', sample_size=1000):
    """
    Create scatter plot of transaction amount vs risk score
    
    Args:
        df: DataFrame with transaction data
        amount_column: Column with transaction amounts
        risk_column: Column with risk scores
        sample_size: Number of points to sample for performance
        
    Returns:
        Plotly figure
    """
    if amount_column not in df.columns or risk_column not in df.columns:
        return None
    
    # Sample data for performance
    df_sample = df.sample(min(sample_size, len(df)))
    
    fig = px.scatter(
        df_sample,
        x=amount_column,
        y=risk_column,
        title='Transaction Amount vs Risk Score',
        labels={amount_column: 'Transaction Amount ($)', risk_column: 'Risk Score'},
        opacity=0.6
    )
    
    # Add risk threshold lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk")
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="High Risk")
    
    return fig

def display_metric_cards(metrics_dict, columns=4):
    """
    Display metrics in a card layout using Streamlit columns
    
    Args:
        metrics_dict: Dictionary with metric names and values
        columns: Number of columns to use
    """
    if not metrics_dict:
        return
    
    cols = st.columns(columns)
    
    for i, (label, value) in enumerate(metrics_dict.items()):
        with cols[i % columns]:
            if isinstance(value, dict) and 'value' in value:
                # Value dictionary with delta
                st.metric(
                    label=label,
                    value=value['value'],
                    delta=value.get('delta', None)
                )
            else:
                # Simple value
                st.metric(label=label, value=value)

def create_model_comparison_chart(model_results):
    """
    Create model comparison chart
    
    Args:
        model_results: Dictionary with model names and their metrics
        
    Returns:
        Plotly figure
    """
    if not model_results:
        return None
    
    models = list(model_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = []
        for model in models:
            if metric in model_results[model]:
                values.append(model_results[model][metric])
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=metric.title(),
            x=models,
            y=values
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group'
    )
    
    return fig
