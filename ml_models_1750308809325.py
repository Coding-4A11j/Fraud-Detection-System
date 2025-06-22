import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import pickle

class FraudDetectionModel:
    """
    Base class for fraud detection models
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def fit(self, X, y=None):
        """Fit the model to training data"""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions"""
        raise NotImplementedError
        
    def predict_proba(self, X):
        """Get prediction probabilities"""
        raise NotImplementedError
        
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': type(self).__name__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = True

class IsolationForestModel(FraudDetectionModel):
    """
    Isolation Forest model for unsupervised fraud detection
    """
    
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
    def fit(self, X, y=None, feature_names=None):
        """
        Fit Isolation Forest model
        
        Args:
            X: Feature matrix
            y: Target vector (ignored for unsupervised learning)
            feature_names: List of feature names
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled)
        self.feature_names = feature_names
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies (1 for fraud, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert -1 (anomaly) to 1 (fraud), 1 (normal) to 0 (not fraud)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X):
        """
        Get anomaly scores as probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert anomaly scores to probabilities (0-1 range)
        # Lower scores indicate more anomalous behavior
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        if max_score == min_score:
            probabilities = np.full(len(anomaly_scores), 0.5)
        else:
            probabilities = (max_score - anomaly_scores) / (max_score - min_score)
        
        return probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance (not directly available for Isolation Forest)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        # Isolation Forest doesn't provide direct feature importance
        # Return uniform importance for all features
        if self.feature_names:
            return dict(zip(self.feature_names, [1.0] * len(self.feature_names)))
        else:
            return {}

class LogisticRegressionModel(FraudDetectionModel):
    """
    Logistic Regression model for supervised fraud detection
    """
    
    def __init__(self, penalty='l2', C=1.0, random_state=42):
        super().__init__()
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        
    def fit(self, X, y, feature_names=None):
        """
        Fit Logistic Regression model
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        self.model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            random_state=self.random_state,
            max_iter=1000
        )
        
        self.model.fit(X_scaled, y)
        self.feature_names = feature_names
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict fraud (1 for fraud, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Return probability of fraud (positive class)
        return probabilities[:, 1]
    
    def get_feature_importance(self):
        """
        Get feature importance based on coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        if self.feature_names:
            coefficients = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, coefficients))
        else:
            return {}

class RandomForestModel(FraudDetectionModel):
    """
    Random Forest model for supervised fraud detection
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y, feature_names=None):
        """
        Fit Random Forest model
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        # Scale features (optional for Random Forest but good for consistency)
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled, y)
        self.feature_names = feature_names
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict fraud (1 for fraud, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Return probability of fraud (positive class)
        return probabilities[:, 1]
    
    def get_feature_importance(self):
        """
        Get feature importance from Random Forest
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        if self.feature_names:
            importance_scores = self.model.feature_importances_
            return dict(zip(self.feature_names, importance_scores))
        else:
            return {}

class EnsembleModel:
    """
    Ensemble model combining multiple fraud detection models
    """
    
    def __init__(self, models=None):
        self.models = models or []
        self.weights = None
        self.is_fitted = False
        
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append({'model': model, 'weight': weight})
        
    def fit(self, X, y=None, feature_names=None):
        """
        Fit all models in the ensemble
        """
        for model_info in self.models:
            model = model_info['model']
            if y is not None:
                model.fit(X, y, feature_names)
            else:
                model.fit(X, feature_names=feature_names)
        
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """
        Make ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
            
        predictions = []
        weights = []
        
        for model_info in self.models:
            model = model_info['model']
            weight = model_info['weight']
            
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(weight)
        
        # Weighted average of predictions
        weighted_predictions = np.average(predictions, axis=0, weights=weights)
        
        # Convert to binary predictions (threshold at 0.5)
        return (weighted_predictions >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get ensemble prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
            
        probabilities = []
        weights = []
        
        for model_info in self.models:
            model = model_info['model']
            weight = model_info['weight']
            
            prob = model.predict_proba(X)
            probabilities.append(prob)
            weights.append(weight)
        
        # Weighted average of probabilities
        weighted_probabilities = np.average(probabilities, axis=0, weights=weights)
        
        return weighted_probabilities

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fraud detection model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC
    try:
        auc_score = roc_auc_score(y_test, y_pred_proba)
    except:
        auc_score = None
    
    # Precision-Recall curve
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = np.trapz(recall, precision)
    except:
        pr_auc = None
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_auc': auc_score,
        'pr_auc': pr_auc,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba
    }

def cross_validate_model(model_class, X, y, cv=5, **model_params):
    """
    Perform cross-validation on a model
    
    Args:
        model_class: Model class to use
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        **model_params: Parameters for model initialization
        
    Returns:
        Dictionary with cross-validation results
    """
    # Initialize model
    model = model_class(**model_params)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model.model, X, y, cv=cv, scoring='roc_auc')
    
    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std()
    }

def tune_isolation_forest(X, contamination_range=(0.01, 0.2), n_estimators_range=(50, 200)):
    """
    Simple hyperparameter tuning for Isolation Forest
    
    Args:
        X: Feature matrix
        contamination_range: Range of contamination values to try
        n_estimators_range: Range of n_estimators values to try
        
    Returns:
        Best parameters and model
    """
    best_score = -float('inf')
    best_params = None
    best_model = None
    
    contamination_values = np.linspace(contamination_range[0], contamination_range[1], 5)
    n_estimators_values = np.linspace(n_estimators_range[0], n_estimators_range[1], 5, dtype=int)
    
    for contamination in contamination_values:
        for n_estimators in n_estimators_values:
            model = IsolationForestModel(contamination=contamination, n_estimators=n_estimators)
            model.fit(X)
            
            # Use silhouette score as a proxy for quality
            predictions = model.predict(X)
            
            # Simple scoring based on contamination rate achieved
            actual_contamination = predictions.mean()
            score = -abs(actual_contamination - contamination)  # Closer to target contamination is better
            
            if score > best_score:
                best_score = score
                best_params = {'contamination': contamination, 'n_estimators': n_estimators}
                best_model = model
    
    return best_params, best_model
