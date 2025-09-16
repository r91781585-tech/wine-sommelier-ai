#!/usr/bin/env python3
"""
Ensemble Wine Quality Model
Advanced ensemble methods for superior wine quality prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from wine_predictor import WineQualityPredictor

class WineEnsembleModel:
    """Advanced ensemble model combining multiple ML algorithms"""
    
    def __init__(self):
        self.base_predictor = WineQualityPredictor()
        self.voting_classifier = None
        self.stacking_classifier = None
        self.ensemble_scores = {}
        
    def create_voting_ensemble(self):
        """Create voting classifier ensemble"""
        self.base_predictor.initialize_models()
        
        # Select best performing models for ensemble
        estimators = [
            ('rf', self.base_predictor.models['random_forest']),
            ('xgb', self.base_predictor.models['xgboost']),
            ('gb', self.base_predictor.models['gradient_boosting'])
        ]
        
        # Hard voting ensemble
        self.voting_classifier = VotingClassifier(
            estimators=estimators,
            voting='hard'
        )
        
        print("🗳️ Created voting ensemble with Random Forest, XGBoost, and Gradient Boosting")
        return self.voting_classifier
    
    def create_stacking_ensemble(self):
        """Create stacking classifier ensemble"""
        self.base_predictor.initialize_models()
        
        # Base models
        base_models = [
            ('rf', self.base_predictor.models['random_forest']),
            ('xgb', self.base_predictor.models['xgboost']),
            ('gb', self.base_predictor.models['gradient_boosting']),
            ('nn', self.base_predictor.models['neural_network'])
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        # Stacking ensemble
        self.stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba'
        )
        
        print("🏗️ Created stacking ensemble with Logistic Regression meta-learner")
        return self.stacking_classifier
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train all ensemble models and compare performance"""
        print("\n🎯 Training Ensemble Models...")
        print("=" * 50)
        
        # Train individual models first
        self.base_predictor.train_all_models(X_train, y_train, X_test, y_test)
        
        # Create and train voting ensemble
        if self.voting_classifier is None:
            self.create_voting_ensemble()
        
        print("Training voting ensemble...")
        self.voting_classifier.fit(X_train, y_train)
        voting_pred = self.voting_classifier.predict(X_test)
        voting_accuracy = accuracy_score(y_test, voting_pred)
        self.ensemble_scores['voting'] = voting_accuracy
        
        # Create and train stacking ensemble
        if self.stacking_classifier is None:
            self.create_stacking_ensemble()
        
        print("Training stacking ensemble...")
        self.stacking_classifier.fit(X_train, y_train)
        stacking_pred = self.stacking_classifier.predict(X_test)
        stacking_accuracy = accuracy_score(y_test, stacking_pred)
        self.ensemble_scores['stacking'] = stacking_accuracy
        
        # Cross-validation scores
        voting_cv = cross_val_score(self.voting_classifier, X_train, y_train, cv=5)
        stacking_cv = cross_val_score(self.stacking_classifier, X_train, y_train, cv=5)
        
        print(f"\n📊 Ensemble Results:")
        print(f"  Voting Ensemble:   Accuracy = {voting_accuracy:.3f}, CV = {voting_cv.mean():.3f} (±{voting_cv.std()*2:.3f})")
        print(f"  Stacking Ensemble: Accuracy = {stacking_accuracy:.3f}, CV = {stacking_cv.mean():.3f} (±{stacking_cv.std()*2:.3f})")
        
        # Compare with best individual model
        best_individual = max(self.base_predictor.model_scores.values())
        print(f"  Best Individual:   Accuracy = {best_individual:.3f}")
        
        # Determine best overall model
        all_scores = {**self.base_predictor.model_scores, **self.ensemble_scores}
        self.best_model_name = max(all_scores, key=all_scores.get)
        self.best_score = all_scores[self.best_model_name]
        
        print(f"\n🏆 Best Overall Model: {self.best_model_name} (Accuracy: {self.best_score:.3f})")
        
        return self.ensemble_scores
    
    def predict_with_ensemble(self, wine_features, method='stacking'):
        """Make prediction using ensemble method"""
        if method == 'voting' and self.voting_classifier is not None:
            model = self.voting_classifier
        elif method == 'stacking' and self.stacking_classifier is not None:
            model = self.stacking_classifier
        else:
            raise ValueError(f"Ensemble method '{method}' not available or not trained")
        
        # Make prediction
        quality_pred = model.predict([wine_features])[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            quality_proba = model.predict_proba([wine_features])[0]
        else:
            quality_proba = None
        
        return quality_pred, quality_proba
    
    def get_ensemble_feature_importance(self, feature_names, method='stacking'):
        """Get feature importance from ensemble model"""
        if method == 'stacking' and self.stacking_classifier is not None:
            # For stacking, we can get importance from the base models
            importances = []
            
            for name, model in self.stacking_classifier.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                # Average importance across base models
                avg_importance = np.mean(importances, axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                
                return importance_df
        
        elif method == 'voting' and self.voting_classifier is not None:
            # For voting, average importance from tree-based models
            importances = []
            
            for name, model in self.voting_classifier.named_estimators_.items():
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importance
                }).sort_values('importance', ascending=False)
                
                return importance_df
        
        print(f"Feature importance not available for {method} ensemble")
        return None
    
    def advanced_prediction_analysis(self, wine_features, feature_names):
        """Comprehensive prediction analysis using all models"""
        print("\n🔬 ADVANCED PREDICTION ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        # Individual model predictions
        for name, model in self.base_predictor.models.items():
            pred = model.predict([wine_features])[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([wine_features])[0]
                confidence = max(proba)
            else:
                confidence = None
            
            results[name] = {'prediction': pred, 'confidence': confidence}
            print(f"  {name:<18}: Quality {pred}/9 (Confidence: {confidence:.1%})")
        
        # Ensemble predictions
        if self.voting_classifier is not None:
            voting_pred, voting_proba = self.predict_with_ensemble(wine_features, 'voting')
            voting_conf = max(voting_proba) if voting_proba is not None else None
            results['voting_ensemble'] = {'prediction': voting_pred, 'confidence': voting_conf}
            print(f"  {'Voting Ensemble':<18}: Quality {voting_pred}/9 (Confidence: {voting_conf:.1%})")
        
        if self.stacking_classifier is not None:
            stacking_pred, stacking_proba = self.predict_with_ensemble(wine_features, 'stacking')
            stacking_conf = max(stacking_proba) if stacking_proba is not None else None
            results['stacking_ensemble'] = {'prediction': stacking_pred, 'confidence': stacking_conf}
            print(f"  {'Stacking Ensemble':<18}: Quality {stacking_pred}/9 (Confidence: {stacking_conf:.1%})")
        
        # Consensus analysis
        predictions = [r['prediction'] for r in results.values()]
        consensus_pred = max(set(predictions), key=predictions.count)
        agreement_rate = predictions.count(consensus_pred) / len(predictions)
        
        print(f"\n🤝 Consensus Prediction: Quality {consensus_pred}/9")
        print(f"   Model Agreement: {agreement_rate:.1%} ({predictions.count(consensus_pred)}/{len(predictions)} models)")
        
        # Prediction distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        print(f"\n📊 Prediction Distribution:")
        for quality, count in pred_counts.items():
            bar = "█" * count + "░" * (len(predictions) - count)
            print(f"   Quality {quality}: {bar} ({count} models)")
        
        return results, consensus_pred, agreement_rate
    
    def save_ensemble_models(self, filepath_prefix='wine_ensemble'):
        """Save ensemble models to disk"""
        if self.voting_classifier is not None:
            joblib.dump(self.voting_classifier, f"{filepath_prefix}_voting.joblib")
            print(f"💾 Saved voting ensemble to {filepath_prefix}_voting.joblib")
        
        if self.stacking_classifier is not None:
            joblib.dump(self.stacking_classifier, f"{filepath_prefix}_stacking.joblib")
            print(f"💾 Saved stacking ensemble to {filepath_prefix}_stacking.joblib")
        
        # Save base models
        self.base_predictor.save_models(filepath_prefix + "_base")
    
    def load_ensemble_models(self, filepath_prefix='wine_ensemble'):
        """Load ensemble models from disk"""
        try:
            self.voting_classifier = joblib.load(f"{filepath_prefix}_voting.joblib")
            print(f"📂 Loaded voting ensemble from {filepath_prefix}_voting.joblib")
        except FileNotFoundError:
            print(f"⚠️ Voting ensemble file not found")
        
        try:
            self.stacking_classifier = joblib.load(f"{filepath_prefix}_stacking.joblib")
            print(f"📂 Loaded stacking ensemble from {filepath_prefix}_stacking.joblib")
        except FileNotFoundError:
            print(f"⚠️ Stacking ensemble file not found")
        
        # Load base models
        self.base_predictor.load_models(filepath_prefix + "_base")

class WineQualityConfidenceAnalyzer:
    """Analyze prediction confidence and uncertainty"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
    
    def analyze_prediction_confidence(self, wine_features):
        """Analyze confidence levels across all models"""
        confidences = []
        predictions = []
        
        # Get predictions from all models
        for name, model in self.ensemble_model.base_predictor.models.items():
            pred = model.predict([wine_features])[0]
            predictions.append(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([wine_features])[0]
                confidence = max(proba)
                confidences.append(confidence)
        
        # Calculate metrics
        avg_confidence = np.mean(confidences) if confidences else 0
        std_confidence = np.std(confidences) if confidences else 0
        prediction_variance = len(set(predictions))
        
        # Confidence level assessment
        if avg_confidence > 0.8 and prediction_variance <= 2:
            confidence_level = "High"
            reliability = "Very Reliable"
        elif avg_confidence > 0.6 and prediction_variance <= 3:
            confidence_level = "Medium"
            reliability = "Reliable"
        else:
            confidence_level = "Low"
            reliability = "Use with Caution"
        
        return {
            'average_confidence': avg_confidence,
            'confidence_std': std_confidence,
            'prediction_variance': prediction_variance,
            'confidence_level': confidence_level,
            'reliability': reliability,
            'individual_confidences': confidences,
            'individual_predictions': predictions
        }
    
    def uncertainty_quantification(self, wine_features, n_bootstrap=100):
        """Quantify prediction uncertainty using bootstrap sampling"""
        bootstrap_predictions = []
        
        # Bootstrap sampling for uncertainty estimation
        for _ in range(n_bootstrap):
            # Add small random noise to features
            noisy_features = wine_features + np.random.normal(0, 0.01, len(wine_features))
            
            # Get prediction from best model
            if self.ensemble_model.stacking_classifier is not None:
                pred = self.ensemble_model.stacking_classifier.predict([noisy_features])[0]
            else:
                pred = self.ensemble_model.base_predictor.best_model.predict([noisy_features])[0]
            
            bootstrap_predictions.append(pred)
        
        # Calculate uncertainty metrics
        pred_mean = np.mean(bootstrap_predictions)
        pred_std = np.std(bootstrap_predictions)
        pred_range = (min(bootstrap_predictions), max(bootstrap_predictions))
        
        return {
            'mean_prediction': pred_mean,
            'prediction_std': pred_std,
            'prediction_range': pred_range,
            'uncertainty_level': 'High' if pred_std > 0.5 else 'Medium' if pred_std > 0.2 else 'Low'
        }

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Ensemble Model - Testing")
    
    ensemble = WineEnsembleModel()
    ensemble.create_voting_ensemble()
    ensemble.create_stacking_ensemble()
    
    print("✅ Wine ensemble models initialized successfully!")
    print("Use these models through the main wine_analyzer.py script.")