#!/usr/bin/env python3
"""
Wine Quality Prediction Models
Multiple ML algorithms for wine quality assessment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class WineQualityPredictor:
    """Advanced wine quality prediction with multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_scores = {}
        
    def initialize_models(self):
        """Initialize different ML models for comparison"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
        }
        
        print("🤖 Initialized 5 ML models for wine quality prediction")
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train and evaluate all models"""
        print("\n🏋️ Training all models...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self.model_scores[name] = accuracy
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            print(f"  ✅ {name}: Accuracy = {accuracy:.3f}, CV Score = {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
        # Find best model
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name} (Accuracy: {self.model_scores[self.best_model_name]:.3f})")
        
        return self.model_scores
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='random_forest'):
        """Perform hyperparameter tuning for specified model"""
        print(f"\n🔧 Hyperparameter tuning for {model_name}...")
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBClassifier(random_state=42)
            
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def get_feature_importance(self, feature_names, model_name=None):
        """Get feature importance from tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Feature Importance ({model_name}):")
            print("=" * 40)
            for idx, row in importance_df.iterrows():
                print(f"  {row['feature']:<20}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def predict_wine_quality(self, wine_features, model_name=None):
        """Predict wine quality using specified model"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models[model_name]
        
        # Make prediction
        quality_pred = model.predict([wine_features])[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            quality_proba = model.predict_proba([wine_features])[0]
        else:
            quality_proba = None
        
        return quality_pred, quality_proba, model_name
    
    def ensemble_prediction(self, wine_features, models_to_use=None):
        """Make ensemble prediction using multiple models"""
        if models_to_use is None:
            models_to_use = ['random_forest', 'xgboost', 'gradient_boosting']
        
        predictions = []
        probabilities = []
        
        for model_name in models_to_use:
            if model_name in self.models:
                model = self.models[model_name]
                pred = model.predict([wine_features])[0]
                predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([wine_features])[0]
                    probabilities.append(proba)
        
        # Ensemble prediction (majority vote)
        ensemble_pred = max(set(predictions), key=predictions.count)
        
        # Average probabilities if available
        if probabilities:
            avg_proba = np.mean(probabilities, axis=0)
        else:
            avg_proba = None
        
        return ensemble_pred, avg_proba, predictions
    
    def save_models(self, filepath_prefix='wine_models'):
        """Save trained models to disk"""
        for name, model in self.models.items():
            filename = f"{filepath_prefix}_{name}.joblib"
            joblib.dump(model, filename)
            print(f"💾 Saved {name} model to {filename}")
    
    def load_models(self, filepath_prefix='wine_models'):
        """Load trained models from disk"""
        model_names = ['random_forest', 'xgboost', 'gradient_boosting', 'neural_network', 'svm']
        
        for name in model_names:
            try:
                filename = f"{filepath_prefix}_{name}.joblib"
                self.models[name] = joblib.load(filename)
                print(f"📂 Loaded {name} model from {filename}")
            except FileNotFoundError:
                print(f"⚠️ Model file {filename} not found")
    
    def model_comparison_report(self):
        """Generate comprehensive model comparison report"""
        print("\n📊 MODEL COMPARISON REPORT")
        print("=" * 50)
        
        sorted_models = sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(sorted_models, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{status} {i}. {model_name:<20}: {score:.4f}")
        
        return sorted_models

class WineStyleClassifier:
    """Classify wines into different style categories"""
    
    def __init__(self):
        self.style_categories = {
            'light_white': {'alcohol': (8, 11), 'residual_sugar': (0, 4), 'body': 'light'},
            'full_white': {'alcohol': (12, 15), 'residual_sugar': (0, 4), 'body': 'full'},
            'sweet_white': {'alcohol': (8, 12), 'residual_sugar': (5, 20), 'body': 'medium'},
            'light_red': {'alcohol': (8, 12), 'tannins': 'low', 'body': 'light'},
            'medium_red': {'alcohol': (12, 13.5), 'tannins': 'medium', 'body': 'medium'},
            'full_red': {'alcohol': (13.5, 16), 'tannins': 'high', 'body': 'full'},
            'dessert': {'alcohol': (15, 20), 'residual_sugar': (10, 50), 'body': 'rich'}
        }
    
    def classify_wine_style(self, wine_features):
        """Classify wine into style category based on features"""
        alcohol = wine_features[10]  # alcohol content
        residual_sugar = wine_features[3]  # residual sugar
        acidity = wine_features[1]  # volatile acidity (proxy for tannins in reds)
        
        # Determine wine style
        if residual_sugar > 10:
            return 'dessert', 'Sweet dessert wine with rich characteristics'
        elif residual_sugar > 5:
            return 'sweet_white', 'Off-dry to sweet white wine'
        elif alcohol > 13.5:
            if acidity > 0.6:  # Higher acidity suggests red wine characteristics
                return 'full_red', 'Full-bodied red wine with robust structure'
            else:
                return 'full_white', 'Full-bodied white wine with rich texture'
        elif alcohol > 12:
            if acidity > 0.5:
                return 'medium_red', 'Medium-bodied red wine with balanced structure'
            else:
                return 'full_white', 'Full-bodied white wine'
        else:
            if acidity > 0.4:
                return 'light_red', 'Light-bodied red wine with delicate structure'
            else:
                return 'light_white', 'Light-bodied white wine with crisp character'
    
    def get_style_recommendations(self, style_category):
        """Get food pairing and serving recommendations for wine style"""
        recommendations = {
            'light_white': {
                'food_pairings': ['Seafood', 'Salads', 'Light appetizers', 'Sushi'],
                'serving_temp': '45-50°F (7-10°C)',
                'glassware': 'White wine glass',
                'examples': ['Sauvignon Blanc', 'Pinot Grigio', 'Albariño']
            },
            'full_white': {
                'food_pairings': ['Roasted chicken', 'Creamy pasta', 'Lobster', 'Aged cheeses'],
                'serving_temp': '50-55°F (10-13°C)',
                'glassware': 'Chardonnay glass',
                'examples': ['Chardonnay', 'White Rioja', 'Viognier']
            },
            'sweet_white': {
                'food_pairings': ['Spicy cuisine', 'Fruit desserts', 'Blue cheese', 'Foie gras'],
                'serving_temp': '45-50°F (7-10°C)',
                'glassware': 'White wine glass',
                'examples': ['Riesling', 'Moscato', 'Gewürztraminer']
            },
            'light_red': {
                'food_pairings': ['Salmon', 'Mushroom dishes', 'Soft cheeses', 'Charcuterie'],
                'serving_temp': '55-60°F (13-15°C)',
                'glassware': 'Pinot Noir glass',
                'examples': ['Pinot Noir', 'Beaujolais', 'Dolcetto']
            },
            'medium_red': {
                'food_pairings': ['Grilled meats', 'Pasta with tomato sauce', 'Pizza', 'Medium cheeses'],
                'serving_temp': '60-65°F (15-18°C)',
                'glassware': 'Bordeaux glass',
                'examples': ['Merlot', 'Sangiovese', 'Tempranillo']
            },
            'full_red': {
                'food_pairings': ['Red meat', 'Game', 'Aged cheeses', 'Dark chocolate'],
                'serving_temp': '65-68°F (18-20°C)',
                'glassware': 'Bordeaux glass',
                'examples': ['Cabernet Sauvignon', 'Syrah', 'Barolo']
            },
            'dessert': {
                'food_pairings': ['Chocolate desserts', 'Fruit tarts', 'Nuts', 'Strong cheeses'],
                'serving_temp': '55-60°F (13-15°C)',
                'glassware': 'Dessert wine glass',
                'examples': ['Port', 'Sauternes', 'Ice Wine']
            }
        }
        
        return recommendations.get(style_category, {})

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Quality Predictor - Model Testing")
    
    # This would typically be called from the main analyzer
    predictor = WineQualityPredictor()
    predictor.initialize_models()
    
    print("✅ Wine prediction models initialized successfully!")
    print("Use these models through the main wine_analyzer.py script.")