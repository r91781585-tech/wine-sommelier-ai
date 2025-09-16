#!/usr/bin/env python3
"""
Wine Sommelier AI - Main Analysis Engine
Comprehensive wine quality analysis with ML predictions and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class WineSommelierAI:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_data(self, file_path=None):
        """Load wine quality dataset"""
        if file_path:
            self.data = pd.read_csv(file_path)
        else:
            # Generate synthetic wine data for demo
            np.random.seed(42)
            n_samples = 1000
            
            self.data = pd.DataFrame({
                'fixed_acidity': np.random.normal(8.3, 1.7, n_samples),
                'volatile_acidity': np.random.normal(0.5, 0.2, n_samples),
                'citric_acid': np.random.normal(0.3, 0.15, n_samples),
                'residual_sugar': np.random.exponential(2.5, n_samples),
                'chlorides': np.random.normal(0.08, 0.05, n_samples),
                'free_sulfur_dioxide': np.random.normal(15, 10, n_samples),
                'total_sulfur_dioxide': np.random.normal(46, 32, n_samples),
                'density': np.random.normal(0.996, 0.002, n_samples),
                'pH': np.random.normal(3.3, 0.15, n_samples),
                'sulphates': np.random.normal(0.65, 0.17, n_samples),
                'alcohol': np.random.normal(10.4, 1.1, n_samples)
            })
            
            # Create quality scores based on feature combinations
            quality_score = (
                (self.data['alcohol'] - 8) * 0.3 +
                (12 - self.data['volatile_acidity'] * 10) * 0.2 +
                (self.data['sulphates'] * 10) * 0.15 +
                np.random.normal(0, 1, n_samples) * 0.5
            )
            
            # Convert to quality ratings (3-9 scale)
            self.data['quality'] = np.clip(
                np.round(quality_score + 6).astype(int), 3, 9
            )
        
        print(f"📊 Loaded wine dataset: {self.data.shape[0]} samples, {self.data.shape[1]} features")
        return self.data
    
    def explore_data(self):
        """Comprehensive data exploration"""
        print("\n🔍 WINE DATASET EXPLORATION")
        print("=" * 50)
        
        print(f"Dataset Shape: {self.data.shape}")
        print(f"\nQuality Distribution:")
        print(self.data['quality'].value_counts().sort_index())
        
        print(f"\nBasic Statistics:")
        print(self.data.describe().round(2))
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            print(f"\nMissing Values:\n{missing[missing > 0]}")
        else:
            print("\n✅ No missing values found")
    
    def create_visualizations(self):
        """Generate comprehensive wine analysis visualizations"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Quality Distribution
        plt.subplot(3, 4, 1)
        self.data['quality'].hist(bins=7, alpha=0.7, color='darkred', edgecolor='black')
        plt.title('🍷 Wine Quality Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        
        # 2. Correlation Heatmap
        plt.subplot(3, 4, 2)
        corr_matrix = self.data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('🔗 Feature Correlations', fontsize=12, fontweight='bold')
        
        # 3. Alcohol vs Quality
        plt.subplot(3, 4, 3)
        sns.boxplot(data=self.data, x='quality', y='alcohol', palette='viridis')
        plt.title('🥃 Alcohol Content by Quality', fontsize=12, fontweight='bold')
        
        # 4. Volatile Acidity vs Quality
        plt.subplot(3, 4, 4)
        sns.boxplot(data=self.data, x='quality', y='volatile_acidity', palette='plasma')
        plt.title('🧪 Volatile Acidity by Quality', fontsize=12, fontweight='bold')
        
        # 5. pH Distribution
        plt.subplot(3, 4, 5)
        self.data['pH'].hist(bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title('⚗️ pH Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('pH Level')
        
        # 6. Sulphates vs Quality
        plt.subplot(3, 4, 6)
        sns.scatterplot(data=self.data, x='sulphates', y='quality', 
                       alpha=0.6, color='purple')
        plt.title('🧂 Sulphates vs Quality', fontsize=12, fontweight='bold')
        
        # 7. Residual Sugar Distribution
        plt.subplot(3, 4, 7)
        self.data['residual_sugar'].hist(bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('🍯 Residual Sugar Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Residual Sugar (g/L)')
        
        # 8. Density vs Alcohol
        plt.subplot(3, 4, 8)
        sns.scatterplot(data=self.data, x='density', y='alcohol', 
                       hue='quality', palette='coolwarm', alpha=0.7)
        plt.title('⚖️ Density vs Alcohol', fontsize=12, fontweight='bold')
        
        # 9. Chlorides Distribution
        plt.subplot(3, 4, 9)
        self.data['chlorides'].hist(bins=30, alpha=0.7, color='cyan', edgecolor='black')
        plt.title('🧂 Chlorides Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Chlorides (g/L)')
        
        # 10. Free vs Total Sulfur Dioxide
        plt.subplot(3, 4, 10)
        sns.scatterplot(data=self.data, x='free_sulfur_dioxide', y='total_sulfur_dioxide',
                       hue='quality', palette='viridis', alpha=0.7)
        plt.title('💨 Sulfur Dioxide Relationship', fontsize=12, fontweight='bold')
        
        # 11. Citric Acid vs Fixed Acidity
        plt.subplot(3, 4, 11)
        sns.scatterplot(data=self.data, x='citric_acid', y='fixed_acidity',
                       hue='quality', palette='plasma', alpha=0.7)
        plt.title('🍋 Acidity Relationship', fontsize=12, fontweight='bold')
        
        # 12. Quality Score Summary
        plt.subplot(3, 4, 12)
        quality_stats = self.data.groupby('quality').size()
        plt.pie(quality_stats.values, labels=quality_stats.index, autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(quality_stats))))
        plt.title('🎯 Quality Score Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('wine_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Comprehensive visualizations created and saved as 'wine_analysis_dashboard.png'")
    
    def prepare_data(self):
        """Prepare data for machine learning"""
        # Separate features and target
        self.feature_names = [col for col in self.data.columns if col != 'quality']
        X = self.data[self.feature_names]
        y = self.data['quality']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"🔧 Data prepared: {self.X_train.shape[0]} training, {self.X_test.shape[0]} testing samples")
    
    def train_model(self):
        """Train Random Forest model for wine quality prediction"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"🎯 Model trained! Accuracy: {accuracy:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return accuracy, feature_importance
    
    def predict_wine_quality(self, wine_features):
        """Predict quality for new wine samples"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Scale the input features
        wine_scaled = self.scaler.transform([wine_features])
        
        # Make prediction
        quality_pred = self.model.predict(wine_scaled)[0]
        quality_proba = self.model.predict_proba(wine_scaled)[0]
        
        return quality_pred, quality_proba
    
    def generate_wine_report(self, wine_features):
        """Generate a sommelier-style wine analysis report"""
        quality_pred, quality_proba = self.predict_wine_quality(wine_features)
        
        # Wine characteristics analysis
        alcohol = wine_features[10]  # alcohol is the last feature
        acidity = wine_features[1]   # volatile acidity
        sweetness = wine_features[3] # residual sugar
        
        print(f"\n🍷 WINE SOMMELIER ANALYSIS REPORT")
        print("=" * 50)
        print(f"Predicted Quality Score: {quality_pred}/9")
        print(f"Confidence: {max(quality_proba):.1%}")
        
        # Style classification
        if alcohol > 12:
            style = "Full-bodied"
        elif alcohol > 10:
            style = "Medium-bodied"
        else:
            style = "Light-bodied"
        
        if sweetness > 5:
            sweetness_level = "Sweet"
        elif sweetness > 2:
            sweetness_level = "Off-dry"
        else:
            sweetness_level = "Dry"
        
        if acidity > 0.6:
            acidity_level = "High acidity"
        elif acidity > 0.4:
            acidity_level = "Medium acidity"
        else:
            acidity_level = "Low acidity"
        
        print(f"\n📝 Wine Profile:")
        print(f"  Style: {style}")
        print(f"  Sweetness: {sweetness_level}")
        print(f"  Acidity: {acidity_level}")
        print(f"  Alcohol: {alcohol:.1f}%")
        
        # Quality assessment
        if quality_pred >= 7:
            assessment = "Excellent wine with outstanding characteristics"
        elif quality_pred >= 6:
            assessment = "Good quality wine with pleasant attributes"
        elif quality_pred >= 5:
            assessment = "Average wine suitable for casual drinking"
        else:
            assessment = "Below average wine with some defects"
        
        print(f"\n🎯 Quality Assessment: {assessment}")
        
        return quality_pred, style, sweetness_level, acidity_level

def main():
    """Main execution function"""
    print("🍷 Welcome to Wine Sommelier AI!")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = WineSommelierAI()
    
    # Load and explore data
    analyzer.load_data()
    analyzer.explore_data()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Prepare and train model
    analyzer.prepare_data()
    accuracy, feature_importance = analyzer.train_model()
    
    # Example wine prediction
    print(f"\n🧪 TESTING WINE PREDICTION")
    print("=" * 30)
    
    # Sample wine features (in order of the dataset columns)
    sample_wine = [
        7.4,    # fixed_acidity
        0.7,    # volatile_acidity
        0.0,    # citric_acid
        1.9,    # residual_sugar
        0.076,  # chlorides
        11.0,   # free_sulfur_dioxide
        34.0,   # total_sulfur_dioxide
        0.9978, # density
        3.51,   # pH
        0.56,   # sulphates
        9.4     # alcohol
    ]
    
    analyzer.generate_wine_report(sample_wine)
    
    print(f"\n✨ Analysis complete! Check 'wine_analysis_dashboard.png' for visualizations.")

if __name__ == "__main__":
    main()