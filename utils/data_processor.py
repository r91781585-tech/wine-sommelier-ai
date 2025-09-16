#!/usr/bin/env python3
"""
Wine Data Processing Utilities
Advanced data preprocessing and feature engineering for wine analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class WineDataProcessor:
    """Comprehensive wine data preprocessing and feature engineering"""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.feature_selector = None
        self.pca = None
        self.kmeans = None
        self.feature_names = None
        
    def load_wine_dataset(self, file_path=None, dataset_type='red'):
        """Load wine quality dataset from file or generate synthetic data"""
        if file_path:
            try:
                data = pd.read_csv(file_path)
                print(f"📂 Loaded wine dataset from {file_path}")
            except FileNotFoundError:
                print(f"⚠️ File {file_path} not found. Generating synthetic data...")
                data = self._generate_synthetic_wine_data(dataset_type)
        else:
            data = self._generate_synthetic_wine_data(dataset_type)
        
        self.feature_names = [col for col in data.columns if col != 'quality']
        return data
    
    def _generate_synthetic_wine_data(self, dataset_type='red', n_samples=1500):
        """Generate realistic synthetic wine data"""
        np.random.seed(42)
        
        if dataset_type == 'red':
            # Red wine characteristics
            data = pd.DataFrame({
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
        else:  # white wine
            data = pd.DataFrame({
                'fixed_acidity': np.random.normal(6.8, 0.8, n_samples),
                'volatile_acidity': np.random.normal(0.28, 0.1, n_samples),
                'citric_acid': np.random.normal(0.33, 0.12, n_samples),
                'residual_sugar': np.random.exponential(6.4, n_samples),
                'chlorides': np.random.normal(0.045, 0.02, n_samples),
                'free_sulfur_dioxide': np.random.normal(35, 17, n_samples),
                'total_sulfur_dioxide': np.random.normal(138, 42, n_samples),
                'density': np.random.normal(0.994, 0.003, n_samples),
                'pH': np.random.normal(3.2, 0.16, n_samples),
                'sulphates': np.random.normal(0.49, 0.11, n_samples),
                'alcohol': np.random.normal(10.5, 1.2, n_samples)
            })
        
        # Ensure realistic ranges
        data = data.clip(lower=0)  # No negative values
        data['pH'] = data['pH'].clip(2.5, 4.0)  # Realistic pH range
        data['alcohol'] = data['alcohol'].clip(8.0, 15.0)  # Realistic alcohol range
        
        # Create quality scores based on feature combinations
        quality_score = (
            (data['alcohol'] - 8) * 0.3 +
            (12 - data['volatile_acidity'] * 10) * 0.2 +
            (data['sulphates'] * 10) * 0.15 +
            (data['citric_acid'] * 5) * 0.1 +
            np.random.normal(0, 1, n_samples) * 0.5
        )
        
        # Convert to quality ratings (3-9 scale)
        data['quality'] = np.clip(np.round(quality_score + 6).astype(int), 3, 9)
        
        print(f"🧪 Generated synthetic {dataset_type} wine dataset: {n_samples} samples")
        return data
    
    def basic_preprocessing(self, data):
        """Basic data cleaning and preprocessing"""
        print("🧹 Performing basic preprocessing...")
        
        # Handle missing values
        if data.isnull().sum().any():
            print("  Handling missing values...")
            # Fill numerical columns with median
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if data[col].isnull().any():
                    data[col].fillna(data[col].median(), inplace=True)
        
        # Remove outliers using IQR method
        print("  Removing outliers...")
        for col in self.feature_names:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(data)
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            outliers_removed = outliers_before - len(data)
            
            if outliers_removed > 0:
                print(f"    Removed {outliers_removed} outliers from {col}")
        
        print(f"  ✅ Preprocessing complete. Final dataset: {len(data)} samples")
        return data
    
    def advanced_feature_engineering(self, data):
        """Create advanced engineered features"""
        print("⚙️ Engineering advanced features...")
        
        # Ratio features
        data['alcohol_to_density'] = data['alcohol'] / data['density']
        data['total_acidity'] = data['fixed_acidity'] + data['volatile_acidity']
        data['free_sulfur_ratio'] = data['free_sulfur_dioxide'] / data['total_sulfur_dioxide']
        data['sugar_to_alcohol'] = data['residual_sugar'] / data['alcohol']
        
        # Interaction features
        data['acidity_alcohol_interaction'] = data['total_acidity'] * data['alcohol']
        data['sulphates_alcohol_interaction'] = data['sulphates'] * data['alcohol']
        
        # Polynomial features for key variables
        data['alcohol_squared'] = data['alcohol'] ** 2
        data['volatile_acidity_squared'] = data['volatile_acidity'] ** 2
        
        # Binned features
        data['alcohol_category'] = pd.cut(data['alcohol'], 
                                        bins=[0, 9, 11, 13, 20], 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
        
        data['acidity_category'] = pd.cut(data['volatile_acidity'], 
                                        bins=[0, 0.3, 0.6, 2.0], 
                                        labels=['Low', 'Medium', 'High'])
        
        # Convert categorical to numerical
        data['alcohol_category_num'] = data['alcohol_category'].cat.codes
        data['acidity_category_num'] = data['acidity_category'].cat.codes
        
        # Drop categorical columns for ML
        data = data.drop(['alcohol_category', 'acidity_category'], axis=1)
        
        # Update feature names
        self.feature_names = [col for col in data.columns if col != 'quality']
        
        print(f"  ✅ Created {len(self.feature_names) - 11} new engineered features")
        return data
    
    def scale_features(self, X_train, X_test, method='standard'):
        """Scale features using specified method"""
        if method not in self.scalers:
            raise ValueError(f"Scaling method '{method}' not supported. Use: {list(self.scalers.keys())}")
        
        scaler = self.scalers[method]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📏 Features scaled using {method} scaling")
        return X_train_scaled, X_test_scaled, scaler
    
    def feature_selection(self, X_train, y_train, method='f_classif', k=10):
        """Select top k features using statistical tests"""
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError("Method must be 'f_classif' or 'mutual_info'")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in self.feature_selector.get_support(indices=True)]
        
        print(f"🎯 Selected top {k} features using {method}:")
        for i, feature in enumerate(selected_features):
            score = self.feature_selector.scores_[self.feature_selector.get_support(indices=True)[i]]
            print(f"  {i+1}. {feature}: {score:.3f}")
        
        return X_train_selected, selected_features
    
    def apply_feature_selection(self, X_test):
        """Apply previously fitted feature selection to test data"""
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted. Call feature_selection() first.")
        
        return self.feature_selector.transform(X_test)
    
    def dimensionality_reduction(self, X_train, X_test, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        self.pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        n_components_actual = self.pca.n_components_
        
        print(f"📉 PCA applied: {n_components_actual} components explain {explained_variance:.1%} of variance")
        
        return X_train_pca, X_test_pca
    
    def wine_clustering(self, data, n_clusters=5):
        """Perform wine clustering analysis"""
        print(f"🍇 Performing wine clustering with {n_clusters} clusters...")
        
        # Prepare data for clustering (exclude quality)
        X = data[self.feature_names]
        
        # Scale data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to data
        data_clustered = data.copy()
        data_clustered['cluster'] = clusters
        
        # Analyze clusters
        print("📊 Cluster Analysis:")
        for i in range(n_clusters):
            cluster_data = data_clustered[data_clustered['cluster'] == i]
            avg_quality = cluster_data['quality'].mean()
            cluster_size = len(cluster_data)
            
            print(f"  Cluster {i}: {cluster_size} wines, Avg Quality: {avg_quality:.2f}")
            
            # Key characteristics
            key_features = ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid']
            characteristics = []
            
            for feature in key_features:
                avg_value = cluster_data[feature].mean()
                overall_avg = data[feature].mean()
                
                if avg_value > overall_avg * 1.1:
                    characteristics.append(f"High {feature}")
                elif avg_value < overall_avg * 0.9:
                    characteristics.append(f"Low {feature}")
            
            if characteristics:
                print(f"    Characteristics: {', '.join(characteristics)}")
        
        return data_clustered, clusters
    
    def create_wine_profiles(self, data_clustered):
        """Create detailed wine style profiles based on clusters"""
        profiles = {}
        
        for cluster_id in data_clustered['cluster'].unique():
            cluster_data = data_clustered[data_clustered['cluster'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'avg_quality': cluster_data['quality'].mean(),
                'quality_std': cluster_data['quality'].std(),
                'characteristics': {},
                'style_description': ''
            }
            
            # Calculate average characteristics
            for feature in self.feature_names:
                profile['characteristics'][feature] = {
                    'mean': cluster_data[feature].mean(),
                    'std': cluster_data[feature].std()
                }
            
            # Generate style description
            alcohol = profile['characteristics']['alcohol']['mean']
            acidity = profile['characteristics']['volatile_acidity']['mean']
            sugar = profile['characteristics']['residual_sugar']['mean']
            
            if alcohol > 12:
                body = "Full-bodied"
            elif alcohol > 10:
                body = "Medium-bodied"
            else:
                body = "Light-bodied"
            
            if sugar > 5:
                sweetness = "sweet"
            elif sugar > 2:
                sweetness = "off-dry"
            else:
                sweetness = "dry"
            
            if acidity > 0.6:
                acidity_level = "high acidity"
            elif acidity > 0.4:
                acidity_level = "medium acidity"
            else:
                acidity_level = "low acidity"
            
            profile['style_description'] = f"{body}, {sweetness} wine with {acidity_level}"
            profiles[f'cluster_{cluster_id}'] = profile
        
        return profiles
    
    def data_quality_report(self, data):
        """Generate comprehensive data quality report"""
        print("\n📋 DATA QUALITY REPORT")
        print("=" * 50)
        
        # Basic statistics
        print(f"Dataset Shape: {data.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target Classes: {sorted(data['quality'].unique())}")
        
        # Missing values
        missing = data.isnull().sum()
        if missing.any():
            print(f"\nMissing Values:")
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
        else:
            print("\n✅ No missing values")
        
        # Data types
        print(f"\nData Types:")
        for dtype, cols in data.dtypes.groupby(data.dtypes).items():
            print(f"  {dtype}: {len(cols)} columns")
        
        # Quality distribution
        print(f"\nQuality Distribution:")
        quality_dist = data['quality'].value_counts().sort_index()
        for quality, count in quality_dist.items():
            percentage = count / len(data) * 100
            bar = "█" * int(percentage / 2)
            print(f"  Quality {quality}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        stats = data[self.feature_names].describe()
        print(stats.round(3))
        
        return {
            'shape': data.shape,
            'missing_values': missing.to_dict(),
            'quality_distribution': quality_dist.to_dict(),
            'feature_stats': stats.to_dict()
        }

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Data Processor - Testing")
    
    processor = WineDataProcessor()
    
    # Generate sample data
    data = processor.load_wine_dataset(dataset_type='red')
    
    # Basic preprocessing
    data_clean = processor.basic_preprocessing(data)
    
    # Feature engineering
    data_engineered = processor.advanced_feature_engineering(data_clean)
    
    # Clustering analysis
    data_clustered, clusters = processor.wine_clustering(data_engineered)
    
    # Create profiles
    profiles = processor.create_wine_profiles(data_clustered)
    
    # Data quality report
    quality_report = processor.data_quality_report(data_engineered)
    
    print("\n✅ Wine data processing pipeline completed successfully!")