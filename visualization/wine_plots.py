#!/usr/bin/env python3
"""
Wine Visualization Utilities
Advanced plotting functions for wine quality analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class WineVisualizer:
    """Comprehensive wine data visualization toolkit"""
    
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use(style)
        self.color_palette = ['#722F37', '#C73E1D', '#F39C12', '#27AE60', '#3498DB', '#9B59B6']
        sns.set_palette(self.color_palette)
        
    def create_quality_distribution_plot(self, data, save_path=None):
        """Create wine quality distribution visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        quality_counts = data['quality'].value_counts().sort_index()
        ax1.bar(quality_counts.index, quality_counts.values, 
                color=self.color_palette[0], alpha=0.8, edgecolor='black')
        ax1.set_title('🍷 Wine Quality Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Number of Wines')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(quality_counts.values):
            ax1.text(quality_counts.index[i], v + 5, str(v), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(quality_counts)))
        wedges, texts, autotexts = ax2.pie(quality_counts.values, 
                                          labels=quality_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax2.set_title('🎯 Quality Score Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_correlation_heatmap(self, data, save_path=None):
        """Create feature correlation heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        plt.title('🔗 Wine Feature Correlations', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_feature_importance_plot(self, feature_importance_df, save_path=None):
        """Create feature importance visualization"""
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        importance_sorted = feature_importance_df.sort_values('importance', ascending=True)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(importance_sorted)), importance_sorted['importance'],
                       color=self.color_palette[1], alpha=0.8)
        
        plt.yticks(range(len(importance_sorted)), importance_sorted['feature'])
        plt.xlabel('Feature Importance')
        plt.title('🔍 Feature Importance for Wine Quality Prediction', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_quality_vs_features_plot(self, data, features=None, save_path=None):
        """Create quality vs features box plots"""
        if features is None:
            features = ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features[:4]):
            sns.boxplot(data=data, x='quality', y=feature, 
                       palette='viridis', ax=axes[i])
            axes[i].set_title(f'🍷 Quality vs {feature.replace("_", " ").title()}',
                             fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_alcohol_quality_scatter(self, data, save_path=None):
        """Create alcohol vs quality scatter plot with trend line"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        scatter = plt.scatter(data['alcohol'], data['quality'], 
                            c=data['quality'], cmap='viridis', 
                            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(data['alcohol'], data['quality'], 1)
        p = np.poly1d(z)
        plt.plot(data['alcohol'], p(data['alcohol']), "r--", alpha=0.8, linewidth=2)
        
        plt.colorbar(scatter, label='Quality Score')
        plt.xlabel('Alcohol Content (%)')
        plt.ylabel('Quality Score')
        plt.title('🥃 Alcohol Content vs Wine Quality', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = data['alcohol'].corr(data['quality'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_interactive_3d_plot(self, data, x_feature='alcohol', y_feature='volatile_acidity', 
                                  z_feature='sulphates'):
        """Create interactive 3D scatter plot"""
        fig = px.scatter_3d(data, x=x_feature, y=y_feature, z=z_feature,
                           color='quality', size='quality',
                           hover_data=['quality'],
                           color_continuous_scale='viridis',
                           title=f'🍷 3D Wine Analysis: {x_feature} vs {y_feature} vs {z_feature}')
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_feature.replace('_', ' ').title(),
                yaxis_title=y_feature.replace('_', ' ').title(),
                zaxis_title=z_feature.replace('_', ' ').title()
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_wine_cluster_plot(self, data_clustered, save_path=None):
        """Create wine cluster visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cluster distribution
        cluster_counts = data_clustered['cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values,
                      color=self.color_palette[:len(cluster_counts)], alpha=0.8)
        axes[0, 0].set_title('🍇 Wine Cluster Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Wines')
        
        # Quality by cluster
        sns.boxplot(data=data_clustered, x='cluster', y='quality', 
                   palette='Set2', ax=axes[0, 1])
        axes[0, 1].set_title('🎯 Quality Distribution by Cluster', fontweight='bold')
        
        # Alcohol vs Volatile Acidity by cluster
        for cluster in data_clustered['cluster'].unique():
            cluster_data = data_clustered[data_clustered['cluster'] == cluster]
            axes[1, 0].scatter(cluster_data['alcohol'], cluster_data['volatile_acidity'],
                              label=f'Cluster {cluster}', alpha=0.7, s=50)
        
        axes[1, 0].set_xlabel('Alcohol Content (%)')
        axes[1, 0].set_ylabel('Volatile Acidity')
        axes[1, 0].set_title('🥃 Alcohol vs Volatile Acidity by Cluster', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cluster characteristics heatmap
        cluster_means = data_clustered.groupby('cluster')[
            ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid', 'quality']
        ].mean()
        
        sns.heatmap(cluster_means.T, annot=True, cmap='RdYlBu_r', 
                   center=cluster_means.values.mean(), ax=axes[1, 1])
        axes[1, 1].set_title('🔥 Cluster Characteristics Heatmap', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_distribution_plots(self, data, features=None, save_path=None):
        """Create feature distribution plots"""
        if features is None:
            features = ['alcohol', 'volatile_acidity', 'sulphates', 'pH', 
                       'residual_sugar', 'citric_acid']
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.ravel() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                # Histogram with KDE
                axes[i].hist(data[feature], bins=30, alpha=0.7, 
                           color=self.color_palette[i % len(self.color_palette)],
                           edgecolor='black', density=True)
                
                # Add KDE curve
                from scipy import stats
                kde = stats.gaussian_kde(data[feature])
                x_range = np.linspace(data[feature].min(), data[feature].max(), 100)
                axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2)
                
                axes[i].set_title(f'📊 {feature.replace("_", " ").title()} Distribution',
                                fontweight='bold')
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel('Density')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8)
                axes[i].text(0.7, 0.9, f'μ = {mean_val:.2f}\\nσ = {std_val:.2f}',
                           transform=axes[i].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_model_comparison_plot(self, model_scores, save_path=None):
        """Create model performance comparison plot"""
        plt.figure(figsize=(12, 8))
        
        models = list(model_scores.keys())
        scores = list(model_scores.values())
        
        # Create bar plot
        bars = plt.bar(models, scores, color=self.color_palette[:len(models)], 
                      alpha=0.8, edgecolor='black')
        
        plt.title('🏆 Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model')
        plt.ylabel('Accuracy Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best model
        best_idx = scores.index(max(scores))
        bars[best_idx].set_color('#FFD700')  # Gold color for best model
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plt.gcf()
    
    def create_prediction_confidence_plot(self, predictions, confidences, save_path=None):
        """Create prediction confidence visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color=self.color_palette[2],
                edgecolor='black')
        ax1.set_title('🎯 Prediction Confidence Distribution', fontweight='bold')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Add mean line
        mean_conf = np.mean(confidences)
        ax1.axvline(mean_conf, color='red', linestyle='--', linewidth=2)
        ax1.text(mean_conf + 0.02, ax1.get_ylim()[1] * 0.8, 
                f'Mean: {mean_conf:.3f}', rotation=90)
        
        # Confidence vs Prediction scatter
        scatter = ax2.scatter(predictions, confidences, alpha=0.6, 
                            c=confidences, cmap='viridis', s=50)
        ax2.set_title('🔮 Predictions vs Confidence', fontweight='bold')
        ax2.set_xlabel('Predicted Quality')
        ax2.set_ylabel('Confidence Score')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax2, label='Confidence')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_comprehensive_dashboard(self, data, model_scores=None, 
                                     feature_importance=None, save_path=None):
        """Create comprehensive wine analysis dashboard"""
        fig = plt.figure(figsize=(20, 24))
        
        # Quality distribution
        plt.subplot(4, 3, 1)
        quality_counts = data['quality'].value_counts().sort_index()
        plt.bar(quality_counts.index, quality_counts.values, 
               color=self.color_palette[0], alpha=0.8)
        plt.title('🍷 Quality Distribution', fontweight='bold')
        plt.xlabel('Quality Score')
        plt.ylabel('Count')
        
        # Correlation heatmap (simplified)
        plt.subplot(4, 3, 2)
        key_features = ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid', 'quality']
        corr_subset = data[key_features].corr()
        sns.heatmap(corr_subset, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('🔗 Key Feature Correlations', fontweight='bold')
        
        # Alcohol vs Quality
        plt.subplot(4, 3, 3)
        plt.scatter(data['alcohol'], data['quality'], alpha=0.6, 
                   c=data['quality'], cmap='viridis')
        plt.xlabel('Alcohol (%)')
        plt.ylabel('Quality')
        plt.title('🥃 Alcohol vs Quality', fontweight='bold')
        
        # Feature distributions (top 6)
        features_to_plot = ['alcohol', 'volatile_acidity', 'sulphates', 
                           'citric_acid', 'pH', 'residual_sugar']
        
        for i, feature in enumerate(features_to_plot):
            plt.subplot(4, 3, i + 4)
            plt.hist(data[feature], bins=20, alpha=0.7, 
                    color=self.color_palette[i % len(self.color_palette)])
            plt.title(f'{feature.replace("_", " ").title()}', fontweight='bold')
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel('Frequency')
        
        # Model comparison (if provided)
        if model_scores:
            plt.subplot(4, 3, 10)
            models = list(model_scores.keys())
            scores = list(model_scores.values())
            plt.bar(models, scores, color=self.color_palette[:len(models)])
            plt.title('🏆 Model Performance', fontweight='bold')
            plt.xticks(rotation=45)
            plt.ylabel('Accuracy')
        
        # Feature importance (if provided)
        if feature_importance is not None:
            plt.subplot(4, 3, 11)
            top_features = feature_importance.head(8)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.title('🔍 Top Feature Importance', fontweight='bold')
            plt.xlabel('Importance')
        
        # Quality statistics
        plt.subplot(4, 3, 12)
        quality_stats = data.groupby('quality').size()
        plt.pie(quality_stats.values, labels=quality_stats.index, autopct='%1.1f%%',
               colors=plt.cm.Set3(np.linspace(0, 1, len(quality_stats))))
        plt.title('🎯 Quality Distribution', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def create_interactive_wine_explorer(data):
    """Create interactive wine data explorer using Plotly"""
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Quality Distribution', 'Alcohol vs Quality', 
                       'Feature Correlations', 'Quality by Alcohol Range'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "heatmap"}, {"type": "box"}]]
    )
    
    # Quality distribution
    quality_counts = data['quality'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=quality_counts.index, y=quality_counts.values, 
               name='Quality Distribution', marker_color='darkred'),
        row=1, col=1
    )
    
    # Alcohol vs Quality scatter
    fig.add_trace(
        go.Scatter(x=data['alcohol'], y=data['quality'], 
                  mode='markers', name='Wines',
                  marker=dict(color=data['quality'], colorscale='viridis',
                            showscale=True)),
        row=1, col=2
    )
    
    # Correlation heatmap
    key_features = ['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid']
    corr_matrix = data[key_features + ['quality']].corr()
    
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, 
                  x=corr_matrix.columns, 
                  y=corr_matrix.columns,
                  colorscale='RdBu', zmid=0),
        row=2, col=1
    )
    
    # Quality by alcohol range
    data['alcohol_range'] = pd.cut(data['alcohol'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    for alcohol_range in data['alcohol_range'].unique():
        if pd.notna(alcohol_range):
            range_data = data[data['alcohol_range'] == alcohol_range]
            fig.add_trace(
                go.Box(y=range_data['quality'], name=str(alcohol_range)),
                row=2, col=2
            )
    
    # Update layout
    fig.update_layout(
        title_text="🍷 Interactive Wine Quality Analysis Dashboard",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    return fig

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Visualization Toolkit - Testing")
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 500
    
    sample_data = pd.DataFrame({
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
    
    # Create quality scores
    quality_score = (
        (sample_data['alcohol'] - 8) * 0.3 +
        (12 - sample_data['volatile_acidity'] * 10) * 0.2 +
        (sample_data['sulphates'] * 10) * 0.15 +
        np.random.normal(0, 1, n_samples) * 0.5
    )
    sample_data['quality'] = np.clip(np.round(quality_score + 6).astype(int), 3, 9)
    
    # Initialize visualizer
    visualizer = WineVisualizer()
    
    # Test visualizations
    print("Creating quality distribution plot...")
    visualizer.create_quality_distribution_plot(sample_data)
    
    print("Creating correlation heatmap...")
    visualizer.create_correlation_heatmap(sample_data)
    
    print("Creating alcohol vs quality scatter plot...")
    visualizer.create_alcohol_quality_scatter(sample_data)
    
    print("✅ Wine visualization toolkit testing completed!")