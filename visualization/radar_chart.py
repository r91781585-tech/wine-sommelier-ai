#!/usr/bin/env python3
"""
Wine Radar Chart Visualization
Specialized radar charts for wine characteristic profiling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import plotly.graph_objects as go
import plotly.express as px

class WineRadarChart:
    """Specialized radar chart creator for wine profiles"""
    
    def __init__(self):
        self.wine_characteristics = [
            'Alcohol', 'Acidity', 'Sweetness', 'Tannins', 
            'Body', 'Complexity', 'Balance', 'Finish'
        ]
        
    def normalize_wine_features(self, wine_features, feature_names):
        """Normalize wine features to 0-10 scale for radar chart"""
        normalized = {}
        
        for i, feature in enumerate(feature_names):
            value = wine_features[i]
            
            if feature == 'alcohol':
                # Normalize alcohol (8-15% -> 0-10)
                normalized['Alcohol'] = min(max((value - 8) / 7 * 10, 0), 10)
                
            elif feature == 'volatile_acidity':
                # Normalize acidity (0-1.6 -> 10-0, inverted because lower is better)
                normalized['Acidity'] = max(10 - (value / 1.6 * 10), 0)
                
            elif feature == 'residual_sugar':
                # Normalize sweetness (0-15 -> 0-10)
                normalized['Sweetness'] = min(value / 15 * 10, 10)
                
            elif feature == 'sulphates':
                # Normalize tannins proxy (0.3-2.0 -> 0-10)
                normalized['Tannins'] = min(max((value - 0.3) / 1.7 * 10, 0), 10)
                
            elif feature == 'density':
                # Normalize body (0.99-1.01 -> 0-10)
                normalized['Body'] = min(max((value - 0.99) / 0.02 * 10, 0), 10)
                
            elif feature == 'citric_acid':
                # Normalize complexity (0-1 -> 0-10)
                normalized['Complexity'] = min(value / 1.0 * 10, 10)
                
            elif feature == 'pH':
                # Normalize balance (2.5-4.0 -> 10-0, optimal around 3.3)
                optimal_ph = 3.3
                deviation = abs(value - optimal_ph)
                normalized['Balance'] = max(10 - (deviation / 0.8 * 10), 0)
                
            elif feature == 'total_sulfur_dioxide':
                # Normalize finish (lower sulfur = longer finish)
                normalized['Finish'] = max(10 - (value / 300 * 10), 0)
        
        # Fill missing characteristics with average values
        for char in self.wine_characteristics:
            if char not in normalized:
                normalized[char] = 5.0  # Default middle value
        
        return normalized
    
    def create_matplotlib_radar(self, wine_features, feature_names, wine_name="Wine Sample", 
                               save_path=None):
        """Create radar chart using matplotlib"""
        # Normalize features
        normalized = self.normalize_wine_features(wine_features, feature_names)
        
        # Extract values in order
        values = [normalized[char] for char in self.wine_characteristics]
        
        # Number of characteristics
        N = len(self.wine_characteristics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value at the end to close the radar chart
        values += values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot the radar chart
        ax.plot(angles, values, 'o-', linewidth=2, label=wine_name, color='#722F37')
        ax.fill(angles, values, alpha=0.25, color='#722F37')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.wine_characteristics, fontsize=12)
        
        # Set y-axis limits
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontsize=10)
        ax.grid(True)
        
        # Add title
        plt.title(f'🍷 Wine Profile: {wine_name}', size=16, fontweight='bold', pad=20)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Add value annotations
        for angle, value, char in zip(angles[:-1], values[:-1], self.wine_characteristics):
            ax.annotate(f'{value:.1f}', 
                       xy=(angle, value), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=9, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def create_plotly_radar(self, wine_features, feature_names, wine_name="Wine Sample"):
        """Create interactive radar chart using Plotly"""
        # Normalize features
        normalized = self.normalize_wine_features(wine_features, feature_names)
        
        # Extract values in order
        values = [normalized[char] for char in self.wine_characteristics]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=self.wine_characteristics,
            fill='toself',
            name=wine_name,
            line_color='rgb(114, 47, 55)',
            fillcolor='rgba(114, 47, 55, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )),
            showlegend=True,
            title=f"🍷 Wine Profile: {wine_name}",
            title_x=0.5,
            font=dict(size=12),
            width=600,
            height=600
        )
        
        return fig
    
    def compare_wines_radar(self, wines_data, wine_names, save_path=None):
        """Create comparative radar chart for multiple wines"""
        colors = ['#722F37', '#C73E1D', '#F39C12', '#27AE60', '#3498DB']
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Number of characteristics
        N = len(self.wine_characteristics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        for i, (wine_features, wine_name) in enumerate(zip(wines_data, wine_names)):
            # Normalize features (assuming same feature order for all wines)
            feature_names = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                'density', 'pH', 'sulphates', 'alcohol'
            ]
            
            normalized = self.normalize_wine_features(wine_features, feature_names)
            values = [normalized[char] for char in self.wine_characteristics]
            values += values[:1]  # Close the circle
            
            color = colors[i % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=wine_name, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.wine_characteristics, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontsize=10)
        ax.grid(True)
        
        plt.title('🍷 Wine Profile Comparison', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def create_wine_style_radar(self, wine_style, save_path=None):
        """Create radar chart for predefined wine styles"""
        wine_styles = {
            'Cabernet Sauvignon': {
                'Alcohol': 8.5, 'Acidity': 6.0, 'Sweetness': 1.0, 'Tannins': 9.0,
                'Body': 9.0, 'Complexity': 8.0, 'Balance': 7.5, 'Finish': 8.5
            },
            'Pinot Noir': {
                'Alcohol': 6.5, 'Acidity': 7.5, 'Sweetness': 1.5, 'Tannins': 4.0,
                'Body': 4.5, 'Complexity': 7.0, 'Balance': 8.0, 'Finish': 7.0
            },
            'Chardonnay': {
                'Alcohol': 7.0, 'Acidity': 6.5, 'Sweetness': 2.0, 'Tannins': 1.0,
                'Body': 7.0, 'Complexity': 6.5, 'Balance': 7.5, 'Finish': 6.5
            },
            'Sauvignon Blanc': {
                'Alcohol': 6.0, 'Acidity': 8.5, 'Sweetness': 1.5, 'Tannins': 0.5,
                'Body': 3.5, 'Complexity': 5.0, 'Balance': 7.0, 'Finish': 5.5
            },
            'Riesling': {
                'Alcohol': 4.5, 'Acidity': 8.0, 'Sweetness': 7.0, 'Tannins': 0.5,
                'Body': 3.0, 'Complexity': 6.5, 'Balance': 6.5, 'Finish': 6.0
            }
        }
        
        if wine_style not in wine_styles:
            print(f"Wine style '{wine_style}' not found. Available styles: {list(wine_styles.keys())}")
            return None
        
        values = [wine_styles[wine_style][char] for char in self.wine_characteristics]
        
        # Create the radar chart
        N = len(self.wine_characteristics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, values, 'o-', linewidth=3, label=wine_style, color='#722F37')
        ax.fill(angles, values, alpha=0.3, color='#722F37')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.wine_characteristics, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11, 2))
        ax.set_yticklabels(range(0, 11, 2), fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.title(f'🍷 {wine_style} Style Profile', size=16, fontweight='bold', pad=20)
        
        # Add characteristic descriptions
        descriptions = {
            'Alcohol': 'Alcohol Content',
            'Acidity': 'Acidity Level',
            'Sweetness': 'Residual Sugar',
            'Tannins': 'Tannin Structure',
            'Body': 'Wine Body',
            'Complexity': 'Flavor Complexity',
            'Balance': 'Overall Balance',
            'Finish': 'Finish Length'
        }
        
        # Add value annotations with descriptions
        for angle, value, char in zip(angles[:-1], values[:-1], self.wine_characteristics):
            ax.annotate(f'{value:.1f}', 
                       xy=(angle, value), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       fontsize=10, 
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def create_interactive_comparison(self, wines_data, wine_names):
        """Create interactive comparison radar chart using Plotly"""
        colors = ['rgb(114, 47, 55)', 'rgb(199, 62, 29)', 'rgb(243, 156, 18)', 
                 'rgb(39, 174, 96)', 'rgb(52, 152, 219)']
        
        fig = go.Figure()
        
        feature_names = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        for i, (wine_features, wine_name) in enumerate(zip(wines_data, wine_names)):
            normalized = self.normalize_wine_features(wine_features, feature_names)
            values = [normalized[char] for char in self.wine_characteristics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=self.wine_characteristics,
                fill='toself',
                name=wine_name,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.1)')
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickmode='linear',
                    tick0=0,
                    dtick=2
                )),
            showlegend=True,
            title="🍷 Interactive Wine Profile Comparison",
            title_x=0.5,
            font=dict(size=12),
            width=800,
            height=800
        )
        
        return fig
    
    def generate_wine_profile_report(self, wine_features, feature_names, wine_name="Wine Sample"):
        """Generate detailed wine profile report with radar chart"""
        normalized = self.normalize_wine_features(wine_features, feature_names)
        
        print(f"\n🍷 WINE PROFILE REPORT: {wine_name}")
        print("=" * 50)
        
        # Overall score
        overall_score = sum(normalized.values()) / len(normalized)
        print(f"Overall Profile Score: {overall_score:.1f}/10")
        
        # Characteristic breakdown
        print(f"\n📊 Characteristic Breakdown:")
        for char in self.wine_characteristics:
            value = normalized[char]
            bar = "█" * int(value) + "░" * (10 - int(value))
            
            # Interpretation
            if value >= 8:
                level = "Excellent"
            elif value >= 6:
                level = "Good"
            elif value >= 4:
                level = "Average"
            else:
                level = "Below Average"
            
            print(f"  {char:<12}: {bar} {value:.1f}/10 ({level})")
        
        # Style classification
        print(f"\n🎯 Style Classification:")
        
        if normalized['Alcohol'] > 7 and normalized['Body'] > 7:
            style = "Full-bodied"
        elif normalized['Alcohol'] > 5 and normalized['Body'] > 5:
            style = "Medium-bodied"
        else:
            style = "Light-bodied"
        
        if normalized['Sweetness'] > 6:
            sweetness = "Sweet"
        elif normalized['Sweetness'] > 3:
            sweetness = "Off-dry"
        else:
            sweetness = "Dry"
        
        if normalized['Tannins'] > 6:
            structure = "Structured"
        elif normalized['Tannins'] > 3:
            structure = "Medium structure"
        else:
            structure = "Soft"
        
        print(f"  Style: {style}, {sweetness}, {structure}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if normalized['Balance'] < 5:
            print("  • Consider decanting to improve balance")
        
        if normalized['Complexity'] > 7:
            print("  • Excellent for special occasions")
        
        if normalized['Finish'] > 7:
            print("  • Great for slow sipping and contemplation")
        
        if normalized['Acidity'] > 7:
            print("  • Pairs well with rich, fatty foods")
        
        return normalized

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Radar Chart - Testing")
    
    # Sample wine features
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
    
    feature_names = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]
    
    # Initialize radar chart creator
    radar = WineRadarChart()
    
    # Create radar chart
    print("Creating wine radar chart...")
    radar.create_matplotlib_radar(sample_wine, feature_names, "Sample Red Wine")
    
    # Generate profile report
    radar.generate_wine_profile_report(sample_wine, feature_names, "Sample Red Wine")
    
    # Create style radar
    print("Creating Cabernet Sauvignon style radar...")
    radar.create_wine_style_radar("Cabernet Sauvignon")
    
    print("✅ Wine radar chart testing completed!")