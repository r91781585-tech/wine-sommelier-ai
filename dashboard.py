#!/usr/bin/env python3
"""
Wine Sommelier AI - Interactive Streamlit Dashboard
Real-time wine quality prediction and analysis interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wine_analyzer import WineSommelierAI

# Page configuration
st.set_page_config(
    page_title="🍷 Wine Sommelier AI",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #722F37;
    }
    .wine-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_wine_data():
    """Load and cache wine data"""
    analyzer = WineSommelierAI()
    data = analyzer.load_data()
    return analyzer, data

def create_radar_chart(wine_features, feature_names):
    """Create a radar chart for wine characteristics"""
    # Normalize features to 0-10 scale for better visualization
    normalized_features = []
    for i, feature in enumerate(wine_features):
        if feature_names[i] == 'alcohol':
            normalized_features.append(min(feature, 15) / 15 * 10)
        elif feature_names[i] == 'pH':
            normalized_features.append((feature - 2.5) / 1.5 * 10)
        elif feature_names[i] == 'density':
            normalized_features.append((feature - 0.99) / 0.01 * 10)
        else:
            # For other features, use percentile-based normalization
            normalized_features.append(min(feature * 2, 10))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=feature_names,
        fill='toself',
        name='Wine Profile',
        line_color='rgb(114, 47, 55)',
        fillcolor='rgba(114, 47, 55, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="🍷 Wine Characteristics Radar Chart",
        font=dict(size=12)
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Sommelier AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### *AI-Powered Wine Quality Analysis & Prediction*")
    
    # Load data
    with st.spinner("Loading wine data..."):
        analyzer, data = load_wine_data()
        analyzer.prepare_data()
        analyzer.train_model()
    
    # Sidebar for wine input
    st.sidebar.header("🍇 Wine Characteristics Input")
    st.sidebar.markdown("Adjust the sliders to input wine parameters:")
    
    # Feature input sliders
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.3, 0.1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3, 0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.9, 15.0, 2.5, 0.1)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.08, 0.001)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1.0, 70.0, 15.0, 1.0)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6.0, 290.0, 46.0, 1.0)
    density = st.sidebar.slider("Density", 0.99, 1.01, 0.996, 0.001)
    pH = st.sidebar.slider("pH", 2.7, 4.0, 3.3, 0.01)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.65, 0.01)
    alcohol = st.sidebar.slider("Alcohol %", 8.0, 15.0, 10.4, 0.1)
    
    wine_features = [
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Wine prediction
        if st.button("🔮 Predict Wine Quality", type="primary"):
            with st.spinner("Analyzing wine characteristics..."):
                quality_pred, quality_proba = analyzer.predict_wine_quality(wine_features)
                
                st.markdown(f"""
                <div class="wine-prediction">
                    <h2>🍷 Wine Quality Prediction</h2>
                    <h1>{quality_pred}/9</h1>
                    <p>Confidence: {max(quality_proba):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate detailed report
                st.subheader("📋 Detailed Analysis Report")
                quality_pred, style, sweetness, acidity = analyzer.generate_wine_report(wine_features)
                
                # Display wine characteristics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Style", style)
                with col_b:
                    st.metric("Sweetness", sweetness)
                with col_c:
                    st.metric("Acidity", acidity)
    
    with col2:
        # Radar chart
        st.subheader("🎯 Wine Profile")
        feature_names = [
            'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
            'Chlorides', 'Free SO2', 'Total SO2', 'Density', 'pH', 'Sulphates', 'Alcohol'
        ]
        radar_fig = create_radar_chart(wine_features, feature_names)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Dataset overview
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Avg Quality", f"{data['quality'].mean():.1f}")
    with col4:
        st.metric("Quality Range", f"{data['quality'].min()}-{data['quality'].max()}")
    
    # Visualizations
    st.header("📈 Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Quality Distribution", "Feature Correlations", "Feature Importance", "Quality vs Features"])
    
    with tab1:
        # Quality distribution
        fig_quality = px.histogram(
            data, x='quality', 
            title="Wine Quality Distribution",
            color_discrete_sequence=['#722F37']
        )
        fig_quality.update_layout(
            xaxis_title="Quality Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with tab2:
        # Correlation heatmap
        corr_matrix = data.corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        # Feature importance (if model is trained)
        if hasattr(analyzer, 'model') and analyzer.model is not None:
            feature_importance = pd.DataFrame({
                'feature': analyzer.feature_names,
                'importance': analyzer.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance for Wine Quality Prediction",
                color='importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab4:
        # Quality vs selected features
        selected_feature = st.selectbox(
            "Select feature to analyze:",
            options=['alcohol', 'volatile_acidity', 'sulphates', 'citric_acid', 'fixed_acidity']
        )
        
        fig_scatter = px.box(
            data, 
            x='quality', 
            y=selected_feature,
            title=f"Quality vs {selected_feature.replace('_', ' ').title()}",
            color='quality',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Wine recommendations
    st.header("🍾 Wine Recommendations")
    
    if st.button("Get Personalized Wine Recommendations"):
        st.subheader("Based on your wine profile:")
        
        recommendations = []
        
        if alcohol > 12:
            recommendations.append("🍷 **Cabernet Sauvignon** - Full-bodied red with rich tannins")
            recommendations.append("🍷 **Chardonnay** - Full-bodied white with oak aging")
        elif alcohol > 10:
            recommendations.append("🍷 **Merlot** - Medium-bodied red with smooth finish")
            recommendations.append("🍷 **Sauvignon Blanc** - Crisp white with citrus notes")
        else:
            recommendations.append("🍷 **Pinot Noir** - Light-bodied red with delicate flavors")
            recommendations.append("🍷 **Riesling** - Light white with floral aromatics")
        
        if residual_sugar > 5:
            recommendations.append("🍷 **Moscato** - Sweet dessert wine")
            recommendations.append("🍷 **Port** - Fortified sweet wine")
        
        for rec in recommendations[:4]:  # Show top 4 recommendations
            st.markdown(rec)
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with ❤️ using Streamlit and Machine Learning*")

if __name__ == "__main__":
    main()