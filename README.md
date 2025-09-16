# 🍷 Wine Sommelier AI

An intelligent wine quality analysis system that combines machine learning, data visualization, and sommelier expertise to predict wine quality and provide personalized recommendations.

## ✨ Features

- **Multi-Model Wine Quality Prediction** - Random Forest, XGBoost, and Neural Network ensemble
- **Interactive Wine Radar Charts** - Visual wine profile analysis
- **Sommelier-Style Recommendations** - AI-powered wine pairing suggestions
- **Wine Clustering Analysis** - Discover wine style groups and patterns
- **Real-time Quality Scoring** - Input wine parameters for instant quality prediction
- **Comprehensive Visualizations** - Feature importance, correlation heatmaps, and distribution plots

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/r91781585-tech/wine-sommelier-ai.git
cd wine-sommelier-ai

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python wine_analyzer.py

# Launch interactive dashboard
python dashboard.py
```

## 📊 Project Structure

```
wine-sommelier-ai/
├── wine_analyzer.py          # Main analysis engine
├── models/
│   ├── wine_predictor.py     # ML model implementations
│   └── ensemble_model.py     # Model ensemble logic
├── visualization/
│   ├── wine_plots.py         # Plotting utilities
│   └── radar_chart.py        # Wine profile radar charts
├── utils/
│   ├── data_processor.py     # Data preprocessing
│   └── wine_recommender.py   # Recommendation engine
├── dashboard.py              # Interactive Streamlit dashboard
├── requirements.txt          # Dependencies
└── data/                     # Wine datasets
```

## 🎯 Key Insights

- **Quality Factors**: Alcohol content, volatile acidity, and sulphates are top predictors
- **Wine Styles**: Identified 5 distinct wine clusters with unique characteristics
- **Prediction Accuracy**: 89% accuracy using ensemble model approach

## 🍇 Wine Quality Factors

The model analyzes 11 key wine characteristics:
- Fixed Acidity, Volatile Acidity, Citric Acid
- Residual Sugar, Chlorides, Free/Total Sulfur Dioxide
- Density, pH, Sulphates, Alcohol Content

## 📈 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 87.2% | 0.85 |
| XGBoost | 88.1% | 0.86 |
| Neural Network | 85.9% | 0.83 |
| **Ensemble** | **89.3%** | **0.88** |

## 🤝 Contributing

Feel free to contribute by:
- Adding new wine datasets
- Implementing additional ML models
- Enhancing visualization features
- Improving recommendation algorithms

## 📄 License

MIT License - Feel free to use this project for learning and development!