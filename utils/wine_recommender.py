#!/usr/bin/env python3
"""
Wine Recommendation Engine
AI-powered wine recommendations based on preferences and characteristics
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class WineRecommendationEngine:
    """Intelligent wine recommendation system"""
    
    def __init__(self):
        self.wine_database = None
        self.similarity_matrix = None
        self.knn_model = None
        self.wine_profiles = None
        
    def initialize_wine_database(self, wine_data=None):
        """Initialize wine database with sample wines or provided data"""
        if wine_data is not None:
            self.wine_database = wine_data.copy()
        else:
            # Create comprehensive wine database
            self.wine_database = self._create_sample_wine_database()
        
        print(f"🍷 Initialized wine database with {len(self.wine_database)} wines")
        return self.wine_database
    
    def _create_sample_wine_database(self):
        """Create a diverse sample wine database"""
        wines = [
            # Red Wines
            {
                'name': 'Cabernet Sauvignon Reserve',
                'type': 'Red',
                'region': 'Napa Valley',
                'vintage': 2019,
                'price_range': 'Premium',
                'fixed_acidity': 7.8,
                'volatile_acidity': 0.4,
                'citric_acid': 0.3,
                'residual_sugar': 2.1,
                'chlorides': 0.075,
                'free_sulfur_dioxide': 12,
                'total_sulfur_dioxide': 35,
                'density': 0.9965,
                'pH': 3.4,
                'sulphates': 0.8,
                'alcohol': 13.5,
                'quality': 8,
                'style': 'Full-bodied, Rich tannins',
                'food_pairings': ['Red meat', 'Aged cheese', 'Dark chocolate']
            },
            {
                'name': 'Pinot Noir Estate',
                'type': 'Red',
                'region': 'Oregon',
                'vintage': 2020,
                'price_range': 'Mid-range',
                'fixed_acidity': 6.5,
                'volatile_acidity': 0.3,
                'citric_acid': 0.4,
                'residual_sugar': 1.8,
                'chlorides': 0.065,
                'free_sulfur_dioxide': 18,
                'total_sulfur_dioxide': 42,
                'density': 0.9945,
                'pH': 3.6,
                'sulphates': 0.6,
                'alcohol': 12.8,
                'quality': 7,
                'style': 'Light-bodied, Elegant',
                'food_pairings': ['Salmon', 'Mushroom dishes', 'Soft cheese']
            },
            {
                'name': 'Merlot Classic',
                'type': 'Red',
                'region': 'Bordeaux',
                'vintage': 2018,
                'price_range': 'Mid-range',
                'fixed_acidity': 7.2,
                'volatile_acidity': 0.35,
                'citric_acid': 0.25,
                'residual_sugar': 2.3,
                'chlorides': 0.08,
                'free_sulfur_dioxide': 15,
                'total_sulfur_dioxide': 38,
                'density': 0.9955,
                'pH': 3.5,
                'sulphates': 0.7,
                'alcohol': 13.2,
                'quality': 7,
                'style': 'Medium-bodied, Smooth',
                'food_pairings': ['Grilled meats', 'Pasta', 'Medium cheese']
            },
            # White Wines
            {
                'name': 'Chardonnay Reserve',
                'type': 'White',
                'region': 'Burgundy',
                'vintage': 2020,
                'price_range': 'Premium',
                'fixed_acidity': 6.8,
                'volatile_acidity': 0.25,
                'citric_acid': 0.35,
                'residual_sugar': 1.5,
                'chlorides': 0.045,
                'free_sulfur_dioxide': 32,
                'total_sulfur_dioxide': 125,
                'density': 0.9935,
                'pH': 3.2,
                'sulphates': 0.5,
                'alcohol': 13.0,
                'quality': 8,
                'style': 'Full-bodied, Oak-aged',
                'food_pairings': ['Lobster', 'Creamy pasta', 'Aged cheese']
            },
            {
                'name': 'Sauvignon Blanc',
                'type': 'White',
                'region': 'Marlborough',
                'vintage': 2021,
                'price_range': 'Budget',
                'fixed_acidity': 7.5,
                'volatile_acidity': 0.2,
                'citric_acid': 0.4,
                'residual_sugar': 1.2,
                'chlorides': 0.04,
                'free_sulfur_dioxide': 45,
                'total_sulfur_dioxide': 150,
                'density': 0.9925,
                'pH': 3.1,
                'sulphates': 0.45,
                'alcohol': 12.5,
                'quality': 6,
                'style': 'Light-bodied, Crisp',
                'food_pairings': ['Seafood', 'Salads', 'Goat cheese']
            },
            {
                'name': 'Riesling Late Harvest',
                'type': 'White',
                'region': 'Mosel',
                'vintage': 2019,
                'price_range': 'Premium',
                'fixed_acidity': 6.2,
                'volatile_acidity': 0.18,
                'citric_acid': 0.45,
                'residual_sugar': 12.5,
                'chlorides': 0.035,
                'free_sulfur_dioxide': 38,
                'total_sulfur_dioxide': 140,
                'density': 0.9965,
                'pH': 3.0,
                'sulphates': 0.4,
                'alcohol': 9.5,
                'quality': 8,
                'style': 'Sweet, Aromatic',
                'food_pairings': ['Spicy food', 'Fruit desserts', 'Blue cheese']
            },
            # Rosé and Sparkling
            {
                'name': 'Provence Rosé',
                'type': 'Rosé',
                'region': 'Provence',
                'vintage': 2021,
                'price_range': 'Mid-range',
                'fixed_acidity': 6.9,
                'volatile_acidity': 0.22,
                'citric_acid': 0.38,
                'residual_sugar': 2.8,
                'chlorides': 0.05,
                'free_sulfur_dioxide': 28,
                'total_sulfur_dioxide': 110,
                'density': 0.9940,
                'pH': 3.3,
                'sulphates': 0.48,
                'alcohol': 12.0,
                'quality': 7,
                'style': 'Dry, Fresh',
                'food_pairings': ['Mediterranean cuisine', 'Light seafood', 'Summer salads']
            },
            {
                'name': 'Champagne Brut',
                'type': 'Sparkling',
                'region': 'Champagne',
                'vintage': 2018,
                'price_range': 'Luxury',
                'fixed_acidity': 7.0,
                'volatile_acidity': 0.15,
                'citric_acid': 0.42,
                'residual_sugar': 8.5,
                'chlorides': 0.038,
                'free_sulfur_dioxide': 35,
                'total_sulfur_dioxide': 120,
                'density': 0.9930,
                'pH': 3.1,
                'sulphates': 0.42,
                'alcohol': 12.2,
                'quality': 9,
                'style': 'Elegant, Complex',
                'food_pairings': ['Oysters', 'Caviar', 'Celebration foods']
            }
        ]
        
        return pd.DataFrame(wines)
    
    def build_similarity_matrix(self):
        """Build wine similarity matrix based on chemical characteristics"""
        # Select numerical features for similarity calculation
        feature_cols = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        # Normalize features
        feature_matrix = self.wine_database[feature_cols].values
        feature_matrix_normalized = (feature_matrix - feature_matrix.mean(axis=0)) / feature_matrix.std(axis=0)
        
        # Calculate cosine similarity
        self.similarity_matrix = cosine_similarity(feature_matrix_normalized)
        
        print("🔗 Built wine similarity matrix")
        return self.similarity_matrix
    
    def train_knn_model(self, n_neighbors=5):
        """Train KNN model for wine recommendations"""
        feature_cols = [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol'
        ]
        
        X = self.wine_database[feature_cols].values
        
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.knn_model.fit(X)
        
        print(f"🤖 Trained KNN model with {n_neighbors} neighbors")
        return self.knn_model
    
    def recommend_similar_wines(self, wine_index, n_recommendations=3):
        """Recommend wines similar to a given wine"""
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        # Get similarity scores for the wine
        similarity_scores = self.similarity_matrix[wine_index]
        
        # Get indices of most similar wines (excluding the wine itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        recommendations = []
        base_wine = self.wine_database.iloc[wine_index]
        
        print(f"\n🍷 Wines similar to {base_wine['name']}:")
        print("=" * 50)
        
        for i, idx in enumerate(similar_indices, 1):
            wine = self.wine_database.iloc[idx]
            similarity = similarity_scores[idx]
            
            recommendation = {
                'rank': i,
                'name': wine['name'],
                'type': wine['type'],
                'region': wine['region'],
                'vintage': wine['vintage'],
                'quality': wine['quality'],
                'similarity_score': similarity,
                'style': wine['style'],
                'price_range': wine['price_range']
            }
            
            recommendations.append(recommendation)
            
            print(f"  {i}. {wine['name']} ({wine['region']}, {wine['vintage']})")
            print(f"     Similarity: {similarity:.3f} | Quality: {wine['quality']}/9")
            print(f"     Style: {wine['style']}")
            print(f"     Price: {wine['price_range']}")
            print()
        
        return recommendations
    
    def recommend_by_preferences(self, preferences, n_recommendations=5):
        """Recommend wines based on user preferences"""
        print(f"\n🎯 Wine recommendations based on your preferences:")
        print("=" * 50)
        
        # Filter wines based on preferences
        filtered_wines = self.wine_database.copy()
        
        # Apply filters
        if 'wine_type' in preferences and preferences['wine_type']:
            filtered_wines = filtered_wines[filtered_wines['type'].isin(preferences['wine_type'])]
        
        if 'price_range' in preferences and preferences['price_range']:
            filtered_wines = filtered_wines[filtered_wines['price_range'].isin(preferences['price_range'])]
        
        if 'min_quality' in preferences:
            filtered_wines = filtered_wines[filtered_wines['quality'] >= preferences['min_quality']]
        
        if 'max_alcohol' in preferences:
            filtered_wines = filtered_wines[filtered_wines['alcohol'] <= preferences['max_alcohol']]
        
        if 'min_alcohol' in preferences:
            filtered_wines = filtered_wines[filtered_wines['alcohol'] >= preferences['min_alcohol']]
        
        # Score wines based on preferences
        filtered_wines = filtered_wines.copy()
        filtered_wines['preference_score'] = 0
        
        # Quality preference (higher is better)
        filtered_wines['preference_score'] += filtered_wines['quality'] * 0.3
        
        # Alcohol preference
        if 'preferred_alcohol' in preferences:
            alcohol_diff = abs(filtered_wines['alcohol'] - preferences['preferred_alcohol'])
            filtered_wines['preference_score'] += (5 - alcohol_diff) * 0.2
        
        # Sweetness preference
        if 'sweetness_preference' in preferences:
            if preferences['sweetness_preference'] == 'dry':
                filtered_wines['preference_score'] += (5 - filtered_wines['residual_sugar']) * 0.15
            elif preferences['sweetness_preference'] == 'sweet':
                filtered_wines['preference_score'] += filtered_wines['residual_sugar'] * 0.15
        
        # Sort by preference score
        top_wines = filtered_wines.nlargest(n_recommendations, 'preference_score')
        
        recommendations = []
        for i, (_, wine) in enumerate(top_wines.iterrows(), 1):
            recommendation = {
                'rank': i,
                'name': wine['name'],
                'type': wine['type'],
                'region': wine['region'],
                'vintage': wine['vintage'],
                'quality': wine['quality'],
                'preference_score': wine['preference_score'],
                'style': wine['style'],
                'price_range': wine['price_range'],
                'food_pairings': wine['food_pairings']
            }
            
            recommendations.append(recommendation)
            
            print(f"  {i}. {wine['name']} ({wine['region']}, {wine['vintage']})")
            print(f"     Quality: {wine['quality']}/9 | Alcohol: {wine['alcohol']}%")
            print(f"     Style: {wine['style']}")
            print(f"     Price: {wine['price_range']}")
            print(f"     Food Pairings: {', '.join(wine['food_pairings'][:3])}")
            print()
        
        return recommendations
    
    def recommend_by_food_pairing(self, food_type, n_recommendations=3):
        """Recommend wines based on food pairing"""
        print(f"\n🍽️ Wine recommendations for {food_type}:")
        print("=" * 40)
        
        # Find wines that pair well with the food
        matching_wines = []
        
        for _, wine in self.wine_database.iterrows():
            food_pairings = [pairing.lower() for pairing in wine['food_pairings']]
            if any(food_type.lower() in pairing for pairing in food_pairings):
                matching_wines.append(wine)
        
        if not matching_wines:
            # Fallback: recommend based on general food categories
            food_wine_mapping = {
                'red meat': ['Red'],
                'seafood': ['White', 'Rosé'],
                'cheese': ['Red', 'White'],
                'dessert': ['White'],
                'spicy': ['White', 'Rosé'],
                'pasta': ['Red', 'White']
            }
            
            for category, wine_types in food_wine_mapping.items():
                if category in food_type.lower():
                    matching_wines = self.wine_database[
                        self.wine_database['type'].isin(wine_types)
                    ].nlargest(n_recommendations, 'quality').to_dict('records')
                    break
        
        if not matching_wines:
            matching_wines = self.wine_database.nlargest(n_recommendations, 'quality').to_dict('records')
        
        recommendations = []
        for i, wine in enumerate(matching_wines[:n_recommendations], 1):
            recommendation = {
                'rank': i,
                'name': wine['name'],
                'type': wine['type'],
                'region': wine['region'],
                'quality': wine['quality'],
                'style': wine['style'],
                'why_recommended': f"Pairs excellently with {food_type}"
            }
            
            recommendations.append(recommendation)
            
            print(f"  {i}. {wine['name']} ({wine['type']})")
            print(f"     Quality: {wine['quality']}/9")
            print(f"     Why: {recommendation['why_recommended']}")
            print()
        
        return recommendations
    
    def recommend_by_occasion(self, occasion, n_recommendations=3):
        """Recommend wines based on occasion"""
        occasion_mapping = {
            'celebration': {'types': ['Sparkling'], 'min_quality': 7},
            'romantic dinner': {'types': ['Red'], 'min_quality': 7, 'style_keywords': ['elegant', 'smooth']},
            'casual dinner': {'types': ['Red', 'White'], 'price_range': ['Budget', 'Mid-range']},
            'business meeting': {'types': ['White', 'Rosé'], 'min_quality': 6},
            'summer party': {'types': ['White', 'Rosé'], 'max_alcohol': 13},
            'winter evening': {'types': ['Red'], 'min_alcohol': 12},
            'gift': {'min_quality': 8, 'price_range': ['Premium', 'Luxury']}
        }
        
        print(f"\n🎉 Wine recommendations for {occasion}:")
        print("=" * 40)
        
        criteria = occasion_mapping.get(occasion.lower(), {})
        
        filtered_wines = self.wine_database.copy()
        
        if 'types' in criteria:
            filtered_wines = filtered_wines[filtered_wines['type'].isin(criteria['types'])]
        
        if 'min_quality' in criteria:
            filtered_wines = filtered_wines[filtered_wines['quality'] >= criteria['min_quality']]
        
        if 'price_range' in criteria:
            filtered_wines = filtered_wines[filtered_wines['price_range'].isin(criteria['price_range'])]
        
        if 'max_alcohol' in criteria:
            filtered_wines = filtered_wines[filtered_wines['alcohol'] <= criteria['max_alcohol']]
        
        if 'min_alcohol' in criteria:
            filtered_wines = filtered_wines[filtered_wines['alcohol'] >= criteria['min_alcohol']]
        
        # Sort by quality and select top recommendations
        top_wines = filtered_wines.nlargest(n_recommendations, 'quality')
        
        recommendations = []
        for i, (_, wine) in enumerate(top_wines.iterrows(), 1):
            recommendation = {
                'rank': i,
                'name': wine['name'],
                'type': wine['type'],
                'region': wine['region'],
                'quality': wine['quality'],
                'style': wine['style'],
                'price_range': wine['price_range'],
                'occasion_fit': f"Perfect for {occasion}"
            }
            
            recommendations.append(recommendation)
            
            print(f"  {i}. {wine['name']} ({wine['type']})")
            print(f"     Quality: {wine['quality']}/9 | Price: {wine['price_range']}")
            print(f"     Why: {recommendation['occasion_fit']}")
            print()
        
        return recommendations
    
    def get_wine_profile(self, wine_name):
        """Get detailed profile of a specific wine"""
        wine = self.wine_database[self.wine_database['name'].str.contains(wine_name, case=False)]
        
        if wine.empty:
            print(f"❌ Wine '{wine_name}' not found in database")
            return None
        
        wine_data = wine.iloc[0]
        
        print(f"\n🍷 WINE PROFILE: {wine_data['name']}")
        print("=" * 50)
        print(f"Type: {wine_data['type']}")
        print(f"Region: {wine_data['region']}")
        print(f"Vintage: {wine_data['vintage']}")
        print(f"Quality Score: {wine_data['quality']}/9")
        print(f"Alcohol: {wine_data['alcohol']}%")
        print(f"Style: {wine_data['style']}")
        print(f"Price Range: {wine_data['price_range']}")
        print(f"Food Pairings: {', '.join(wine_data['food_pairings'])}")
        
        # Chemical profile
        print(f"\n🧪 Chemical Profile:")
        print(f"  Acidity: {wine_data['fixed_acidity']:.1f} (Fixed), {wine_data['volatile_acidity']:.2f} (Volatile)")
        print(f"  Residual Sugar: {wine_data['residual_sugar']:.1f} g/L")
        print(f"  pH: {wine_data['pH']:.1f}")
        print(f"  Sulphates: {wine_data['sulphates']:.2f}")
        
        return wine_data.to_dict()

def create_user_preference_profile():
    """Interactive function to create user preference profile"""
    print("🍷 Let's create your wine preference profile!")
    print("=" * 45)
    
    preferences = {}
    
    # Wine type preference
    print("1. What types of wine do you prefer? (Enter numbers separated by commas)")
    print("   1) Red  2) White  3) Rosé  4) Sparkling")
    type_choice = input("Your choice: ").strip()
    
    type_mapping = {'1': 'Red', '2': 'White', '3': 'Rosé', '4': 'Sparkling'}
    preferences['wine_type'] = [type_mapping.get(choice.strip()) for choice in type_choice.split(',') if choice.strip() in type_mapping]
    
    # Price range
    print("\n2. What's your preferred price range?")
    print("   1) Budget  2) Mid-range  3) Premium  4) Luxury")
    price_choice = input("Your choice: ").strip()
    
    price_mapping = {'1': 'Budget', '2': 'Mid-range', '3': 'Premium', '4': 'Luxury'}
    preferences['price_range'] = [price_mapping.get(price_choice)]
    
    # Quality preference
    print("\n3. Minimum quality score (1-9)?")
    try:
        preferences['min_quality'] = int(input("Minimum quality: ").strip())
    except ValueError:
        preferences['min_quality'] = 6
    
    # Alcohol preference
    print("\n4. Preferred alcohol content (%)?")
    try:
        preferences['preferred_alcohol'] = float(input("Preferred alcohol %: ").strip())
    except ValueError:
        preferences['preferred_alcohol'] = 12.0
    
    # Sweetness preference
    print("\n5. Sweetness preference?")
    print("   1) Dry  2) Off-dry  3) Sweet")
    sweet_choice = input("Your choice: ").strip()
    
    sweet_mapping = {'1': 'dry', '2': 'off-dry', '3': 'sweet'}
    preferences['sweetness_preference'] = sweet_mapping.get(sweet_choice, 'dry')
    
    print("\n✅ Preference profile created!")
    return preferences

if __name__ == "__main__":
    # Example usage
    print("🍷 Wine Recommendation Engine - Testing")
    
    # Initialize recommendation engine
    recommender = WineRecommendationEngine()
    recommender.initialize_wine_database()
    recommender.build_similarity_matrix()
    recommender.train_knn_model()
    
    # Test recommendations
    print("\n" + "="*60)
    
    # Similar wine recommendations
    recommender.recommend_similar_wines(0, 3)
    
    # Preference-based recommendations
    sample_preferences = {
        'wine_type': ['Red'],
        'price_range': ['Mid-range', 'Premium'],
        'min_quality': 7,
        'preferred_alcohol': 13.0,
        'sweetness_preference': 'dry'
    }
    
    recommender.recommend_by_preferences(sample_preferences, 3)
    
    # Food pairing recommendations
    recommender.recommend_by_food_pairing('red meat', 3)
    
    # Occasion-based recommendations
    recommender.recommend_by_occasion('celebration', 3)
    
    print("\n✅ Wine recommendation engine testing completed!")