import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class Recommender:
    def __init__(self, data_path="Cleaned_Indian_Food_Dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self._load_data()
        self._train_model()

    def _load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            # Fill NaN values
            self.df['Cleaned-Ingredients'] = self.df['Cleaned-Ingredients'].fillna('')
            self.df['TranslatedInstructions'] = self.df['TranslatedInstructions'].fillna('')
            self.df['image-url'] = self.df['image-url'].fillna('')
        else:
            print(f"Warning: Dataset not found at {self.data_path}")
            self.df = pd.DataFrame(columns=['TranslatedRecipeName', 'Cleaned-Ingredients', 'TranslatedInstructions', 'Cuisine', 'TotalTimeInMins', 'image-url', 'Ingredient-count'])

    def _train_model(self):
        if not self.df.empty:
            self.tfidf = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['Cleaned-Ingredients'])
        else:
            self.tfidf = None
            self.tfidf_matrix = None

    def estimate_servings(self, ingredient_count, total_time):
        """
        Heuristic for servings estimation based on ingredient count and cooking time.
        More ingredients and longer time often imply a larger/richer dish.
        """
        # Base serving
        servings = 2 
        
        # Adjust based on ingredient count (e.g., every 4 extra ingredients adds a serving)
        if ingredient_count > 5:
            servings += (ingredient_count - 5) // 4
            
        # Adjust based on time (e.g., > 60 mins might be a larger batch)
        if total_time > 60:
            servings += 1
            
        return min(servings, 8) # Cap at 8 servings

    def recommend(self, user_ingredients, top_k=5):
        if self.df.empty or self.tfidf is None:
            return []

        # Transform user input
        user_tfidf = self.tfidf.transform([user_ingredients])

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()

        # Get top k indices
        top_indices = cosine_sim.argsort()[-top_k:][::-1]

        recommendations = []
        for idx in top_indices:
            score = cosine_sim[idx]
            if score < 0.1: # Threshold for relevance
                continue
                
            row = self.df.iloc[idx]
            
            # Estimate servings
            ing_count = row.get('Ingredient-count', 0)
            time_mins = row.get('TotalTimeInMins', 30)
            servings = self.estimate_servings(ing_count, time_mins)

            rec = {
                "name": row['TranslatedRecipeName'],
                "ingredients": row['Cleaned-Ingredients'],
                "instructions": row['TranslatedInstructions'],
                "cuisine": row['Cuisine'],
                "cooking_time": int(time_mins),
                "image_url": row['image-url'],
                "match_score": float(score),
                "estimated_servings": float(servings)
            }
            recommendations.append(rec)

        return recommendations
