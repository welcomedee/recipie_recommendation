from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from models import IngredientInput, RecommendationResponse, RecipeRecommendation
from gemini_client import GeminiClient
from recommender import Recommender

app = FastAPI(title="Indian Recipe Recommender")

# Initialize components
gemini_client = GeminiClient()
recommender = Recommender()

@app.get("/")
def read_root():
    return {"message": "Welcome to Indian Recipe Recommender API"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(input_data: IngredientInput):
    user_ingredients = input_data.ingredients
    preferences = input_data.preferences

    # 1. Preprocessing with Gemini
    normalized_ingredients = gemini_client.normalize_ingredients(user_ingredients)
    print(f"Normalized Ingredients: {normalized_ingredients}")

    # 2. Get Recommendations
    recommendations_data = recommender.recommend(normalized_ingredients)
    
    recommendations = [RecipeRecommendation(**rec) for rec in recommendations_data]

    # 3. Fallback if no recommendations
    fallback_text = None
    if not recommendations:
        print("No matches found. Generating fallback...")
        fallback_text = gemini_client.generate_fallback_recipe(user_ingredients, preferences or "")

    return RecommendationResponse(
        recommendations=recommendations,
        fallback_suggestion=fallback_text
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
