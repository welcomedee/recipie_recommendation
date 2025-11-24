from pydantic import BaseModel
from typing import List, Optional

class IngredientInput(BaseModel):
    ingredients: str
    preferences: Optional[str] = None

class RecipeRecommendation(BaseModel):
    name: str
    ingredients: str
    instructions: str
    cuisine: str
    cooking_time: int
    image_url: str
    match_score: float
    estimated_servings: float

class RecommendationResponse(BaseModel):
    recommendations: List[RecipeRecommendation]
    fallback_suggestion: Optional[str] = None
