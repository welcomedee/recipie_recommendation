import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class GeminiClient:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            print(f"Error initializing model: {e}")


    def normalize_ingredients(self, input_text: str) -> str:
        """
        Uses Gemini to extract and normalize ingredients from user input.
        """
        prompt = f"""
        Extract the list of ingredients from the following text and normalize them into a comma-separated list of standard ingredient names. 
        Ignore quantities and cooking actions.
        If the input is just a list of ingredients, simply normalize them.
        
        Input: {input_text}
        
        Output (comma-separated list):
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini for normalization: {e}")
            return input_text # Fallback to raw input

    def generate_fallback_recipe(self, ingredients: str, preferences: str = "") -> str:
        """
        Generates a recipe suggestion using Gemini when no good matches are found.
        """
        prompt = f"""
        Create a unique Indian-fusion recipe using the following ingredients: {ingredients}.
        Preferences: {preferences}
        
        Provide the recipe name, ingredients list, and brief instructions.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error calling Gemini for fallback: {e}")
            return "Could not generate a recipe at this time."
