import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path
import os
import sys
from dotenv import load_dotenv

# Add current directory to sys.path
sys.path.append(os.getcwd())

from gemini_client import GeminiClient
from recommender import Recommender

def test_system():
    print("--- Interactive Recipe Recommender ---")
    print("Loading models... please wait.")
    
    try:
        client = GeminiClient()
        recommender = Recommender()
        
        if recommender.df.empty:
             print("FAILURE: Dataset not loaded. Check 'Cleaned_Indian_Food_Dataset.csv'.")
             return
             
        print("\nSystem Ready! Enter 'q' or 'exit' to quit.")
        
        while True:
            user_input = input("\nEnter ingredients (e.g., 'paneer, spinach'): ").strip()
            
            if user_input.lower() in ['q', 'exit']:
                print("Exiting...")
                break
                
            if not user_input:
                continue

            print(f"Processing: '{user_input}'...")
            
            # 1. Normalize
            normalized = client.normalize_ingredients(user_input)
            print(f"Normalized: {normalized}")
            
            # 2. Recommend
            recommendations = recommender.recommend(normalized, top_k=5)
            
            if recommendations:
                print(f"\nFound {len(recommendations)} recommendations:")
                for i, rec in enumerate(recommendations):
                    print(f"\n  {i+1}. {rec['name']}")
                    print(f"     Match Score: {rec['match_score']:.2f}")
                    print(f"     Est. Servings: {rec['estimated_servings']}")
                    print(f"     Time: {rec['cooking_time']} mins")
                    # print(f"     Ingredients: {rec['ingredients'][:100]}...") 
            else:
                print("\nNo direct matches found in dataset.")
                print("Generating fallback recipe with Gemini...")
                fallback = client.generate_fallback_recipe(user_input)
                print(f"\n--- Fallback Suggestion ---\n{fallback}\n---------------------------")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_system()
