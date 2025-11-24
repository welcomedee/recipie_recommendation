import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("sooryaprakash12/cleaned-indian-recipes-dataset")

print("Path to dataset files:", path)

# Move the file to the current directory for easier access
# Find the csv file in the downloaded path
for file in os.listdir(path):
    if file.endswith(".csv"):
        source = os.path.join(path, file)
        destination = os.path.join(os.getcwd(), "Cleaned_Indian_Food_Dataset.csv")
        shutil.copy(source, destination)
        print(f"Moved {file} to {destination}")
        break
