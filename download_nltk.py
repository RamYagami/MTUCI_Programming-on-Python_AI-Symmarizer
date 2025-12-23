# download_nltk.py
import nltk
import os

nltk_data_dir = "./nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("punkt_tab", download_dir=nltk_data_dir)
nltk.download("averaged_perceptron_tagger_eng", download_dir=nltk_data_dir)

print(f"✅ NLTK data saved to {nltk_data_dir}")
