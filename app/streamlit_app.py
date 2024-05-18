# Import necessary libraries
import streamlit as st
import requests
import os
import transformers
from transformers import CamembertTokenizer, CamembertForSequenceClassification, pipeline
import sentencepiece
import tokenizers
from newsapi import NewsApiClient

# NewsApi
newsapi = NewsApiClient(api_key='e7c7cca4d5184b069f195de63ad0d86c')
top_headlines = newsapi.get_top_headlines(language='fr', country='fr')


st.title('Levelingo')
st.write('Welcome to Levelingo!')

# Load the model from GitHub
def download_file_from_github(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        st.error("Failed to download file. Check the URL and network connection.")

def setup_model():
    """Setup the model by ensuring all necessary files are downloaded and loaded."""
    model_dir = 'cache'  # Correct path relative to the streamlit_app.py file
    os.makedirs(model_dir, exist_ok=True)

    # List of model files you need to download
    model_files = [
        'config.json',
        'model.safetensors',
        'added_tokens.json',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'sentencepiece.bpe.model'
    ]

    base_url = "https://raw.githubusercontent.com/vgentile98/text_difficulty_prediction/main/app/cache/"

    # Download model and tokenizer files
    for file_name in model_files:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            download_file_from_github(f"{base_url}{file_name}", file_path)

    # Load model and tokenizer
    try:
        tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

# Fetch french texts via newsapi
#def fetch_french_news():
    #newsapi = NewsApiClient(api_key='e7c7cca4d5184b069f195de63ad0d86c')
    #url = f"https://newsapi.org/v2/top-headlines?country=fr&apiKey=e7c7cca4d5184b069f195de63ad0d86c"
    #response = requests.get(url)
    #articles = response.json().get('articles', [])
    #french_texts = [article['content'] for article in articles if article['content']]
    #return french_texts

def main():
    model, tokenizer = setup_model()
    if model and tokenizer:
        nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
        
        # Fetch French news headlines
        top_headlines = newsapi.get_top_headlines(language='fr', country='fr')
        texts = [article['title'] for article in top_headlines['articles'] if article['title']]
        
        if texts:
            results = [nlp(text) for text in texts]
            for result in results:
                st.write(result)
        else:
            st.write("No news content available or failed to fetch news.")

if __name__ == '__main__':
    main()

