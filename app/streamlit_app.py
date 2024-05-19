# Import necessary libraries
import streamlit as st
import requests
import os
import transformers
from transformers import CamembertTokenizer, CamembertForSequenceClassification, pipeline
import sentencepiece
import tokenizers
#from newsapi import NewsApiClient
#newsapi = NewsApiClient(api_key='e7c7cca4d5184b069f195de63ad0d86c')

st.title('Levelingo')
st.write('Welcome to Levelingo!')

# Function to predict language level based on your algorithm
def predict_language_level(text):
    # Placeholder for your algorithm
    num_words = len(text.split())
    if num_words < 50:
        return "A1"
    elif num_words < 100:
        return "A2"
    elif num_words < 150:
        return "B1"
    elif num_words < 200:
        return "B2"
    elif num_words < 250:
        return "C1"
    else:
        return "C2"

# Fetch news articles from NewsAPI
news_api_key = 'e7c7cca4d5184b069f195de63ad0d86c'
def fetch_news():
    url = f'https://newsapi.org/v2/top-headlines?language=fr&apiKey={news_api_key}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# Fetch news articles from MediaStack
mediastack_api_key = '34361d5ce77e0449786fe2d144e015a4'
base_url = "http://api.mediastack.com/v1/news"

# Select options for the API request
categories = st.multiselect("Choose categories:", ['general', 'business', 'technology', 'entertainment', 'sports', 'science', 'health'])

# Fetch news articles from mediastack API
def fetch_news():
    params = {
        'access_key': mediastack_api_key,
        'languages': "fr",
        'categories': ','.join(categories) if categories,
        'limit': 4  # Limit to 4 articles for demonstration
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error('Failed to retrieve news articles.')
        return []

# Display news articles in the app
def main():
    articles = fetch_news()
    if articles:
        for article in articles:
            st.image(article['image'] if article['image'] else '')
            st.subheader(article['title'])
            st.write('Published At:', article['published_at'])
            st.write(article['description'] if article['description'] else 'No description available.')
            
            # Expandable section to show article content
            with st.expander("Show Content"):
                st.write(article['content'] if article['content'] else 'Content not available.')
            st.markdown("---")
    else:
        st.write("No articles found. Try adjusting your filters.")


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


if __name__ == '__main__':
    main()

