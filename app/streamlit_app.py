# Import necessary libraries
import streamlit as st
import requests
import os
import transformers
import sentencepiece as spm
from transformers import CamembertTokenizer, CamembertForSequenceClassification, pipeline
import tokenizers
import streamlit.components.v1 as components

st.title('Levelingo')

# Fetch news articles from MediaStack
mediastack_api_key = '34361d5ce77e0449786fe2d144e015a4'
base_url = "http://api.mediastack.com/v1/news"

# Select options for the API request
category = st.selectbox("What do you want to read about?", ['general', 'business', 'technology', 'entertainment', 'sports', 'science', 'health'], index=1)

# Function to check if the image URL is valid
def is_valid_image_url(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200 and 'image' in response.headers['Content-Type']
    except requests.RequestException:
        return False
        
# Fetch news articles from mediastack API
def fetch_news():
    params = {
        'access_key': mediastack_api_key,
        'languages': "fr",
        'categories': category,
        'limit': 1  
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()['data']
    else:
        st.error('Failed to retrieve news articles.')
        return []

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
    model_dir = 'cache' 
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

    github_base_url = "https://raw.githubusercontent.com/vgentile98/text_difficulty_prediction/main/app/cache/"

    # Download model and tokenizer files
    for file_name in model_files:
        file_path = os.path.join(model_dir, file_name)
        if not os.path.exists(file_path):
            download_file_from_github(f"{github_base_url}{file_name}", file_path)

    # Load model and tokenizer
    try:
        tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None
        
def main():
    model, tokenizer = setup_model()
    if not model or not tokenizer:
        st.error("Failed to load the NLP model and tokenizer.")
        return
        
    articles = fetch_news()
    if articles:
        for article in articles:
           if article['image'] and is_valid_image_url(article['image']):
                with st.container():
                    st.image(article['image'], width=300)
                    st.subheader(article['title'])
                    st.write(article['description'] if article['description'] else 'No description available.')
                    # Button to open article in an iframe within the app
                    with st.expander("Read Now"):
                        components.iframe(article['url'], height=450, scrolling=True)
                    st.markdown("---")
    else:
        st.write("No articles found. Try adjusting your filters.")


if __name__ == '__main__':
    main()


