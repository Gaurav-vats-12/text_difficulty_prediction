pip install transformers

import streamlit as st
import pandas as pd
import requests

st.title('Levelingo')
st.write('Welcome to Levelingo!')

# Load the model
import streamlit as st
import requests
import os
from transformers import CamembertTokenizer, CamembertForSequenceClassification

def download_file(url, save_path):
    """ Helper function to download a file from a given URL to a specified save path """
    response = requests.get(url)
    with open(save_path, "wb") as file:
        file.write(response.content)

def load_model_and_tokenizer(model_dir):
    """ Load the model and tokenizer from the specified directory """
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def main():
    st.title('French Text Difficulty Classifier')

    # Define the GitHub URL for the model and tokenizer files
    base_url = 'https://github.com/your_username/your_repository/raw/main/app/cache/'
    model_files = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 
                   'sentencepiece.bpe.model', 'added_tokens.json', 'model.safetensors']

    model_dir = 'cached_model'
    os.makedirs(model_dir, exist_ok=True)

    # Download model files if not already downloaded
    if not os.path.exists(f'{model_dir}/config.json'):
        for file_name in model_files:
            download_file(f"{base_url}{file_name}", f"{model_dir}/{file_name}")

    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(model_dir)

    # Example usage in Streamlit
    user_input = st.text_area("Enter your French text here:")
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(-1).item()  # Simplified example prediction

        st.write(f"Predicted Label: {prediction}")

if __name__ == '__main__':
    main()


# Fetch french texts via newsapi
def fetch_french_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=fr&apiKey=e7c7cca4d5184b069f195de63ad0d86c"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    french_texts = [article['content'] for article in articles if article['content']]
    return french_texts
