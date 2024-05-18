pip install -r requirements.txt

# Import necessary libraries
import streamlit as st
import requests
import os
from transformers import CamembertTokenizer, CamembertForSequenceClassification, pipeline

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
    model_dir = 'model'
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
    tokenizer = CamembertTokenizer.from_pretrained(model_dir)
    model = CamembertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def main():
    st.title("French Text Difficulty Prediction")
    st.write("Enter the French text below and press 'Predict' to get the difficulty level.")

    # Setup model and tokenizer
    model, tokenizer = setup_model()
    nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # User text input
    user_input = st.text_area("French Text", "Type your text here...")
    if st.button('Predict'):
        if user_input:
            # Perform prediction
            predictions = nlp(user_input)
            st.write(predictions)
        else:
            st.write("Please enter some text to predict its difficulty.")

if __name__ == '__main__':
    main()




# Fetch french texts via newsapi
def fetch_french_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=fr&apiKey=e7c7cca4d5184b069f195de63ad0d86c"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    french_texts = [article['content'] for article in articles if article['content']]
    return french_texts
