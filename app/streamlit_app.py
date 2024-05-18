import streamlit as st
import pandas as pd
import requests

st.title('Levelingo')
st.write('Welcome to Levelingo!')

# Load the model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CamembertForSequenceClassification.from_pretrained('path_to_model')
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()
tokenizer = CamembertTokenizer.from_pretrained('path_to_tokenizer')

# Fetch french texts via newsapi
def fetch_french_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=fr&apiKey=e7c7cca4d5184b069f195de63ad0d86c"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    french_texts = [article['content'] for article in articles if article['content']]
    return french_texts
