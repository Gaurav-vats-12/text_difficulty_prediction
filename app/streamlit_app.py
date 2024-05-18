import streamlit as st
import pandas as pd
import requests

st.title('Levelingo')
st.write('Welcome to Levelingo!')

def fetch_french_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=fr&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    french_texts = [article['content'] for article in articles if article['content']]
    return french_texts
