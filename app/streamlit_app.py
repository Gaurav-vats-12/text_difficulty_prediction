# Import necessary libraries
import streamlit as st
import requests
import os
import transformers
import sentencepiece 
#try:
    ##import sentencepiece as spm
    #st.success('SentencePiece is successfully imported!')
#except ImportError as e:
    #st.error(f'Failed to import SentencePiece: {e}')
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification, pipeline
import tokenizers
import streamlit.components.v1 as components
import traceback
from itertools import cycle  

# Initialize user data storage
if 'users' not in st.session_state:
    st.session_state['users'] = {}

# Initialize user data and levels
cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
users = {'user_id': {'level': 'A1', 'feedback_points': 0}}




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
@st.cache_data(allow_output_mutation=True)
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



# Dummy function to assign levels to articles
def assign_article_levels(articles):
    level_cycle = cycle(cefr_levels)  # Create a cycle iterator from CEFR levels
    valid_articles = [article for article in articles if is_valid_image_url(article['image'])]
    for article in valid_articles:
        article['level'] = next(level_cycle)  # Assign levels in a cyclic manner
    return valid_articles

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
    model_dir = 'text_difficulty_prediction/app/cache' 
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
    #try:
        #tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
        #return model, tokenizer
    #except Exception as e:
        #st.text("Error details:")
        #st.text(traceback.format_exc())  # This prints the traceback of the exception
        #st.error(f"Error loading model or tokenizer: {str(e)}")
        #return None, None




# Function to update user level based on feedback
def update_user_level(user_id, feedback):
    feedback_points = {'Too Easy': 1, 'Just Right': 0, 'Challenging': 0, 'Too Difficult': -1}
    
    user_data = users[user_id]
    user_data['feedback_points'] += feedback_points[feedback]
    
    # Thresholds for level change
    upgrade_threshold = 3 # Points needed to move up a level
    downgrade_threshold = -3 # Points needed to move down a level
    
    current_index = cefr_levels.index(user_data['level'])
    if user_data['feedback_points'] >= upgrade_threshold:
        new_index = min(current_index + 1, len(cefr_levels) - 1)
        user_data['level'] = cefr_levels[new_index]
        user_data['feedback_points'] = 0
    elif user_data['feedback_points'] <= downgrade_threshold:
        new_index = max(current_index - 1, 0)
        user_data['level'] = cefr_levels[new_index]
        user_data['feedback_points'] = 0
        
    return user_data['level']



        
def main():
    st.title('Levelingo')
    user_id = 'default_user'
    if 'users' not in st.session_state:
        st.session_state['users'] = users
    
    user_level = st.session_state['users'][user_id]['level']
    st.write(f"Your current level: {user_level}")

    articles = fetch_news()
    if articles:
        articles = [article for article in articles if article['level'] == user_level and is_valid_image_url(article['image'])]
        for idx, article in enumerate(articles):
            with st.container():
                st.image(article['image'], width=300)
                st.subheader(article['title'])
                st.write(f"Level: {article['level']}")
                st.write(article['description'])
                with st.expander("Read Now"):
                    components.iframe(article['url'], height=450, scrolling=True)

                feedback = st.radio(
                    "How difficult did you find this article?",
                    ('Too Easy', 'Just Right', 'Challenging', 'Too Difficult'),
                    key=f"feedback_{idx}"  # Unique key for each radio
                )

                if st.button('Submit Feedback', key=f"submit_{idx}"):
                    new_level = update_user_level(user_id, feedback)
                    st.session_state['users'][user_id]['level'] = new_level
                    st.experimental_rerun()  # Rerun the app to update displayed articles
                st.markdown("---")
    else:
        st.write("No articles found. Try adjusting your filters.")


if __name__ == '__main__':
    main()


