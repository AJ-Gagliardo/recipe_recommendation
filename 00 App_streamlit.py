import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from PIL import Image
import requests
from io import BytesIO  
import base64

# Function to preprocess text
def my_tokenizer(sentence):
    # Instantiate stemmer and stopwords from nltk
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Tokenize the sentence into words
    words = word_tokenize(sentence)

    # Making them lower case and remove punctuation
    stemmed_words = [stemmer.stem(word.lower()) for word in words if word.isalpha()]

    # Remove if it is a stop word
    filtered_words = [word for word in stemmed_words if word not in stop_words]

    return filtered_words

# Load data and models
tfidf_vectorizer = joblib.load('pkl/tfidf_vectorizer.pkl')
classifier_model = joblib.load('pkl/mylogreg_model.pkl')
TF_IDF_matrix = joblib.load('pkl/tfidf_content_matrix.pkl')
recipes = pd.read_feather('streamlit_files/5k_recipes.feather')

## Function to predict sentiment
def predict_sentiment(sentence, tfidf_vectorizer, classifier_model):
    # Vectorize the input sentence using the TfidfVectorizer
    sentence_vectorized = tfidf_vectorizer.transform([sentence])

    # Predict the sentiment using the classifier model
    sentiment = classifier_model.predict(sentence_vectorized)

    return sentiment[0]



# def predict_sentiment(sentence, tfidf_vectorizer, classifier_model, threshold):
#     # Vectorize the input sentence using the TfidfVectorizer
#     sentence_vectorized = tfidf_vectorizer.transform([sentence])
#     threshold = 0.4

#     # Predict the probabilities of each class using the classifier model
#     probabilities = classifier_model.predict_proba(sentence_vectorized)[0]
    
#     # Check if the difference between the two highest probabilities is greater than the threshold
#     if abs(max(probabilities) - sorted(probabilities)[-2]) > threshold:
#         # Get the predicted sentiment class
#         sentiment = classifier_model.predict(sentence_vectorized)[0]
#     else:
#         # Set sentiment as neutral or undecided (you can modify this based on your needs)
#         sentiment = 'neutral'

#     return sentiment



####s

# Define the search function
def search_recipes_by_ingredients(ingredients, recipes):
    mask = True
    for ingredient in ingredients:
        mask = mask & recipes['RecipeIngredientParts'].str.contains(ingredient, case=False)
    result = recipes.loc[mask]
    return result
    
    return result
    
  

def get_similar_recipes(recipe_name, tfidf_vectorizer, TF_IDF_matrix, recipes):
    filtered_recipes = recipes[recipes['Name'] == recipe_name]
    if len(filtered_recipes) == 0:
        st.write("Recipe not found.")
        return None
    
    recipe_index = filtered_recipes.index[0]
    tfidf_matrix = tfidf_vectorizer.transform(recipes['RecipeIngredientParts'])
    similarities = cosine_similarity(tfidf_matrix, dense_output=False)
    sim_df = pd.DataFrame({
        'RecipeName': recipes['Name'],
        'SimilarityScore': similarities[recipe_index, :].A.squeeze(),
        'Ingredients': recipes['RecipeIngredientParts'],
        'Instructions': recipes['RecipeInstructions'],
        'ImageLink': recipes['Images']
    })
    sim_df = sim_df.sort_values(by='SimilarityScore', ascending=False)
    
    # Display similar recipes with images
    for index, row in sim_df.head(4).iterrows():
        st.subheader(row['RecipeName'])
        st.text(row['Ingredients'])
        st.text(row['Instructions'])
        image_link = row['ImageLink']
        if pd.notnull(image_link):
            try:
                response = requests.get(image_link)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Image', use_column_width=True)
            except Exception as e:
                st.write("Error loading image:", e)
    
    return sim_df.head(4)





# Streamlit app

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

image_path = 'streamlit_files/chef001.png'  
image = Image.open(image_path)

desired_size = (200, 200) 
resized_image = image.resize(desired_size, Image.ANTIALIAS)

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h1 style="text-align: center;">TasteTrail</h1>
    </div>
    """,
    unsafe_allow_html=True)

st.markdown(
    f'<div style="display: flex; justify-content: center;">'
    f'<img src="data:image/png;base64,{image_to_base64(resized_image)}" alt="Your Image" width="{desired_size[0]}" height="{desired_size[1]}">'
    f'</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <h4 style="text-align: center;">Unleash you inner chef with every ingredient, one recipe at a time!</h1>
    </div>
    """,
    unsafe_allow_html=True)
# st.markdown('"Unleash you inner chef with every ingredient, one recipe at a time!"')

st.title('Sentiment Analysis')


input_sentence = st.text_input("Enter a sentence, be kind :heartbeat:, be mean :imp:, up to you")
if st.button("Predict"):
    sentiment = predict_sentiment(input_sentence, tfidf_vectorizer, classifier_model)
    # threshold = 0.4 ##############
    # Display the sentiment prediction
    if sentiment == 1:
        st.write("The sentence has a positive sentiment   :grin: :thumbsup:.")
    else:
        st.write("The sentence has a negative sentiment :thumbsdown::nauseated_face:.")


st.title('Recipe Search')

# User input for ingredients
ingredients = st.text_input('Enter ingredients (separated by commas)')

if st.button('Search'):
    if ingredients:
        ingredient_list = [ingredient.strip() for ingredient in ingredients.split(',')]
        search_results = search_recipes_by_ingredients(ingredient_list, recipes)
        if search_results is not None:
            st.write(f"Found {len(search_results)} recipes:")
            for index, row in search_results.head(4).iterrows():
                st.subheader(row['Name'])
                st.text(row['RecipeIngredientParts'])
                image_link = row['Images']
                if pd.notnull(image_link):
                    try:
                        response = requests.get(image_link)
                        image = Image.open(BytesIO(response.content))
                        st.image(image, caption='Image', use_column_width=True)
                    except Exception as e:
                        st.write("Error loading image:", e)
        else:
            st.write("No recipes found.")


st.title('Content-Based Recipe Recommender')


recipe_name = st.text_input("Enter a recipe name")
if st.button("Get Similar Recipes"):
    similar_recipes = get_similar_recipes(recipe_name, tfidf_vectorizer, TF_IDF_matrix, recipes)
    st.dataframe(similar_recipes)



image_path_git = 'streamlit_files/website.png'  
image_git = Image.open(image_path_git)

desired_size = (200, 200) 
resized_image = image.resize(desired_size, Image.ANTIALIAS)