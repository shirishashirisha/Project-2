import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# data preprossing


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    # Stem the tokens
    stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
    
    # Join the processed tokens into a single string
    processed_text = " ".join(stemmed_tokens)
    
    return processed_text

    
# load the pickle files
tfidf_vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_svm.pkl','rb'))

st.title("Hotel Review Analysis")
st.image('https://uploads-ssl.webflow.com/60fd4503684b46390cc0d337/6346af0655b9625759e278ec_01.png')
st.sidebar.image('https://media.istockphoto.com/id/507474468/photo/banner-with-the-phrase-cut.jpg?s=612x612&w=0&k=20&c=JPlwsVUZ0y4pdOUzGsEAUIF4m40O2G0ltbrHzmI4yck=')

# Create a radio button to select a value
selected_value = st.sidebar.radio("Select:", [' ','Business Objective', 'Project Members', 'Trainer'])

if selected_value == 'Business Objective':
    st.title('Business Objective')
    st.write('''The major objective is what are the attributes that travelers are
considering while selecting a hotel. With this manager can understand which
elements of their hotel influence more in forming a positive review or improves
hotel brand image.''')

if selected_value == 'Project Members':
    st.title('Project Members')
    st.write('''1. Yenneti lekhasree
2. Atike Bhadralaxmi

3. Muddangula.Shirisha

4. Dappu vaishnavi

5. Akshitha mudhiraj

6. karavadi Jayalakshmi

7. Hema gorentla

''')

if selected_value == 'Trainer':
    st.title('Trainer')
    st.write('Advaith')




input_text = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. Preprocess
    transformed_text = preprocess_text(input_text)

    # 2. Vectorize
    vector_input = tfidf_vectorizer.transform([transformed_text])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 0:
        st.header("Negative Review")
    else:
        st.header("Positive Review")
