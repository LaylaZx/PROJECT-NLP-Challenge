import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import joblib


# Load the pre-trained model and cache it (so that we don't have to load it every time)
@st.cache_resource
def load_model():
    model = joblib.load('logistic_regression_model.pkl')
    return model

# Define the classes names
class_names = ['Fake', 'Real']

# Set up the Streamlit app
st.title("Fake and Real News Classifier")
st.write("Upload a CSV file with a 'text' column to classify news as Fake or Real.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def feature_extraction(data):
    # Feature extraction using TF-IDF
     tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=0.02)  
     return tfidf_vectorizer.fit_transform(data['text'])
 
def predict_news(text):
    model = load_model()
    preprocessed_text = preprocess_text(text)
    vectorized_text = feature_extraction(pd.DataFrame({'text': [preprocessed_text]}))
    return class_names[np.argmax(model.predict(vectorized_text))]


if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the required column is present
    if 'text' in df.columns:
        st.write("Preview of the uploaded data:")
        st.write(df.head())
        # Make predictions
        df['prediction'] = df['text'].apply(predict_news)#this will apply it for each row

        
        # Display the results
        st.write("Predictions:")
        st.write(df[['text', 'prediction']])
        
        # Download the results as a CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("The CSV file must contain a 'text' column.")
        
        
    
    