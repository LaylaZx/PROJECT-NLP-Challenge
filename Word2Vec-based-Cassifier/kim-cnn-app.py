import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load the Kim-CNN Model and Tokenizer ---
model = load_model('kim-cnn-model-on-text.h5')

with open('tokenizer-on-text.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set the max sequence length (must be same as used during training)
max_sequence_length = 12  # Adjust as needed

# --- Streamlit App Interface ---
st.title("Kim-CNN Text Classification")
st.write("Enter a text sample for classification or upload a CSV file with text data.")

# Select the input method: Text Input or CSV File Upload
input_option = st.radio("Choose input method", ("Text Input", "CSV File Upload"))

if input_option == "Text Input":
    user_input = st.text_area("Text Input", "")
    if st.button("Predict Text"):
        if user_input:
            # Preprocess the input text: tokenization and padding
            sequence = tokenizer.texts_to_sequences([user_input])
            padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
            
            # Make prediction using the Kim-CNN model
            prediction = model.predict(padded_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            st.write("Predicted class:", predicted_class)
            st.write("Prediction probabilities:", prediction)
        else:
            st.write("Please enter some text.")
            
elif input_option == "CSV File Upload":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of CSV data:")
            st.dataframe(df.head())
            
            # Let the user select the column that contains the text data
            text_column = st.selectbox("Select the column that contains text", options=df.columns)
            if st.button("Predict CSV"):
                if text_column:
                    texts = df[text_column].astype(str).tolist()
                    
                    # Preprocess all texts: tokenization and padding
                    sequences = tokenizer.texts_to_sequences(texts)
                    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
                    
                    # Make predictions using the Kim-CNN model
                    predictions = model.predict(padded_sequences)
                    predicted_classes = np.argmax(predictions, axis=1)
                    
                    # Append the predicted classes to the dataframe
                    df['Predicted_Class'] = predicted_classes
                    st.write("Predictions added to the dataframe:")
                    st.dataframe(df)
                else:
                    st.write("Please select a text column.")
        except Exception as e:
            st.write("Error processing file:", e)
