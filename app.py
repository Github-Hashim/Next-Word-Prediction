import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page configuration
st.set_page_config(page_title="Next Word Prediction", page_icon="📝", layout="centered")

## Load the LSTM model
model = load_model('next_word_lstm.h5')

## Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Add custom CSS for visibility
st.markdown("""
    <style>
    .predicted-word {
        background-color: #e0f7fa;
        color: #00796b;
        padding: 10px;
        border-radius: 5px;
        font-size: 1.2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Create the UI
st.title('Next Word Prediction with LSTM')

st.write("""
This application predicts the next word in a given sequence using a pre-trained Long Short-Term Memory (LSTM) model. 
Enter a phrase or sequence of words, and the model will attempt to predict the next word based on the input.
""")

# User input
input_text = st.text_input("Enter a sequence of words", "To be or not to")

# Predict next word on button click
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    if next_word:
        st.markdown(f'<div class="predicted-word">The predicted next word is: <b>{next_word}</b></div>', unsafe_allow_html=True)
    else:
        st.error("Unable to predict the next word. Please try a different sequence.")

# Footer
st.markdown("""
    <div style="margin-top: 50px; text-align: center;">
        <h2>Developed using LSTM and Streamlit for Next Word Prediction</h2>
    </div>
    """, unsafe_allow_html=True)
