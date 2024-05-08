
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_lg")

def fake_real_prediction(input_data):
    # Loading the saved model
    loaded_model = pickle.load(open('trained model.sav', 'rb'))
    
    # Create a DataFrame with the input data
    df_input = pd.DataFrame({'Text': input_data})
    
    # Apply the nlp() function to generate vectors
    df_input['vector'] = df_input['Text'].apply(lambda text: nlp(text).vector)
    
    # Reshape the vector
    vector = np.array(df_input['vector'].tolist())  # Convert DataFrame column to list of arrays
    vector = vector.reshape(len(vector), -1)  # Reshape to 2D array
    
    # Predict using the model
    result = loaded_model.predict(vector)
    
    # Check the result
    if result[0] == 1:  # Accessing the first element of the result array
        return 'It is Real'
    else:
        return 'It is fake'

def main():
    # Setting the title of the application page
    st.title('Fake Vs Real News Prediction')
    
    # Creating the input text box
    input_text_sentence = st.text_input('Enter the sentence to predict')
    
    # Code for prediction
    sentence = ''
    
    # Creating the button
    if st.button('Prediction result'):
        sentence = fake_real_prediction([input_text_sentence])
    
    # Displaying the prediction result
    st.write(sentence)

if __name__ == "__main__":
    main()