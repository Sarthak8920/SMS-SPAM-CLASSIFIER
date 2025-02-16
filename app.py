import streamlit as st
import pickle
import streamlit as st


def load_model():
    with open("vectorizer.pkl", "rb") as vec_file, open("model.pkl", "rb") as model_file:
        vectorizer = pickle.load(vec_file)
        model = pickle.load(model_file)
    return vectorizer, model

vectorizer, model = load_model()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms = vectorizer.transform([input_sms])
    result = model.predict(transformed_sms)[0]
    st.header("Spam" if result == 1 else "Not Spam")
