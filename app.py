import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# App title
st.title("Hinglish Sentiment Analysis")

model = joblib.load(r"models\linear_svc.joblib")
tfidf_vectorizer = joblib.load(r"models\vectorizer.joblib")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

def advanced_preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

user_input = st.text_area("Enter Text")

if st.button("Analyze"):
    if user_input:
        processed_text = advanced_preprocess(user_input)
        
        test_array = tfidf_vectorizer.transform([processed_text]).toarray()
        
        predictions = model.predict(test_array)

        if predictions[0] == 0:
            st.error("Sentiment: Negative")
        elif predictions[0] == 1:
            st.info("Sentiment: Neutral")
        elif predictions[0] == 2:
            st.success("Sentiment: Positive")
    else:
        st.error("Please enter a valid text.")
