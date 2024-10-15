import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load stopwords once
stop_words = set(stopwords.words('english'))

def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove non-alphanumeric characters and stopwords, and stem the tokens
    processed_tokens = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(processed_tokens)

# Load TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Animation and text styles for the title
st.markdown("""
    <style>
    @keyframes fadeIn {
        0% {
            opacity: 0;
        }
        100% {
            opacity: 1;
        }
    }
    .title {
        animation: fadeIn 1.5s;
        color: #003366; /* Dark Blue */
        font-family: 'Arial Black', sans-serif;
        font-size: 3em;
        text-align: center;
        padding: 10px;
        margin-bottom: 20px;
    }
    .text {
        font-family: 'Calibri', sans-serif;
        font-size: 1.2em;
        color: #333; /* Dark Gray */
        margin-bottom: 10px;
    }
    .button {
        background-color: #ff9933; /* Orange */
        color: #fff; /* White */
        font-family: 'Calibri', sans-serif;
        font-size: 1.2em;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .button:hover {
        background-color: #cc6600; /* Dark Orange */
    }
    </style>
    <h1 class="title">Info Classifier</h1>
""", unsafe_allow_html=True)

# Display logo
#st.image('my logo.png', width=200)

input_sms = st.text_area("Enter the message", height=100)

if st.button('Predict', key='predict_button', help='Click to predict',):
    if not input_sms.strip():
        st.error("Please enter a message.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Convert sparse input to dense array
        dense_vector_input = vector_input.toarray()
        # Predict
        result = model.predict(dense_vector_input)[0]
        # Display
        st.header("Prediction:")
        if result == 0:
            st.error("Fake")
        elif result == 1:
            st.success("Legit "
                       "or Cannot predict(model might not be trained on specific type of data)")
        else:
            st.warning("Spam")
