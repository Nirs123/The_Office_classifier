import joblib
import spacy
import string
import streamlit as st

nlp = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.STOP_WORDS

def preprocess_text(text):
    text = text.lower().strip()
    text = text.replace('\n', ' ')
    text = ' '.join([word for word in text.split() if word not in spacy_stopwords])
    text = ''.join([char for char in text if char not in string.punctuation])

    doc = nlp(text)
    lemmatized = [token.lemma_ for token in doc]
    text = ' '.join(lemmatized)

    return text

# Chargez le modèle enregistré
rl_classifier = joblib.load('rl_classifier.joblib')
tfidf_vectorizer = joblib.load('vectorizer.joblib')

def main():
    st.title("Which character of The Office are you ?")

    user_input = st.text_area("Enter a sentence to see which character of The Office you are")

    if st.button("Predict"):
        user_input = preprocess_text(user_input)
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        prediction = rl_classifier.predict(user_input_tfidf)
        st.markdown(f"## The character you are is **{prediction[0]}**")

        st.image(f"images/{prediction[0].lower()}.jpg", width=700)

if __name__ == "__main__":
    main()
