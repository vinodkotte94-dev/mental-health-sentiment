import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Mental Health Sentiment App", page_icon="🧠")

# -----------------------------
# Dataset
# -----------------------------
data = {
    "text": [
        # Positive
        "I am happy","I feel happy","I am very happy today","Life is beautiful",
        "I feel amazing","I feel great","I am good","I am feeling good",
        "I am excited","I feel confident","I am proud","I am satisfied",
        "Everything is going well","I feel peaceful","I feel relaxed",
        "Today is a wonderful day","I feel positive","I am hopeful",
        "I feel motivated","I am not sad anymore",

        # Negative
        "I am sad","I feel sad","I am very sad","I feel depressed",
        "I am depressed","I feel anxious","I feel stressed","I feel lonely",
        "I feel hopeless","I am not happy","I am not good",
        "I am not feeling well","I feel terrible","I feel bad",
        "Nothing makes me happy","I am tired of everything",
        "I feel negative","I am frustrated","I feel upset",
        "I feel miserable"
    ],
    "label": ["Positive"]*20 + ["Negative"]*20
}

df = pd.DataFrame(data)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_text"] = df["text"].apply(preprocess)

# -----------------------------
# Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# -----------------------------
# Model
# -----------------------------
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X, y)
accuracy = model.score(X, y)

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Mental Health Sentiment Detection App")
st.write("Detects whether text expresses **Positive** or **Negative** sentiment.")
st.write(f"### 📊 Model Accuracy: {round(accuracy*100,2)}%")

st.markdown("---")

user_input = st.text_area("✍️ Enter your text here:")

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip():
        cleaned = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max()

        if prediction == "Positive":
            st.success(f"✅ Prediction: {prediction}")
        else:
            st.error(f"⚠️ Prediction: {prediction}")

        st.info(f"📈 Confidence Score: {round(probability*100,2)}%")
    else:
        st.warning("⚠️ Please enter some text.")