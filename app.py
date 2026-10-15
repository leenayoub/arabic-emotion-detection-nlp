import html
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import streamlit as st


OUTPUT_DIR = Path("phase2_complete_outputs")
LOGISTIC_MODEL_PATH = OUTPUT_DIR / "tfidf_logistic_regression.joblib"
ARABERT_MODEL_PATH = OUTPUT_DIR / "arabert_best_model"

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.I)
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")

ARABIC_LABELS = {
    "anger": "الغضب",
    "anticipation": "الترقب",
    "confusion": "الحيرة",
    "disgust": "الاشمئزاز",
    "fear": "الخوف",
    "happiness": "السعادة",
    "neutral": "محايد",
    "optimism": "التفاؤل",
    "pessimism": "التشاؤم",
    "sadness": "الحزن",
    "surprise": "المفاجأة",
}

EMOTION_EMOJIS = {
    "anger": "😡",
    "anticipation": "⏳",
    "confusion": "😕",
    "disgust": "🤢",
    "fear": "😨",
    "happiness": "😊",
    "neutral": "😐",
    "optimism": "🌟",
    "pessimism": "🌧️",
    "sadness": "😢",
    "surprise": "😮",
}


def clean_text(value):
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_PATTERN.sub(" URL ", text)
    text = MENTION_PATTERN.sub(" USER ", text)
    text = re.sub(r"#(?=\w)", "", text)
    text = DIACRITICS.sub("", text).replace("ـ", "")
    text = text.translate(
        str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
    )
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = "".join(
        character
        if (
            character.isalnum()
            or character.isspace()
            or character in "!?_"
            or ord(character) > 0x1F000
        )
        else " "
        for character in text
    )
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_logistic_model():
    saved = joblib.load(LOGISTIC_MODEL_PATH)
    return saved["extractor"], saved["classifier"]


@st.cache_resource
def load_arabert_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ARABERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(ARABERT_MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_logistic(text):
    extractor, classifier = load_logistic_model()
    features = extractor.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    label = str(classifier.classes_[best_index]).lower()
    return label, float(probabilities[best_index]), classifier.classes_, probabilities


def predict_arabert(text):
    import torch

    tokenizer, model, device = load_arabert_model()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    best_index = int(np.argmax(probabilities))
    label = str(model.config.id2label[best_index]).lower()
    labels = [str(model.config.id2label[index]).lower() for index in range(len(probabilities))]
    return label, float(probabilities[best_index]), labels, probabilities


st.set_page_config(
    page_title="Arabic Emotion Detection",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Arabic Emotion Detection")
st.caption("Emotion Detection from Text Using NLP and Machine Learning Approaches")

model_choice = st.selectbox(
    "Choose the classification model",
    ["TF-IDF + Logistic Regression", "Fine-tuned AraBERT"],
)

arabic_text = st.text_area(
    "Enter Arabic text",
    placeholder="مثال: أنا سعيد جداً بهذا الخبر 😊",
    height=140,
)

if st.button("Predict Emotion", type="primary", use_container_width=True):
    processed_text = clean_text(arabic_text)

    if not processed_text:
        st.warning("Please enter a valid Arabic text before prediction.")
    else:
        try:
            with st.spinner("Analyzing the text..."):
                if model_choice == "TF-IDF + Logistic Regression":
                    label, confidence, labels, probabilities = predict_logistic(processed_text)
                else:
                    label, confidence, labels, probabilities = predict_arabert(processed_text)

            arabic_label = ARABIC_LABELS.get(label, label)
            emotion_emoji = EMOTION_EMOJIS.get(label, "🔎")

            st.success(f"{emotion_emoji} Predicted Emotion: {label.title()} - {arabic_label}")
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")

            with st.expander("View cleaned text"):
                st.write(processed_text)

            ranked = sorted(
                zip([str(item).lower() for item in labels], probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:5]

            st.subheader("Top Predictions")
            for prediction_label, probability in ranked:
                translated = ARABIC_LABELS.get(prediction_label, prediction_label)
                st.write(
                    f"{EMOTION_EMOJIS.get(prediction_label, '•')} "
                    f"{prediction_label.title()} - {translated}: {probability * 100:.2f}%"
                )
                st.progress(float(probability))

        except FileNotFoundError:
            st.error(
                "The selected saved model was not found. Check that "
                "tfidf_logistic_regression.joblib and arabert_best_model are inside "
                "phase2_complete_outputs."
            )
        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")

st.divider()
st.caption(
    "The predictions are experimental and may be affected by dialect, short context, "
    "class imbalance, and overlapping emotions."
)
