# poetry_generator_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import json
import logging

# ----------------------------
# Setup Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="🎨 AI Poetry Generator",
    page_icon="🎨",
    layout="centered",
)

# ----------------------------
# Model Joining (Split Parts)
# ----------------------------
MODEL_PATH = "poetry_generation_model.keras"
PART_FILES = [
    "poetry_model_part_aa",
    "poetry_model_part_ab",
    "poetry_model_part_ac",
    "poetry_model_part_ad",
]

def join_model_parts(part_files, output_file):
    if os.path.exists(output_file):
        return True
    try:
        with open(output_file, "wb") as outfile:
            for part in part_files:
                if not os.path.exists(part):
                    st.error(f"❌ Missing model part: {part}")
                    return False
                with open(part, "rb") as infile:
                    outfile.write(infile.read())
        return True
    except Exception as e:
        st.error(f"❌ Error joining model parts: {e}")
        return False

if not join_model_parts(PART_FILES, MODEL_PATH):
    st.stop()

# ----------------------------
# Load Word Mappings
# ----------------------------
@st.cache_data
def load_word_mappings():
    try:
        with open("word2idx.json", "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        with open("idx2word.json", "r", encoding="utf-8") as f:
            idx2word_str = json.load(f)
            idx2word = {int(k): v for k, v in idx2word_str.items()}
        return word2idx, idx2word
    except Exception as e:
        st.error(f"❌ Error loading word mappings: {e}")
        return {}, {}

word2idx, idx2word = load_word_mappings()
if not word2idx or not idx2word:
    st.stop()

# ----------------------------
# Load Trained Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# ----------------------------
# Text Generation Function
# ----------------------------
def generate_text(model, start_string, word2idx, idx2word, num_generate=50, temperature=1.0, top_k=5):
    try:
        input_words = start_string.lower().split()
        input_indices = [word2idx.get(word, word2idx.get("<UNK>", 1)) for word in input_words]
        input_eval = tf.expand_dims(input_indices, 0)

        text_generated = []
        for _ in range(num_generate):
            predictions = model(input_eval)
            predictions = predictions[:, -1, :] / temperature
            predictions_np = predictions.numpy()[0]

            unk_index = word2idx.get("<UNK>", 1)
            if unk_index < len(predictions_np):
                predictions_np[unk_index] = -np.inf

            top_k_indices = predictions_np.argsort()[-top_k:]
            top_k_probs = predictions_np[top_k_indices]
            top_k_probs = np.exp(top_k_probs - np.max(top_k_probs))
            top_k_probs /= np.sum(top_k_probs)

            selected_id = np.random.choice(top_k_indices, p=top_k_probs)
            next_word = idx2word.get(selected_id, "")
            if not next_word:
                break

            text_generated.append(next_word)
            input_eval = tf.concat([input_eval, tf.expand_dims([selected_id], 0)], axis=1)

        return start_string + ' ' + ' '.join(text_generated)
    except Exception as e:
        st.error(f"❌ Error during generation: {e}")
        return ""

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🎨 AI-Powered Poetry Generator")

st.markdown("""
Enter a starting phrase and customize the options in the sidebar to generate unique AI-powered poetry! 🎭
""")

st.sidebar.header("🛠️ Settings")
start_string = st.sidebar.text_input("Start Phrase", "Once upon a midnight dreary")
num_generate = st.sidebar.slider("Number of Words", 10, 200, 50, step=10)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.5, 1.5, 1.0, step=0.1)
top_k = st.sidebar.slider("Top-K Sampling", 1, 20, 5)

if st.sidebar.button("✨ Generate Poetry"):
    if not start_string.strip():
        st.warning("⚠️ Please enter a starting phrase.")
    else:
        with st.spinner("Composing your poem..."):
            poem = generate_text(
                model=model,
                start_string=start_string,
                word2idx=word2idx,
                idx2word=idx2word,
                num_generate=num_generate,
                temperature=temperature,
                top_k=top_k
            )
        poem = poem.replace("<UNK>", "").strip()
        st.subheader("🖋️ Your Generated Poem")
        st.write(poem)
else:
    st.info("📝 Enter your settings and press 'Generate Poetry' to begin!")

# ----------------------------
# Dark Theme CSS
# ----------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #1E1E1E;
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: white !important;
}
.sidebar .sidebar-content {
    background-color: #2D2D2D;
}
.stButton>button {
    background-color: #2e7bcf;
    color: white;
}
</style>
""", unsafe_allow_html=True)
