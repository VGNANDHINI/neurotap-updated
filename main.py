import streamlit as st
import json
import numpy as np
import pickle
import librosa
import soundfile as sf
import tempfile
from st_audiorec import st_audiorec

# Page Config
st.set_page_config(page_title="NeuroTap - Parkinson's Detection", page_icon="🧠", layout="centered")

# Load Doctors data
def load_doctors():
    with open("doctors.json", "r") as f:
        doctors = json.load(f)
    return doctors

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #f0f8ff, #e6f2ff);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("ℹ️ About NeuroTap")
st.sidebar.info("""
**NeuroTap** uses AI to analyze voice recordings 
and assess the risk of **Parkinson’s Disease**.

You can either:
- 📂 Upload a `.wav` or `.mp3` file  
- 🎤 Record directly from your laptop/phone mic  


 **Why use NeuroTap?**
    - Multi-modal input (voice + tapping + drawing) for better accuracy.  
    - Easy remote testing and trend tracking.  
    - Secure data handling and optional telemedicine connections.

     **Data & privacy:**  
    All data is stored securely. You control sharing — nothing is shared with doctors or researchers without your explicit consent.

    **Contact & support:**  
    Email: gvns1029@gmail.com.com  
    Learn more: https://your-neurotap-website.example.com

""")

# Load ML Model
with open("voice_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature Extraction
def extract_voice_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=22)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        st.error(f"Error extracting MFCC: {e}")
        return None

# Prediction
def predict_voice(mfcc_vector):
    if mfcc_vector is None:
        return "Error", 0, "Could not process voice file."

    proba = model.predict_proba([mfcc_vector])[0][1]
    percent = round(proba * 100)

    if percent < 20:
        return "🟩 Very Low Risk", percent, "Your voice shows no sign of Parkinson’s."
    elif percent < 40:
        return "🟩 Low Risk", percent, "Minor voice variations found — healthy."
    elif percent < 60:
        return "🟨 Monitor", percent, "Some voice traits overlap. Recheck suggested."
    elif percent < 80:
        return "🟧 Moderate Risk", percent, "Speech patterns suggest possible early signs."
    else:
        return "🟥 High Risk", percent, "Strong vocal patterns linked to Parkinson’s. Please consult a doctor."


# Main UI
st.title("🧠 NeuroTap – Parkinson's Detection from Voice")
st.write("Upload a voice sample or record your voice live for instant analysis.")

option = st.radio("Choose Input Method:", ["📂 Upload File", "🎤 Record with Laptop/Phone Mic"])

# --- File Upload ---
if option == "📂 Upload File":
    voice_file = st.file_uploader("Upload your voice sample (.wav or .mp3)", type=["wav", "mp3"])
    if st.button("🔍 Predict from File"):
        if voice_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                data, samplerate = sf.read(voice_file)
                sf.write(temp.name, data, samplerate)
                features = extract_voice_features(temp.name)

            level, score, description = predict_voice(features)
            st.subheader(level)
            st.progress(score)
            st.markdown(f"### 📊 Confidence: **{score}%**")
            st.info(description)
        else:
            st.warning("Please upload a voice sample.")


# --- Add this new section at the end or where it fits ---

st.header("Nearby Specialists")
user_city = st.text_input("Enter your city")

if user_city:
    doctors = load_doctors()
    filtered = [doc for doc in doctors if doc['city'].lower() == user_city.lower()]
    
    if filtered:
        for doc in filtered:
            st.subheader(doc['name'])
            st.write(f"Specialization: {doc['specialization']}")
            st.write(f"Address: {doc['address']}")
            st.write(f"Phone: {doc['phone']}")
            st.write(f"Email: {doc['email']}")
            st.write(f"[Website]({doc['website']})")
            st.write("---")
    else:
        st.write("No doctors found in this location.")


# --- Record with Laptop/Phone Mic ---
elif option == "🎤 Record with Laptop/Phone Mic":
    st.write("Press the record button below to capture your voice:")

    audio_data = st_audiorec()

    if audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_data)
            temp.flush()
            features = extract_voice_features(temp.name)

        level, score, description = predict_voice(features)
        st.subheader(level)
        st.progress(score)
        st.markdown(f"### 📊 Confidence: **{score}%**")
        st.info(description)

        # Playback
        st.audio(audio_data, format="audio/wav")
