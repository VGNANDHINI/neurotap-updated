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

import streamlit as st

# --- About text ---
about_short = """
**NeuroTap** uses AI to analyze voice recordings and assess the risk of **Parkinson’s Disease**.

You can either:
- 📂 Upload a `.wav` or `.mp3` file  
- 🎤 Record directly from your laptop/phone mic
"""

about_long = """
**What is NeuroTap?**  
NeuroTap uses machine learning to analyse voice patterns, finger-tapping performance and spiral drawings to provide an early risk assessment for Parkinson's and other motor-neuro conditions.

**How it works (simple):**
1. Record a short voice sample and complete simple tapping & drawing tests.  
2. NeuroTap extracts features (MFCCs, jitter/shimmer, tapping timings, tremor in drawings).  
3. The model returns a risk score + easy guidance.  
4. Optionally connect to a nearby specialist or save tests to your history.

**Data & privacy:**  
All data is stored securely. You control sharing — nothing is shared with doctors or researchers without your explicit consent.

**Contact & support:**  
Email: <a href="mailto:gvns1029@gmail.com">gvns1029@gmail.com</a>  
Website: <a href="https://your-neurotap-website.example.com" target="_blank">your-neurotap-website.example.com</a>
"""

# --- Sidebar UI ---
def add_about_sidebar():
    # Title (top of sidebar)
    st.sidebar.title("ℹ️ About NeuroTap")

    # Short summary (render markdown)
    st.sidebar.markdown(about_short, unsafe_allow_html=True)

    # Collapsible expander (explicitly collapsed by default)
    with st.sidebar.expander("Learn more about NeuroTap", expanded=False):
        # Use st.markdown here to ensure content is inside the expander
        st.markdown(about_long, unsafe_allow_html=True)

    # Separator and contact button
    st.sidebar.markdown("---")
    if st.sidebar.button("Contact Support"):
        st.sidebar.info("Email: gvns1029@gmail.com")

# Call the function once in your app flow
add_about_sidebar()

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


# spiral canvas in put
# main.py

import streamlit as st
from spiral_drawing import show_spiral_canvas

def main():
    st.title("Spiral Drawing Parkinson’s Test")
    show_spiral_canvas()

if name == "main":
    main()

st.header("🖌️ Spiral Drawing Test")
st.write("Please draw a spiral starting from center, try to draw smoothly in one stroke.")

canvas_result = st_canvas(
    stroke_width=2,
    stroke_color="#000000",
    background_color="#ffffff",
    height=400, width=400,
    drawing_mode="freedraw",
    key="spiral_canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data  # This is a HxWx4 RGBA numpy array

    st.image(img, caption="Your drawing")

    # Call the analysis function from your spiral_analysis.py
    result = analyze_spiral(img, debug_plots=True)

    features = result["features"]
    visuals = result["visuals"]

    # Show risk score and risk level
    st.metric("Risk score", f"{features['risk_score']:.1f} / 100")
    st.write(f"Risk level: **{features['risk_level']}**")

    # Show feature details in a table/dict
    st.subheader("Extracted Features")
    st.write({
        "Stroke length (pixels)": features['stroke_length'],
        "Mean radius (pixels)": features['mean_radius'],
        "Radius std dev (pixels)": features['radius_std'],
        "Tremor metric (0-1)": features['tremor_metric'],
        "Curvature std dev": features['curvature_std'],
        "Branch points count": features['branch_points'],
        "Intersections count": features['intersections'],
        "Edge density": features['edge_density'],
    })

    # Show diagnostic visuals
    st.subheader("Diagnostics Visualization")
    if 'overview' in visuals:
        st.image(visuals['overview'], caption="Overview: original, binarized, edges, skeleton")
    if 'radial_hist' in visuals:
        st.image(visuals['radial_hist'], caption="Radial Distance Histogram")
    if 'curvature_plot' in visuals:
        st.image(visuals['curvature_plot'], caption="Curvature Plot")






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
