# ------------------------------
# Real-Time AI-Powered Phishing‑Website Detection (Enhanced UI & Experience)
# ------------------------------
# Run with:  streamlit run phishing_detection_project.py

import streamlit as st
import pandas as pd
import numpy as np
import tldextract
import joblib
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from urllib.parse import urlparse

MODEL_FILE = "phishing_rf_advanced.joblib"

# ------------------------------
# Enhanced Feature Engineering
# ------------------------------
def extract_features(url):
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)
    features = {
        "url_length": len(url),
        "has_https": int(parsed_url.scheme == "https"),
        "num_dots": url.count("."),
        "has_ip": int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "domain_length": len(parsed_url.netloc),
        "subdomain_length": len(ext.subdomain),
        "has_at_symbol": int("@" in url),
        "has_hyphen": int("-" in parsed_url.netloc),
        "num_digits": sum(char.isdigit() for char in url),
        "count_https": url.lower().count("https")
    }
    return pd.DataFrame([features]), features

# ------------------------------
# Load or Train Model
# ------------------------------
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        data = pd.DataFrame({
            "url_length": [25, 75, 120, 180, 90, 150],
            "has_https": [1, 0, 1, 0, 1, 0],
            "num_dots": [2, 5, 1, 8, 3, 6],
            "has_ip": [0, 1, 0, 1, 0, 1],
            "domain_length": [15, 25, 18, 32, 22, 30],
            "subdomain_length": [5, 10, 2, 20, 8, 15],
            "has_at_symbol": [0, 1, 0, 1, 0, 1],
            "has_hyphen": [0, 1, 0, 1, 0, 1],
            "num_digits": [2, 12, 3, 9, 4, 10],
            "count_https": [1, 0, 2, 0, 1, 0],
            "label": [0, 1, 0, 1, 0, 1]
        })
        X = data.drop("label", axis=1)
        y = data["label"]
        model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model

# ------------------------------
# Streamlit UI - Smart AI Detection with Animation
# ------------------------------
model = load_or_train_model()

st.set_page_config(page_title="AI Phishing Detector", layout="wide")
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
        color: #f5f5f5;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI';
    }
    .big-font {
        font-size: 18px;
        font-weight: 500;
    }
    .scan-button-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: -10px;
    }
    .particles {
        position: fixed;
        width: 100%;
        height: 100%;
        z-index: -1;
        top: 0;
        left: 0;
    }
</style>
<canvas class="particles"></canvas>
<script src="https://cdn.jsdelivr.net/npm/tsparticles@2/tsparticles.bundle.min.js"></script>
<script>
tsParticles.load("tsparticles", {
  fullScreen: { enable: true },
  particles: {
    number: { value: 50 },
    color: { value: "#00ffff" },
    shape: { type: "circle" },
    opacity: { value: 0.3 },
    size: { value: 4 },
    move: { enable: true, speed: 1.5 }
  }
});
</script>
""", unsafe_allow_html=True)

st.title("Real-Time AI Phishing Detection with Smart Suggestions")
st.markdown("""
Analyze suspicious URLs in real-time. This smart detector not only gives a prediction,
but also offers **AI-powered suggestions** to understand *why* a URL is flagged.
""")

url_input = st.text_input("Enter a URL to scan:", "https://example.com")
st.markdown('<div class="scan-button-container">', unsafe_allow_html=True)
scan = st.button("Scan Now", key="scan_button")
st.markdown('</div>', unsafe_allow_html=True)

if scan and url_input:
    with st.spinner("Analyzing with machine learning magic..."):
        df, features_dict = extract_features(url_input)
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        confidence = proba[prediction] * 100

        st.subheader("Prediction Result")
        if prediction == 1:
            st.markdown(f"<h3 style='color:red'>Phishing Website Detected</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:limegreen'>Legitimate Website</h3>", unsafe_allow_html=True)

        st.progress(int(confidence))
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Feature Breakdown")
        st.dataframe(df.T.rename(columns={0: "Value"}))

        st.subheader("AI Suggestions & Warnings")
        suggestions = []
        if features_dict["has_ip"]:
            suggestions.append("• The URL uses an IP address. This is common in phishing attempts.")
        if features_dict["has_at_symbol"]:
            suggestions.append("• The presence of '@' may indicate redirection or obfuscation tactics.")
        if features_dict["has_hyphen"]:
            suggestions.append("• Domains with hyphens often mimic real brands (e.g., pay-pal.com).")
        if features_dict["num_dots"] > 4:
            suggestions.append("• Too many subdomains may suggest a suspicious structure.")
        if features_dict["num_digits"] > 10:
            suggestions.append("• High digit count could mean autogenerated or fake URLs.")
            
        if not suggestions:
            suggestions.append("• No obvious phishing traits detected. Still, always be cautious.")

        for s in suggestions:
            st.markdown(f"<div class='big-font'>🔹 {s}</div>", unsafe_allow_html=True)
