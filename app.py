"""
Emotion Detection System - Streamlit Web Application
Author: Divyansh
Description: NLP-based emotion detection using TF-IDF + ML classifiers
"""

import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOTION_META = {
    "joy":      {"emoji": "😊", "color": "#FFD700", "bg": "#FFFDE7"},
    "sadness":  {"emoji": "😢", "color": "#5C9BD6", "bg": "#E3F2FD"},
    "anger":    {"emoji": "😠", "color": "#E53935", "bg": "#FFEBEE"},
    "fear":     {"emoji": "😨", "color": "#7B1FA2", "bg": "#F3E5F5"},
    "love":     {"emoji": "❤️",  "color": "#E91E63", "bg": "#FCE4EC"},
    "surprise": {"emoji": "😲", "color": "#FF9800", "bg": "#FFF3E0"},
}

MODEL_FILES = {
    "Logistic Regression":     "log_model.pkl",
    "Multinomial Naive Bayes": "nb_model.pkl",
}

VECTORIZER_FILE = "vectorizer.pkl"

# Mapping for models trained with integer-encoded labels
# Order matches the typical encoding: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
# Adjust this if your dataset used a different encoding
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        /* ── Global ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── Header ── */
        .main-title {
            text-align: center;
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #6B7280;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* ── Prediction card ── */
        .emotion-card {
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }
        .emotion-card:hover { transform: translateY(-2px); }
        .emotion-emoji  { font-size: 4rem; line-height: 1.2; }
        .emotion-label  { font-size: 1.8rem; font-weight: 700; margin: 0.3rem 0; }
        .emotion-conf   { font-size: 1rem; color: #6B7280; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
        .sidebar-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.12);
            margin: 2px 3px;
        }

        /* ── Predict button ── */
        div[data-testid="stButton"] > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.15s;
        }
        div[data-testid="stButton"] > button:hover {
            opacity: 0.90;
            transform: translateY(-1px);
        }

        /* ── Text area ── */
        textarea { border-radius: 10px !important; font-size: 0.97rem !important; }

        /* ── Footer ── */
        .footer {
            margin-top: 3rem;
            padding: 20px;
            text-align: center;
            border-top: 1px solid #E5E7EB;
            color: #9CA3AF;
            font-size: 0.88rem;
        }
        .footer a { color: #667eea; text-decoration: none; font-weight: 500; }
        .footer a:hover { text-decoration: underline; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOADERS  (cached so they don't reload every run)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_file: str):
    """Load a joblib model from disk. Returns None if file not found."""
    if not os.path.exists(model_file):
        return None
    return joblib.load(model_file)


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    """Load the TF-IDF vectorizer. Returns None if file not found."""
    if not os.path.exists(VECTORIZER_FILE):
        return None
    return joblib.load(VECTORIZER_FILE)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_emotion(text: str, model, vectorizer):
    """
    Transform input text with TF-IDF vectorizer and predict emotion.
    Returns (predicted_label, probabilities_dict or None).
    """
    vec = vectorizer.transform([text])
    raw_label = model.predict(vec)[0]

    # Convert numeric label → string if needed
    if isinstance(raw_label, (int, np.integer)):
        label = LABEL_MAP.get(int(raw_label), str(raw_label))
    else:
        label = str(raw_label)

    probs = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(vec)[0]
        # Also convert class keys to strings
        probs = {
            LABEL_MAP.get(int(cls), str(cls)) if isinstance(cls, (int, np.integer)) else str(cls): float(p)
            for cls, p in zip(model.classes_, raw)
        }

    return label, probs


# ─────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────
def render_header():
    st.markdown('<h1 class="main-title">🎭 Emotion Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Detect the emotion behind any piece of text using machine learning</p>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("## 📋 Project Info")
        st.markdown("---")

        st.markdown("**🧠 Models Available**")
        for name in MODEL_FILES:
            st.markdown(f"<span class='sidebar-badge'>✦ {name}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🏷️ Detectable Emotions**")
        cols = st.columns(2)
        for i, (emotion, meta) in enumerate(EMOTION_META.items()):
            cols[i % 2].markdown(f"{meta['emoji']} {emotion.capitalize()}")

        st.markdown("---")
        st.markdown("**⚙️ Technical Stack**")
        stack = ["Python", "Scikit-learn", "TF-IDF", "Streamlit", "Plotly", "Joblib"]
        for item in stack:
            st.markdown(f"<span class='sidebar-badge'>{item}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**📌 How to Use**")
        st.markdown(
            "1. Type or paste text below\n"
            "2. Choose a classifier\n"
            "3. Click **Predict Emotion**\n"
            "4. View results & probabilities"
        )


def render_emotion_card(label: str, confidence: float | None):
    """Render a styled card for the predicted emotion."""
    meta = EMOTION_META.get(label.lower(), {"emoji": "🤔", "color": "#6B7280", "bg": "#F9FAFB"})
    conf_html = (
        f'<p class="emotion-conf">Confidence: <strong>{confidence:.1%}</strong></p>'
        if confidence is not None else ""
    )
    st.markdown(
        f"""
        <div class="emotion-card" style="background:{meta['bg']}; border-left: 5px solid {meta['color']};">
            <div class="emotion-emoji">{meta['emoji']}</div>
            <div class="emotion-label" style="color:{meta['color']};">{label.upper()}</div>
            {conf_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probs: dict):
    """Render a horizontal Plotly bar chart of class probabilities."""
    emotions = list(probs.keys())
    values   = [probs[e] * 100 for e in emotions]
    colors   = [EMOTION_META.get(e.lower(), {}).get("color", "#667eea") for e in emotions]
    emojis   = [EMOTION_META.get(e.lower(), {}).get("emoji", "🤔") for e in emotions]
    labels   = [f"{em} {e.capitalize()}" for em, e in zip(emojis, emotions)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Prediction Probabilities", font=dict(size=16, color="#374151")),
        xaxis=dict(title="Probability (%)", range=[0, max(values) * 1.25], showgrid=True,
                   gridcolor="#F3F4F6"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=50, b=10),
        height=280,
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_footer():
    st.markdown(
        """
        <div class="footer">
            Developed with ❤️ by <strong>Divyansh</strong> &nbsp;|&nbsp;
            <a href="https://github.com/Div7anshKushwaha" target="_blank">GitHub</a>
            &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/divyansh-kushwaha-603616383?lipi=urn%3Ali%3Apage%3Ad_flagship3_messaging_conversation_detail%3BfLNr8qUvSs2a4zCd8heMyg%3D%3D" target="_blank">LinkedIn</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    inject_css()
    render_header()
    render_sidebar()

    # ── Input section ──
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### ✍️ Enter Text")
        user_text = st.text_area(
            label="Input Text",
            placeholder="Type or paste your text here…  e.g. 'I am so happy today!'",
            height=180,
            label_visibility="collapsed",
        )

        model_choice = st.selectbox(
            "🤖 Select Classifier",
            options=list(MODEL_FILES.keys()),
            help="Choose the machine-learning algorithm for prediction.",
        )

        predict_btn = st.button("🔮 Predict Emotion", use_container_width=True)

    # ── Result section ──
    with col_result:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            # ── Validation ──
            if not user_text.strip():
                st.warning("⚠️ Please enter some text before predicting.")
                return

            # ── Load artifacts ──
            with st.spinner("Loading model & vectorizer…"):
                vectorizer = load_vectorizer()
                model      = load_model(MODEL_FILES[model_choice])

            if vectorizer is None:
                st.error(
                    f"❌ `{VECTORIZER_FILE}` not found. "
                    "Place it in the same directory as `app.py`."
                )
                return

            if model is None:
                st.error(
                    f"❌ `{MODEL_FILES[model_choice]}` not found. "
                    "Place the model file in the same directory as `app.py`."
                )
                return

            # ── Predict ──
            with st.spinner("Analysing emotion…"):
                label, probs = predict_emotion(user_text.strip(), model, vectorizer)

            confidence = probs[label] if probs else None
            render_emotion_card(label, confidence)

            # ── Probability chart ──
            if probs:
                render_probability_chart(probs)
            else:
                st.info("ℹ️ This model does not expose class probabilities.")

        else:
            # Placeholder state before first prediction
            st.markdown(
                """
                <div style="border:2px dashed #D1D5DB; border-radius:12px; padding:40px;
                            text-align:center; color:#9CA3AF; margin-top:10px;">
                    <div style="font-size:3rem;">🎭</div>
                    <p style="font-size:1rem; margin-top:8px;">
                        Enter text on the left and click<br>
                        <strong>Predict Emotion</strong> to see results here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_footer()


if __name__ == "__main__":
    main()