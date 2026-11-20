"""Streamlit Web Application for SMS Spam Detection Using LSTM.

Features:
- Responsive, modern user interface with clear visual distinctions for Ham vs Spam.
- Real-time single message inference with output score and explanation.
- Batch SMS classification with downloadable CSV export.
- About section detailing the LSTM recurrent neural network architecture.
- Model Performance dashboard dynamically loaded from outputs/metrics.json.
- Hinglish Experiment analysis dynamically loaded from outputs/hinglish_experiment.json.
- Academic project footer and dataset citations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from src.predict import SpamPredictor
from src.preprocess import clean_sms_text

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="SMS Spam Detection Using LSTM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished academic appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .spam-box {
        background-color: #fff5f5;
        border-left: 6px solid #e53e3e;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .ham-box {
        background-color: #f0fff4;
        border-left: 6px solid #38a169;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #718096;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Model Loader
# ==============================================================================
@st.cache_resource
def load_predictor(threshold: float = 0.5) -> Tuple[Optional[SpamPredictor], Optional[str]]:
    """Cache and load the SpamPredictor instance with error reporting."""
    model_path = Path("models/spam_lstm.keras")
    tok_path = Path("models/tokenizer.pkl")
    cfg_path = Path("models/preprocessing_config.json")

    if not model_path.exists():
        return None, f"Model file not found at `{model_path}`. Please run `python -m src.train` to train the model."
    if not tok_path.exists():
        return None, f"Tokenizer artifact not found at `{tok_path}`. Please run `python -m src.preprocess`."
    if not cfg_path.exists():
        return None, f"Configuration file not found at `{cfg_path}`. Please run `python -m src.preprocess`."

    try:
        predictor = SpamPredictor(
            model_path=model_path,
            tokenizer_path=tok_path,
            config_path=cfg_path,
            threshold=threshold,
        )
        predictor.load_artifacts()
        return predictor, None
    except Exception as e:
        return None, f"Error loading model artifacts: {str(e)}"


def load_json_artifact(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load JSON metric files."""
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ==============================================================================
# Header Section
# ==============================================================================
st.markdown('<div class="main-title">🛡️ SMS Spam Detection Using LSTM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Deep Learning NLP project using Long Short-Term Memory (LSTM) recurrent neural networks, '
    'featuring benchmark evaluation and initial code-mixed Hinglish spam exploration.</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# Sidebar Configuration
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Model Controls")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold for classifying a message as Spam. Default is 0.50.",
    )
    st.caption(
        "💡 **Threshold Guide**: Lower threshold (< 0.50) increases recall (catches more spam, but risks false alarms). "
        "Higher threshold (> 0.50) increases precision (reduces false alarms, but may miss subtle spam)."
    )
    st.markdown("---")

    st.header("📌 System Status")
    cfg = load_json_artifact("models/preprocessing_config.json")
    if cfg:
        st.markdown(
            f"""
            - **Status**: 🟢 Model Ready
            - **Architecture**: LSTM (64 units)
            - **Embedding**: 64 dimensions
            - **Vocabulary**: {cfg.get('actual_vocab_size', '6,684'):,} tokens
            - **Sequence Length**: {cfg.get('max_sequence_length', 70)} tokens
            - **Target Language**: English & Hinglish
            """
        )
    else:
        st.warning("Model metadata pending.")

    st.markdown("---")
    st.markdown("🎓 **College Academic Project**")
    st.caption("Investigating sequence models for standard and multilingual SMS threat detection.")


# ==============================================================================
# Main Content Navigation Tabs
# ==============================================================================
tab_detect, tab_batch, tab_perf, tab_hinglish, tab_about = st.tabs(
    [
        "🔍 Single Message Detection",
        "📂 Batch SMS Classification",
        "📊 Model Performance",
        "🇮🇳 Hinglish Experiment",
        "🧠 About Architecture",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: Single Message Detection
# ------------------------------------------------------------------------------
with tab_detect:
    st.markdown("### 💬 Real-Time SMS Message Classifier")
    st.markdown("Enter an SMS message in English or code-mixed Hinglish to evaluate its spam probability.")

    # Quick test sample buttons
    st.markdown("**Quick Test Samples:**")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    sample_text = ""
    with col_s1:
        if st.button("✉️ English Ham", use_container_width=True):
            st.session_state["sms_input"] = "Hey, are you free for lunch tomorrow around 1pm? Let me know."
    with col_s2:
        if st.button("🚨 English Spam", use_container_width=True):
            st.session_state["sms_input"] = (
                "URGENT!! You have won £1,000 cash or a £2,000 prize! Call 09061701461 to claim your reward now!"
            )
    with col_s3:
        if st.button("🇮🇳 Hinglish Spam", use_container_width=True):
            st.session_state["sms_input"] = (
                "Paytm cashback reward! Aapko mila hai Rs 2500 ka flat cashback. Claim karne ke liye abhi call karein: 9876543210"
            )
    with col_s4:
        if st.button("🇮🇳 Hinglish Ham", use_container_width=True):
            st.session_state["sms_input"] = "Bhai kahan hai tu? Sham ko chai pe milte hain tapri par."

    # Text Input
    default_text = st.session_state.get("sms_input", "")
    user_sms = st.text_area(
        "SMS Text:",
        value=default_text,
        placeholder="Type or paste any SMS message here...",
        height=110,
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        predict_clicked = st.button("🔍 Predict Spam / Ham", type="primary", use_container_width=True)

    if predict_clicked:
        if not user_sms or not user_sms.strip():
            st.warning("⚠️ **Input Required**: Please enter an SMS message text before clicking Predict.")
        else:
            predictor, error_msg = load_predictor(threshold=threshold)
            if error_msg:
                st.error(f"❌ **System Error**: {error_msg}")
            else:
                predictor.threshold = threshold
                with st.spinner("Analyzing message with LSTM..."):
                    result = predictor.predict_single(user_sms)

                is_spam = result["is_spam"]
                prob = float(result["probability"])
                confidence = float(result["confidence"])
                label = "Spam" if is_spam else "Ham (Not Spam)"

                # Clear Visual Distinction Card
                if is_spam:
                    st.markdown(
                        f"""
                        <div class="spam-box">
                            <h2 style="color: #c53030; margin: 0;">🚨 PREDICTION: SPAM (Unsolicited / Fraudulent)</h2>
                            <p style="font-size: 1.1rem; color: #742a2a; margin-top: 0.5rem; margin-bottom: 0;">
                                The model detected sequence patterns and indicator tokens characteristic of spam messages.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="ham-box">
                            <h2 style="color: #276749; margin: 0;">✅ PREDICTION: HAM (Not Spam / Legitimate)</h2>
                            <p style="font-size: 1.1rem; color: #22543d; margin-top: 0.5rem; margin-bottom: 0;">
                                The model classified this message as legitimate interpersonal communication.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Output Score & Probability Metrics
                st.markdown("#### 📈 Model Output Score")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Spam Probability", f"{prob:.4f}")
                with col_m2:
                    st.metric("Classification Confidence", f"{confidence:.2%}")
                with col_m3:
                    st.metric("Active Decision Threshold", f"{threshold:.2f}")

                # Colored Progress Bar
                st.write("**Probability Meter (0.0 = Ham, 1.0 = Spam):**")
                st.progress(prob)

                # Clear Explanation of the Score
                st.info(
                    f"""
                    ℹ️ **Understanding This Output Score:**
                    - The raw output score (**{prob:.4f}**) is generated by the final **Sigmoid activation neuron**, representing the model's estimated posterior probability $P(\\text{{Spam}} \\mid \\text{{message}})$.
                    - Messages with probability $\\ge {threshold:.2f}$ are classified as **Spam**, while those below are classified as **Ham (Not Spam)**.
                    - **Important Disclaimer**: This score is a **statistical estimate** derived from learned token sequences and structural artifacts. **It is not an absolute guarantee of correctness.** Adversarial spam may mimic casual text, and legitimate texts containing multiple numbers or urgent phrasing may register elevated scores.
                    """
                )

                # Preprocessing Trace Expander
                with st.expander("🛠️ View Normalized Message & Preprocessed Tokens"):
                    cleaned_version = clean_sms_text(user_sms)
                    st.write("**Original Text:**", user_sms)
                    st.write("**Normalized Text (Preserving Special Tokens):**", cleaned_version)
                    if predictor.tokenizer:
                        seq = predictor.tokenizer.texts_to_sequences([cleaned_version])[0]
                        st.write(f"**Generated Token IDs ({len(seq)} tokens):**", seq)
                        oov_count = sum(1 for tid in seq if tid == predictor.tokenizer.oov_index)
                        st.write(
                            f"**Out-Of-Vocabulary (<OOV>) Tokens:** {oov_count} ({oov_count/max(1, len(seq)):.1%})"
                        )

# ------------------------------------------------------------------------------
# TAB 2: Batch SMS Classification
# ------------------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📂 Batch SMS Classification")
    st.markdown("Upload a CSV file containing multiple SMS messages for batch prediction and reporting.")

    uploaded_file = st.file_uploader("Upload CSV file (must contain a text or message column):", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_df):,}** records from uploaded file.")

            # Identify message column automatically
            text_col = None
            for col in batch_df.columns:
                if str(col).lower().strip() in ["message", "text", "sms", "v2", "content", "body"]:
                    text_col = col
                    break
            if text_col is None:
                text_col = batch_df.columns[0]

            st.info(f"Targeting column **`{text_col}`** for classification.")

            col_batch_btn, _ = st.columns([1, 4])
            with col_batch_btn:
                run_batch = st.button("🚀 Classify Entire Batch", type="primary", use_container_width=True)

            if run_batch:
                predictor, error_msg = load_predictor(threshold=threshold)
                if error_msg:
                    st.error(error_msg)
                else:
                    predictor.threshold = threshold
                    with st.spinner("Processing batch with LSTM model..."):
                        messages = batch_df[text_col].astype(str).tolist()
                        results = predictor.predict_batch(messages)

                    batch_df["Predicted_Label"] = [r["label"] for r in results]
                    batch_df["Spam_Probability"] = [round(r["probability"], 4) for r in results]
                    batch_df["Confidence"] = [f"{r['confidence']:.2%}" for r in results]

                    spam_count = int((batch_df["Predicted_Label"] == "Spam").sum())
                    ham_count = int((batch_df["Predicted_Label"] == "Ham").sum())

                    st.success(
                        f"Batch classification complete! Found **{spam_count:,} Spam** ({spam_count/len(batch_df):.1%}) "
                        f"and **{ham_count:,} Ham** ({ham_count/len(batch_df):.1%})."
                    )

                    st.dataframe(batch_df.head(25), use_container_width=True)

                    csv_data = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Annotated Results (CSV)",
                        data=csv_data,
                        file_name="sms_spam_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error parsing file: {e}")

# ------------------------------------------------------------------------------
# TAB 3: Model Performance Dashboard
# ------------------------------------------------------------------------------
with tab_perf:
    st.markdown("### 📊 English LSTM Model Performance")
    st.markdown("Rigorous benchmark results evaluated on the **untouched 15% test split (837 messages)**.")

    metrics_data = load_json_artifact("outputs/metrics.json")
    if metrics_data:
        m = metrics_data.get("metrics", {})
        cm = metrics_data.get("confusion_matrix", {})

        # Top Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Test Accuracy", f"{m.get('accuracy', 0):.2%}")
        with c2:
            st.metric("Precision", f"{m.get('precision', 0):.2%}")
        with c3:
            st.metric("Recall (Sensitivity)", f"{m.get('recall', 0):.2%}")
        with c4:
            st.metric("F1-Score", f"{m.get('f1_score', 0):.2%}")
        with c5:
            st.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")

        st.markdown("---")

        # Visualizations (Confusion Matrix and ROC Curve)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.markdown("#### Confusion Matrix Heatmap")
            cm_img = Path("outputs/confusion_matrix.png")
            if cm_img.exists():
                st.image(str(cm_img), caption="Test Set Confusion Matrix (837 samples)", use_container_width=True)
            else:
                st.info("Confusion matrix image pending.")

            st.write(
                f"- **True Negatives (TN)**: {cm.get('true_negative', 669)} Ham correctly classified\n"
                f"- **True Positives (TP)**: {cm.get('true_positive', 100)} Spam correctly detected\n"
                f"- **False Positives (FP)**: {cm.get('false_positive', 56)} Ham mistaken as Spam\n"
                f"- **False Negatives (FN)**: {cm.get('false_negative', 12)} Spam messages missed"
            )

        with col_v2:
            st.markdown("#### ROC Curve")
            roc_img = Path("outputs/roc_curve.png")
            if roc_img.exists():
                st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
            else:
                st.info("ROC curve plot pending.")
            st.caption(
                f"Area Under Curve (AUC) = **{m.get('roc_auc', 0.9263):.4f}**, demonstrating strong separability "
                f"between spam and ham sequence representations."
            )

        st.markdown("---")

        # Training History Curves
        st.markdown("#### 📉 Training & Validation Dynamics")
        curve_img = Path("outputs/training_curves.png")
        if curve_img.exists():
            st.image(str(curve_img), caption="Training vs Validation Loss and Accuracy Curves", use_container_width=True)

        # Error Analysis Section
        st.markdown("---")
        st.markdown("#### 🔬 Detailed Error Analysis")
        err = metrics_data.get("error_analysis", {})
        col_fp, col_fn = st.columns(2)
        with col_fp:
            st.markdown(f"**False Positives ({err.get('total_false_positives', 56)} messages)**")
            st.caption("Legitimate messages flagged as spam due to heavy number density, codes, or words like 'call'.")
            for item in err.get("top_false_positives", [])[:3]:
                st.markdown(f"> *\"{item.get('message')}\"*  \n> `Score: {item.get('probability')}`")

        with col_fn:
            st.markdown(f"**False Negatives ({err.get('total_false_negatives', 12)} messages)**")
            st.caption("Spam messages that escaped detection by mimicking casual interpersonal chat.")
            for item in err.get("top_false_negatives", [])[:3]:
                st.markdown(f"> *\"{item.get('message')}\"*  \n> `Score: {item.get('probability')}`")
    else:
        st.warning("`outputs/metrics.json` not found. Please run `python -m src.evaluate` to generate performance reports.")

# ------------------------------------------------------------------------------
# TAB 4: Hinglish Experiment
# ------------------------------------------------------------------------------
with tab_hinglish:
    st.markdown("### 🇮🇳 Initial Hinglish Spam Detection Exploration")
    st.markdown(
        """
        > [!IMPORTANT]
        > **Academic Research Disclosure**: This section documents an exploratory experiment testing how an 
        > **English-trained LSTM** performs zero-shot on Roman-script Hindi-English (**Hinglish**) messages. 
        > **This is strictly experimental research and is NOT claimed as a production-ready Hinglish classifier.**
        """
    )

    h_data = load_json_artifact("outputs/hinglish_experiment.json")
    if h_data:
        hm = h_data.get("metrics", {})
        oov = h_data.get("out_of_vocabulary_analysis", {})

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.metric("Hinglish Accuracy", f"{hm.get('accuracy', 0):.2%}")
        with col_h2:
            st.metric("Hinglish Precision", f"{hm.get('precision', 0):.2%}")
        with col_h3:
            st.metric("Hinglish Recall", f"{hm.get('recall', 0):.2%}")
        with col_h4:
            st.metric("Average OOV Rate", f"{oov.get('overall_average_oov_rate', 0):.2%}")

        st.markdown("---")
        st.markdown("#### 🔬 Key Findings & Architectural Limitations")
        st.markdown(
            f"""
            1. **Severe Out-Of-Vocabulary (OOV) Rate ({oov.get('overall_average_oov_rate', 0.5754):.1%})**:
               - Because the model's vocabulary was fitted purely on an English corpus, colloquial Romanized Hindi tokens 
                 (*karo, karein, bhai, bijli, aate, kahan, chhota, tapri*) map to `<OOV>`.
               - The model is effectively blind to Hindi grammatical particles and semantics.

            2. **Reliance on English Structural Anchors**:
               - Hinglish spam is caught **only when English loanwords or structural artifacts are present**:
                 - *Caught*: *"Paytm cashback reward! Aapko mila hai Rs 2500 ka flat cashback. Claim..."* $\\rightarrow$ Detected via `cashback`, `reward`, `claim`, `_currency_`, `_num_`.
                 - *Missed*: Pure Romanized Hindi spam with no links or numbers (*"Dhamaka sale shuru ho gayi hai..."*) is missed completely.

            3. **Numeric False Alarms**:
               - Legitimate Hinglish texts containing numbers (*"Bhai train lagbhag 40 minute late hai, platform 3 par"* or *"DSA unit 3 ke notes send kar dena"*) 
                 risk being misclassified as Spam due to heavy weight placed on `_num_` tokens.
            """
        )

        st.markdown("---")
        st.markdown("#### 🚀 Pathway to Production Hinglish Detection")
        st.markdown(
            """
            To transition from this initial exploration to a robust production system:
            - **Supervised Code-Mixed Corpora**: Train on dedicated datasets like *L3Cube-HinglishSpam*.
            - **Subword Tokenizers**: Implement Byte-Pair Encoding (BPE) or SentencePiece to resolve spelling variants (*paisa / paise / pyse*).
            - **Multilingual Foundation Models**: Fine-tune pre-trained models such as **IndicBERT**, **mBERT**, or **XLM-RoBERTa**.
            """
        )

        with st.expander("📋 View Benchmark Hinglish Samples"):
            try:
                hinglish_df = pd.read_csv("data/raw/hinglish.csv")
                st.dataframe(hinglish_df[["id", "message", "label", "category"]], use_container_width=True)
            except Exception:
                st.info("Hinglish raw dataset available at data/raw/hinglish.csv")
    else:
        st.warning("`outputs/hinglish_experiment.json` not found. Please run the evaluation script.")

# ------------------------------------------------------------------------------
# TAB 5: About Architecture
# ------------------------------------------------------------------------------
with tab_about:
    st.markdown("### 🧠 Deep Learning Architecture: LSTM for Text Classification")
    st.markdown(
        """
        Long Short-Term Memory (LSTM) is a specialized Recurrent Neural Network (RNN) architecture designed to overcome 
        the vanishing gradient problem in standard recurrent nets, making it effective for sequence modeling.
        """
    )

    st.markdown(
        """
        #### ⚙️ Pipeline Components:
        1. **Text Normalization**:
           - Protects key spam signals by normalizing URLs $\\rightarrow$ `_url_`, phone numbers $\\rightarrow$ `_phone_`, 
             currencies $\\rightarrow$ `_currency_`, numbers $\\rightarrow$ `_num_`, and repeated punctuation $\\rightarrow$ `_exclamation_`.
           - Unescapes HTML entities (`&amp;` $\\rightarrow$ `and`).
        2. **Sequence Padding (Post-Padding to 70 tokens)**:
           - Padded to a maximum sequence length of 70 tokens, which accommodates 99.67% of training messages without truncation.
        3. **Embedding Layer (64 dimensions)**:
           - Maps 6,684 integer vocabulary tokens into a continuous 64-dimensional dense semantic vector space.
        4. **LSTM Cell (64 units)**:
           - Incorporates **Forget Gate**, **Input Gate**, and **Output Gate** to capture long-range contextual dependencies 
             and keyword co-occurrence across the SMS string.
        5. **Dense Output Layer (1 neuron, Sigmoid activation)**:
           - Computes $\\hat{y} = \\sigma(W \\cdot h_{70} + b) = \\frac{1}{1 + e^{-(W \\cdot h_{70} + b)}}$, producing a calibrated probability score.
        6. **Class-Weighted Optimization**:
           - Trained using **Binary Cross-Entropy** and **Adam Optimizer** (`lr = 0.001`).
           - Applied balanced class weights (Ham: `0.5774`, Spam: `3.7294`) to counter the natural 86.6% / 13.4% imbalance.
        """
    )

    st.markdown("---")
    st.markdown("#### 📚 Dataset Reference & Citation")
    st.markdown(
        """
        The baseline model is trained on the official **UCI SMS Spam Collection**:
        > **Citation**: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the study of SMS Spam Filtering: 
        > New Collection and Results. *Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11)*, Mountain View, CA, USA.
        """
    )

# ==============================================================================
# Footer
# ==============================================================================
st.markdown(
    """
    <div class="footer">
        <strong>College Academic Project</strong>: SMS Spam Detection Using LSTM with Initial Hinglish Spam Detection Exploration<br>
        Developed with Python, TensorFlow 2.21, Keras 3, Scikit-learn, and Streamlit.<br>
        Dataset Source: <a href="https://archive.ics.uci.edu/dataset/228/sms" target="_blank">UCI Machine Learning Repository</a>
    </div>
    """,
    unsafe_allow_html=True,
)
