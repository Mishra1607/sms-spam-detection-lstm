# SMS Spam Detection Using LSTM with Initial Hinglish Spam Detection Exploration

A modular, production-ready deep learning project for SMS spam detection using Long Short-Term Memory (LSTM) recurrent neural networks, featuring exploratory analysis on code-mixed **Hinglish** (Romanized Hindi-English) spam.

---

## 📌 Project Overview

Mobile SMS remains a primary vector for phishing, financial scams, lottery frauds, and promotional spam. While standard English SMS spam detection is well-researched, real-world spam in multilingual regions (particularly South Asia) heavily leverages **code-mixed Hinglish** (e.g., *"Aapka account block ho gaya hai, turant KYC verify karein"*). 

This project aims to:
1. **Benchmark LSTM / BiLSTM Networks**: Evaluate sequence modeling capabilities on standard SMS corpora.
2. **Explore Hinglish Code-Mixed Spam**: Investigate phonetic spelling variants, vocabulary distribution, and challenges in tokenizing Romanized Hindi.
3. **Provide an Interactive Deployment**: Deliver a Streamlit web application for real-time single and batch SMS inspection.

---

## 📊 Dataset Source & Academic Citation

The project utilizes the official **UCI SMS Spam Collection** dataset, preserved in its original unedited format.

- **Repository**: [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms) (Dataset ID: 228)
- **File Location**: `data/raw/SMSSpamCollection` (and original archive `data/raw/smsspamcollection.zip`)
- **Compilation**: Tiago A. Almeida (Federal University of São Carlos) & José María Gómez Hidalgo (R&D Department, Optenet).
- **Supported Formats**: 
  1. Original UCI tab-separated format without header (`SMSSpamCollection`)
  2. Kaggle 2-column or 5-column CSV format (`spam.csv` / `v1, v2`)

### Official Citation
```bibtex
@inproceedings{almeida2011sms,
  title     = {Contributions to the study of SMS spam filtering: new collection and results},
  author    = {Almeida, Tiago A and G{\'o}mez Hidalgo, Jos{\'e} Mar{\'\i}a and Yamakami, Akebo},
  booktitle = {Proceedings of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11)},
  pages     = {259--262},
  year      = {2011},
  address   = {Mountain View, CA, USA},
  publisher = {ACM}
}
```

---

## 📈 Exploratory Data Analysis (EDA) Summary

Validation and linguistic analysis conducted on the 5,572 raw records:

| Metric | Full Corpus | Ham (0) | Spam (1) |
| :--- | :--- | :--- | :--- |
| **Total Records** | 5,572 | 4,825 (**86.59%**) | 747 (**13.41%**) |
| **Missing Messages / Labels** | 0 / 0 | 0 | 0 |
| **Duplicate Message Texts** | 403 (5,169 unique) | 384 duplicates | 19 duplicates |
| **Conflicting Label Texts** | **0** (100% clean) | - | - |
| **Median Character Count** | 61 | **52** | **149** |
| **Mean Character Count** | 80.48 ± 59.94 | 71.48 ± 58.44 | 138.67 ± 28.87 |
| **Median Word Count** | 12 | **11** | **25** |
| **Mean Word Count** | 15.60 ± 11.42 | 14.31 ± 11.52 | 23.91 ± 5.78 |

### Key EDA Artifacts Saved in `outputs/`
- `outputs/class_distribution.png`: Bar and donut visualizations illustrating severe class imbalance.
- `outputs/message_length_distribution.png`: Character length KDE with the 160-character SMS limit and word count boxplots.
- `outputs/word_frequency_chart.png`: Top 20 discriminatory keywords in Spam vs. Ham.
- `outputs/eda_metrics_summary.json`: Serialized data validation and descriptive statistics.
- Interactive Notebook: `notebooks/eda.ipynb`.

---

## 📂 Directory Structure

```text
SMS-Spam-Detection/
│
├── data/
│   ├── raw/                # Immutable original datasets (SMSSpamCollection, smsspamcollection.zip)
│   └── processed/          # Cleaned, tokenized, and split data (train/val/test matrices)
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis (EDA) and prototyping
│   └── eda.ipynb           # Comprehensive exploratory data analysis notebook
│
├── src/                    # Core modular Python package
│   ├── __init__.py         # Package initialization
│   ├── preprocess.py       # Raw dataset loader (UCI/Kaggle), normalization, tokenization
│   ├── train.py            # LSTM / BiLSTM model architecture, compilation, training loops
│   ├── evaluate.py         # Evaluation metrics, confusion matrix, ROC-AUC, Hinglish slice tests
│   └── predict.py          # Inference class for single/batch message prediction
│
├── models/                 # Saved model weights (.keras), tokenizers, and vectorizers
├── outputs/                # Evaluation plots, confusion matrices, logs, and performance metrics
│   ├── class_distribution.png
│   ├── message_length_distribution.png
│   ├── word_frequency_chart.png
│   └── eda_metrics_summary.json
├── research/               # Research notes, literature papers, Hinglish lexicon & findings
│
├── app.py                  # Interactive Streamlit web interface
├── requirements.txt        # Pinned dependencies (target: Python 3.11)
├── .gitignore              # Ignores virtual environments, raw/processed data, model binaries
└── README.md               # Project documentation and guidelines
```

---

## 🛠️ Technology Stack

| Category | Tools / Libraries |
| :--- | :--- |
| **Language** | Python 3.11 (Recommended for TensorFlow compatibility) |
| **Deep Learning** | TensorFlow 2.15+ / Keras |
| **Data Processing** | Pandas, NumPy (<2.0.0) |
| **Machine Learning** | Scikit-learn, Joblib |
| **NLP** | NLTK (Tokenization, Stopwords) |
| **Visualization** | Matplotlib, Seaborn |
| **User Interface** | Streamlit |

---

## ⚙️ Environment Setup & Prerequisites

### 1. Python 3.11 Requirement
TensorFlow pre-built binaries require Python 3.9 through 3.12, with **Python 3.11** being the recommended stable version. 

> [!WARNING]
> Python 3.13 is currently not supported by official TensorFlow wheels. Please ensure Python 3.11 is used when setting up the virtual environment for model training.

To install Python 3.11 on Windows:
```powershell
winget install Python.Python.3.11
```
Or use Conda:
```bash
conda create -n sms-spam python=3.11 -y
conda activate sms-spam
```

### 2. Virtual Environment Setup
```bash
# Using standard Python 3.11 venv
py -3.11 -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### 3. Dependency Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Workflow Guide

### Stage 1: Data Placement & Preprocessing
1. Raw dataset is already placed in `data/raw/SMSSpamCollection`.
2. Inspect data via the interactive notebook:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
3. Execute preprocessing routines in `src/preprocess.py` to clean texts, map Hinglish colloquial terms, and save padded sequences into `data/processed/`.

### Stage 2: Model Training
- Run training with class weighting, early stopping, and automatic checkpointing:
  ```bash
  python -m src.train
  ```
  *(or `python src/train.py`)*
- **Architecture**:
  - Embedding Layer: 64 dimensions (Vocabulary: 6,684 tokens, Sequence Length: 70)
  - LSTM Layer: 64 units
  - Dense Output: 1 neuron with Sigmoid activation
- **Training Setup**:
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam (learning rate = 0.001)
  - Class Weights: Ham (0) = 0.5774, Spam (1) = 3.7294 (derived from training labels)
  - Callback: EarlyStopping on validation loss (restored best epoch weights)
- **Artifacts Generated**:
  - Saved model: `models/spam_lstm.keras`
  - Training curves: `outputs/training_curves.png`
  - Training history: `outputs/training_history.json`

### Stage 3: Evaluation & Diagnostics
- Evaluate classification performance and generate diagnostic plots:
  ```bash
  python -m src.evaluate
  ```
- **Test Performance on Untouched Split (837 messages: 725 Ham, 112 Spam)**:
  - **Accuracy**: `91.88%`
  - **Precision**: `64.10%`
  - **Recall**: `89.29%` (100 out of 112 spam messages detected)
  - **F1-Score**: `74.63%`
  - **ROC-AUC**: `92.63%`
- **Confusion Matrix**:
  - True Negatives (TN): `669` | False Positives (FP): `56`
  - False Negatives (FN): `12` | True Positives (TP): `100`
- **Outputs Generated**:
  - Metrics record: `outputs/metrics.json`
  - Confusion matrix heatmap: `outputs/confusion_matrix.png`
  - ROC curve plot: `outputs/roc_curve.png`

### Stage 4: Launch Web Interface
- Start the Streamlit application:
  ```bash
  streamlit run app.py
  ```

---

## 🔬 Hinglish Spam Exploration & Experimental Findings

### Experimental Dataset
- Located at [`data/raw/hinglish.csv`](data/raw/hinglish.csv), consisting of **30 manually curated and labeled Romanized Hindi-English SMS messages** (15 Spam, 15 Ham) representing realistic mobile scenarios (KYC/banking fraud, lottery/KBC scams, utility cutoffs, job frauds, alongside legitimate family/colleague chats).
- This dataset is strictly isolated as an evaluation benchmark and was **never mixed into the English training corpus**.

### Experimental Results (Zero-Shot English LSTM on Hinglish)
- **Accuracy**: `73.33%`
- **Precision**: `81.82%`
- **Recall**: `60.00%` (9 out of 15 spam messages caught)
- **F1-Score**: `69.23%`
- **ROC-AUC**: `81.33%`
- **Average Out-Of-Vocabulary (OOV) Rate**: **`57.54%`**
  - Spam OOV Rate: `51.67%`
  - Ham OOV Rate: `63.40%`
- **Detailed Findings JSON**: `outputs/hinglish_experiment.json`

### Academic Insights & Architectural Limitations
1. **Vocabulary Blindness & High OOV**: Because the tokenizer was trained exclusively on English SMS, colloquial Hindi tokens (*karo, bijli, aate, kahan, chhota, dahi, tapri*) map to `<OOV>`, losing syntactic and semantic meaning.
2. **Reliance on Structural Artifacts**: The English model successfully detects Hinglish spam **primarily when English loanwords or structural signals are present** (e.g., `_url_`, `_phone_`, `_num_`, *account*, *free*, *cashback*, *credit card*). Pure Romanized Hindi spam with no numbers or links (*"Dhamaka sale shuru ho gayi hai..."*) is missed (False Negative).
3. **Numeric False Alarms**: Legitimate Hinglish texts containing numbers (*"Bhai train 40 minute late hai, platform 3 par"* or *"DSA unit 3 ke notes send kar dena"*) risk being misclassified as Spam due to heavy numeric weighting.
4. **Pathway to Production Hinglish Detection**: Production-grade multilingual spam filtering requires:
   - Supervised code-mixed Hinglish training datasets (e.g., L3Cube-HinglishSpam).
   - Subword tokenization (Byte-Pair Encoding / SentencePiece) to handle phonetic spelling variations (*paisa*, *paise*, *pyse*).
   - Pre-trained multilingual language representations (IndicBERT, XLM-RoBERTa, or mBERT).

---

## 📜 License
Developed for academic / college project evaluation.
