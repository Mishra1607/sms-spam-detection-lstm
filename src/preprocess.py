"""Data Preprocessing and Loading Module for SMS Spam Detection.

Handles:
- Raw dataset ingestion (supporting official UCI tab-separated format with csv.QUOTE_NONE
  and Kaggle CSV formats).
- Working DataFrame normalization (standardizing column names and binary label encoding: ham=0, spam=1).
- Careful text normalization preserving spam indicators (URLs, phone numbers,
  currency symbols, numbers, and repeated punctuation).
- Stratified dataset partitioning (70% Train, 15% Validation, 15% Test) preventing data leakage.
- Independent sequence tokenization and padding (fitted ONLY on the training split).
- Serialization of processed splits, tokenizer artifacts (.pkl), and preprocessing configuration (.json).
"""

from collections import Counter
import csv
import html
import json
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw SMS dataset supporting UCI tab-separated and Kaggle CSV formats.

    Preserves the original dataset completely without modification.
    Uses csv.QUOTE_NONE for tab-separated data so unescaped quotation marks in SMS texts
    do not cause multi-line record merging.

    Args:
        file_path: Path to raw dataset file (e.g. SMSSpamCollection, spam.csv).

    Returns:
        Raw pandas DataFrame as read from disk.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")

    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                first_line = f.readline()
                has_tab = "\t" in first_line

            if has_tab:
                # Tab-separated file: Use csv.QUOTE_NONE to preserve all 5,574 original lines
                df = pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    names=["label", "message"],
                    encoding=enc,
                    quoting=csv.QUOTE_NONE,
                )
                return df
            else:
                # Standard CSV file
                df = pd.read_csv(path, encoding=enc)
                return df
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    # Fallback with latin-1 and QUOTE_NONE
    return pd.read_csv(path, sep="\t", header=None, names=["label", "message"], quoting=csv.QUOTE_NONE, encoding="latin-1")


def normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Identify and normalize message and label columns into a separate working DataFrame.

    Does not modify the input raw DataFrame.
    Converts labels: 'ham' -> 0, 'spam' -> 1.

    Args:
        raw_df: Original un-normalized DataFrame.

    Returns:
        Normalized DataFrame with columns ['message', 'label', 'label_encoded'].
    """
    df = raw_df.copy()

    # Identify label and message columns
    cols = [str(c).lower().strip() for c in df.columns]

    label_col = None
    message_col = None

    for idx, c in enumerate(cols):
        orig_col = df.columns[idx]
        if c in ["label", "v1", "category", "target", "class"]:
            label_col = orig_col
        elif c in ["message", "v2", "text", "sms", "content", "body"]:
            message_col = orig_col

    # Fallback by column position if not matched by name
    if label_col is None and len(df.columns) >= 1:
        label_col = df.columns[0]
    if message_col is None and len(df.columns) >= 2:
        message_col = df.columns[1]

    if label_col is None or message_col is None:
        raise ValueError(f"Could not identify message and label columns among: {df.columns.tolist()}")

    working_df = pd.DataFrame()
    working_df["message"] = df[message_col].astype(str)
    working_df["label"] = df[label_col].astype(str).str.lower().str.strip()

    # Normalize labels and encode: ham -> 0, spam -> 1
    label_map = {
        "ham": 0,
        "legitimate": 0,
        "0": 0,
        "spam": 1,
        "1": 1,
    }
    working_df["label_encoded"] = working_df["label"].map(label_map)

    # Validate that all labels were mapped successfully
    if working_df["label_encoded"].isna().any():
        unmapped = working_df[working_df["label_encoded"].isna()]["label"].unique()
        raise ValueError(f"Unrecognized labels found in dataset: {unmapped}")

    working_df["label_encoded"] = working_df["label_encoded"].astype(int)
    return working_df


def clean_sms_text(text: str) -> str:
    """Normalize SMS text carefully while preserving key spam signals.

    Preserves:
    - URLs -> normalized to atomic token '_url_'
    - Phone numbers & shortcodes -> normalized to '_phone_'
    - Currency symbols (£, $, €, ₹, rs, inr, etc.) -> normalized to '_currency_'
    - Standalone numbers & amounts -> normalized to '_num_'
    - Repeated punctuation (!!, ???) -> normalized to '_exclamation_', '_question_'
    - HTML entities -> unescaped (e.g., &amp; -> and, <#> -> _num_)
    - Code-mixed words / Hinglish terms are preserved in lowercase.

    Args:
        text: Raw SMS message string.

    Returns:
        Cleaned, normalized string ready for tokenization.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Decode HTML entities (e.g. &amp; -> &, &lt; -> <, &gt; -> >)
    text = html.unescape(text)

    # 2. Convert remaining standalone ampersands to 'and'
    text = re.sub(r"&", " and ", text)

    # 3. Dataset-specific masked numbers (e.g. <#>)
    text = re.sub(r"<#>", " _num_ ", text)

    # 3. Detect and replace URLs (http, https, www, domain names, bit.ly)
    url_pattern = r"(?:https?://\S+|www\.\S+|bit\.ly/\S+|\b\S+\.(?:com|co\.uk|org|net|wap|biz|info)\b(?:/\S*)?)"
    text = re.sub(url_pattern, " _url_ ", text, flags=re.IGNORECASE)

    # 4. Currency symbols and monetary references (£, $, €, ¥, ₹, rs, inr, gbp, usd)
    text = re.sub(r"[£$€¥₹]", " _currency_ ", text)
    text = re.sub(r"\b(?:rs|inr|gbp|usd|bucks)\b", " _currency_ ", text, flags=re.IGNORECASE)

    # 5. Phone numbers & shortcodes (5-6 digit shortcodes or 10-12 digit mobile numbers)
    phone_pattern = r"\b(?:\+?\d{1,3}[- ]?)?(?:0\d{9,11}|\d{10,12}|\b\d{5,6}\b)\b"
    text = re.sub(phone_pattern, " _phone_ ", text)

    # 6. Standalone numbers / integer / decimal figures
    text = re.sub(r"\b\d+(?:[\.,]\d+)?\b", " _num_ ", text)

    # 7. Repeated punctuation signals (urgency & spam hallmarks)
    text = re.sub(r"!{2,}", " _exclamation_ ", text)
    text = re.sub(r"\?{2,}", " _question_ ", text)

    # 8. Lowercase and remove remaining non-alphanumeric punctuation (except our underscore tokens)
    # Tokenize preserving alphanumeric words and _token_ formats
    tokens = re.findall(r"\b_\w+_\b|\b\w+\b", text.lower())

    return " ".join(tokens)


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into stratified train, validation, and test subsets.

    Args:
        df: Normalized working DataFrame containing 'label_encoded'.
        train_ratio: Proportion for training set (default: 0.70).
        val_ratio: Proportion for validation set (default: 0.15).
        test_ratio: Proportion for test set (default: 0.15).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df) with index reset.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, val, and test ratios must sum to 1.0")

    # Step 1: Stratified train / temp split
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df["label_encoded"],
    )

    # Step 2: Stratified val / test split from temp
    val_share_of_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        random_state=random_state,
        stratify=temp_df["label_encoded"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class SMSTokenizer:
    """Self-contained sequence tokenizer for SMS messages.

    Compatible with standard sequence deep learning models (Embedding -> LSTM).
    Guarantees strict zero-leakage training by fitting vocabulary only on training texts.
    Reserves index 0 for '<PAD>' and index 1 for '<OOV>'.
    """

    def __init__(
        self,
        max_vocab_size: int = 10000,
        oov_token: str = "<OOV>",
        pad_token: str = "<PAD>",
    ) -> None:
        """Initialize tokenizer with vocabulary parameters."""
        self.max_vocab_size = max_vocab_size
        self.oov_token = oov_token
        self.pad_token = pad_token

        # Reserved indices
        self.pad_index = 0
        self.oov_index = 1

        self.word_counts: Dict[str, int] = {}
        self.word_index: Dict[str, int] = {
            self.pad_token: self.pad_index,
            self.oov_token: self.oov_index,
        }
        self.index_word: Dict[int, str] = {
            self.pad_index: self.pad_token,
            self.oov_index: self.oov_token,
        }
        self.is_fitted: bool = False

    def tokenize_text(self, text: str) -> List[str]:
        """Split text into lowercase token words including special indicator tokens."""
        return re.findall(r"\b_\w+_\b|\b\w+\b", text.lower())

    def fit_on_texts(self, texts: List[str]) -> None:
        """Fit vocabulary ONLY on training messages.

        Args:
            texts: List of cleaned training message strings.
        """
        counts: Counter = Counter()
        for text in texts:
            counts.update(self.tokenize_text(text))

        self.word_counts = dict(counts)

        # Sort words by frequency (descending)
        # Cap vocabulary to (max_vocab_size - 2) to account for PAD and OOV
        allowed_vocab = self.max_vocab_size - 2
        most_common = counts.most_common(allowed_vocab)

        current_idx = 2
        for word, _ in most_common:
            self.word_index[word] = current_idx
            self.index_word[current_idx] = word
            current_idx += 1

        self.is_fitted = True

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of message strings into integer index sequences.

        Args:
            texts: List of cleaned message strings.

        Returns:
            List of integer sequences.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before transforming texts to sequences.")

        sequences: List[List[int]] = []
        for text in texts:
            tokens = self.tokenize_text(text)
            seq = [self.word_index.get(token, self.oov_index) for token in tokens]
            sequences.append(seq)
        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        maxlen: int,
        padding: str = "post",
        truncating: str = "post",
    ) -> np.ndarray:
        """Pad or truncate sequences to a fixed maximum length.

        Args:
            sequences: List of variable-length integer sequences.
            maxlen: Maximum sequence length.
            padding: 'post' (pad at end) or 'pre' (pad at start).
            truncating: 'post' (truncate end) or 'pre' (truncate start).

        Returns:
            2D numpy array of shape (num_samples, maxlen) of type int32.
        """
        num_samples = len(sequences)
        output = np.full((num_samples, maxlen), self.pad_index, dtype=np.int32)

        for i, seq in enumerate(sequences):
            if len(seq) == 0:
                continue

            # Truncating
            if len(seq) > maxlen:
                if truncating == "post":
                    trunc_seq = seq[:maxlen]
                else:
                    trunc_seq = seq[-maxlen:]
            else:
                trunc_seq = seq

            # Padding
            if padding == "post":
                output[i, : len(trunc_seq)] = trunc_seq
            else:
                output[i, -len(trunc_seq) :] = trunc_seq

        return output

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize tokenizer state dictionary to disk via pickle."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "max_vocab_size": self.max_vocab_size,
            "oov_token": self.oov_token,
            "pad_token": self.pad_token,
            "pad_index": self.pad_index,
            "oov_index": self.oov_index,
            "word_counts": self.word_counts,
            "word_index": self.word_index,
            "index_word": self.index_word,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SMSTokenizer":
        """Load serialized tokenizer state dictionary from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer artifact not found at {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        tokenizer = cls(
            max_vocab_size=state["max_vocab_size"],
            oov_token=state["oov_token"],
            pad_token=state["pad_token"],
        )
        tokenizer.pad_index = state["pad_index"]
        tokenizer.oov_index = state["oov_index"]
        tokenizer.word_counts = state["word_counts"]
        tokenizer.word_index = state["word_index"]
        tokenizer.index_word = state["index_word"]
        tokenizer.is_fitted = state["is_fitted"]
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including PAD and OOV."""
        return len(self.word_index)


def determine_max_sequence_length(
    train_cleaned_texts: List[str],
    tokenizer: SMSTokenizer,
    percentile: float = 99.5,
) -> int:
    """Determine optimal sequence length based on training length distribution.

    Args:
        train_cleaned_texts: Cleaned messages from training split only.
        tokenizer: Fitted tokenizer.
        percentile: Target coverage percentile (default: 99.5%).

    Returns:
        Selected sequence length (integer).
    """
    seqs = tokenizer.texts_to_sequences(train_cleaned_texts)
    lengths = [len(s) for s in seqs]
    calculated_len = int(np.ceil(np.percentile(lengths, percentile)))
    # Round up to a clean multiple of 5 or at least 70 to provide generous headroom
    recommended_len = max(calculated_len, 70)
    return recommended_len


def run_preprocessing_pipeline(
    raw_data_path: Union[str, Path] = "data/raw/SMSSpamCollection",
    processed_dir: Union[str, Path] = "data/processed",
    models_dir: Union[str, Path] = "models",
    max_vocab_size: int = 10000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end dataset preparation and preprocessing pipeline.

    1. Loads raw dataset using csv.QUOTE_NONE preserving all 5,574 lines.
    2. Normalizes to working DataFrame (ham=0, spam=1).
    3. Cleans messages preserving spam signals (URLs, phones, currency, numbers, repeated punctuation).
    4. Stratifies into 70% Train, 15% Validation, 15% Test.
    5. Fits SMSTokenizer ONLY on training set (zero data leakage).
    6. Determines sequence length from training distribution.
    7. Transforms and pads all three splits.
    8. Serializes processed datasets to data/processed/, tokenizer to models/tokenizer.pkl,
       and config to models/preprocessing_config.json.

    Returns:
        Dictionary summarizing execution metrics and shapes.
    """
    raw_path = Path(raw_data_path)
    proc_dir = Path(processed_dir)
    mod_dir = Path(models_dir)

    proc_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Raw Dataset
    raw_df = load_raw_dataset(raw_path)
    total_records = len(raw_df)

    # 2. Normalize
    working_df = normalize_dataframe(raw_df)

    # 3. Clean text
    working_df["cleaned_message"] = working_df["message"].apply(clean_sms_text)

    # 4. Stratified Split (70 / 15 / 15)
    train_df, val_df, test_df = split_dataset(
        working_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=random_state,
    )

    # 5. Fit Tokenizer ONLY on training messages
    tokenizer = SMSTokenizer(max_vocab_size=max_vocab_size)
    tokenizer.fit_on_texts(train_df["cleaned_message"].tolist())

    # 6. Determine Max Sequence Length from Train distribution
    max_seq_len = determine_max_sequence_length(
        train_df["cleaned_message"].tolist(),
        tokenizer,
        percentile=99.5,
    )

    # 7. Convert to padded sequences
    train_seqs = tokenizer.texts_to_sequences(train_df["cleaned_message"].tolist())
    val_seqs = tokenizer.texts_to_sequences(val_df["cleaned_message"].tolist())
    test_seqs = tokenizer.texts_to_sequences(test_df["cleaned_message"].tolist())

    X_train = tokenizer.pad_sequences(train_seqs, maxlen=max_seq_len, padding="post")
    X_val = tokenizer.pad_sequences(val_seqs, maxlen=max_seq_len, padding="post")
    X_test = tokenizer.pad_sequences(test_seqs, maxlen=max_seq_len, padding="post")

    y_train = train_df["label_encoded"].to_numpy(dtype=np.int32)
    y_val = val_df["label_encoded"].to_numpy(dtype=np.int32)
    y_test = test_df["label_encoded"].to_numpy(dtype=np.int32)

    # 8. Save Processed Datasets (CSV and NPY)
    train_df.to_csv(proc_dir / "train.csv", index=False)
    val_df.to_csv(proc_dir / "val.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    np.save(proc_dir / "X_train.npy", X_train)
    np.save(proc_dir / "y_train.npy", y_train)
    np.save(proc_dir / "X_val.npy", X_val)
    np.save(proc_dir / "y_val.npy", y_val)
    np.save(proc_dir / "X_test.npy", X_test)
    np.save(proc_dir / "y_test.npy", y_test)

    # 9. Save Tokenizer artifact (.pkl)
    tokenizer_path = mod_dir / "tokenizer.pkl"
    tokenizer.save(tokenizer_path)

    # 10. Save Preprocessing Config (.json)
    train_lengths = [len(s) for s in train_seqs]
    coverage_pct = float(np.mean([length <= max_seq_len for length in train_lengths]) * 100)

    config = {
        "dataset_name": "UCI SMS Spam Collection",
        "total_records": total_records,
        "max_vocab_size": max_vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "max_sequence_length": max_seq_len,
        "training_coverage_percentage": round(coverage_pct, 2),
        "padding_type": "post",
        "truncating_type": "post",
        "pad_token": "<PAD>",
        "oov_token": "<OOV>",
        "label_mapping": {"ham": 0, "spam": 1},
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_class_distribution": {
            "ham": int((train_df["label_encoded"] == 0).sum()),
            "spam": int((train_df["label_encoded"] == 1).sum()),
            "spam_pct": round(float((train_df["label_encoded"] == 1).mean() * 100), 2),
        },
        "val_class_distribution": {
            "ham": int((val_df["label_encoded"] == 0).sum()),
            "spam": int((val_df["label_encoded"] == 1).sum()),
            "spam_pct": round(float((val_df["label_encoded"] == 1).mean() * 100), 2),
        },
        "test_class_distribution": {
            "ham": int((test_df["label_encoded"] == 0).sum()),
            "spam": int((test_df["label_encoded"] == 1).sum()),
            "spam_pct": round(float((test_df["label_encoded"] == 1).mean() * 100), 2),
        },
        "preserved_special_tokens": [
            "_url_",
            "_phone_",
            "_currency_",
            "_num_",
            "_exclamation_",
            "_question_",
        ],
    }

    config_path = mod_dir / "preprocessing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "config": config,
        "shapes": {
            "X_train": X_train.shape,
            "y_train": y_train.shape,
            "X_val": X_val.shape,
            "y_val": y_val.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
        },
        "artifacts": {
            "tokenizer": str(tokenizer_path),
            "config": str(config_path),
            "train_csv": str(proc_dir / "train.csv"),
            "val_csv": str(proc_dir / "val.csv"),
            "test_csv": str(proc_dir / "test.csv"),
        },
    }


if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    print("Preprocessing pipeline completed successfully!")
    print(json.dumps(results, indent=2))
