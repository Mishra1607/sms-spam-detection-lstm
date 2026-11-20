"""Unit Tests for SMS Spam Detection Preprocessing and Data Splitting Pipeline.

Tests:
- SMS text normalization and spam indicator preservation (_url_, _phone_, _currency_, _num_, _exclamation_)
- Label normalization and binary encoding (ham=0, spam=1)
- Stratified dataset partitioning (70% Train, 15% Validation, 15% Test) with zero index overlap
- Zero-leakage tokenization (fitting exclusively on training split, OOV mapping)
- Sequence padding dimensions and alignment
- Tokenizer artifact serialization and deserialization
"""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.preprocess import (
    SMSTokenizer,
    clean_sms_text,
    determine_max_sequence_length,
    load_raw_dataset,
    normalize_dataframe,
    split_dataset,
)


class TestTextNormalization(unittest.TestCase):
    """Test individual text normalization rules and spam indicator preservation."""

    def test_url_preservation(self):
        text = "Visit https://spam-promo.com or www.win-cash.co.uk now!"
        cleaned = clean_sms_text(text)
        self.assertIn("_url_", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_phone_and_shortcode_preservation(self):
        text = "Call 09061701461 or text CLAIM to 87121 to receive your prize."
        cleaned = clean_sms_text(text)
        self.assertIn("_phone_", cleaned)

    def test_currency_symbol_preservation(self):
        text = "You won £900 cash or $1000 gift card or 5000 rs bonus!"
        cleaned = clean_sms_text(text)
        self.assertIn("_currency_", cleaned)

    def test_numbers_preservation(self):
        text = "Only 3 chances left out of 100!"
        cleaned = clean_sms_text(text)
        self.assertIn("_num_", cleaned)

    def test_repeated_punctuation_preservation(self):
        text = "URGENT!! Is this true??? Reply ASAP"
        cleaned = clean_sms_text(text)
        self.assertIn("_exclamation_", cleaned)
        self.assertIn("_question_", cleaned)

    def test_html_unescaping(self):
        text = "Tom &amp; Jerry special with &lt;#&gt; points"
        cleaned = clean_sms_text(text)
        self.assertIn("and", cleaned)
        self.assertIn("_num_", cleaned)
        self.assertNotIn("&amp;", cleaned)
        self.assertNotIn("&lt;", cleaned)

    def test_hinglish_text_preservation(self):
        text = "Aapka account block ho gaya hai, turant call karein!"
        cleaned = clean_sms_text(text)
        self.assertIn("aapka", cleaned)
        self.assertIn("account", cleaned)
        self.assertIn("karein", cleaned)


class TestDataFrameNormalization(unittest.TestCase):
    """Test column normalization and label encoding."""

    def test_label_encoding(self):
        sample_df = pd.DataFrame({
            "v1": ["ham", "spam", "HAM", "Spam"],
            "v2": ["hello friend", "free cash call now", "ok coming", "claim prize"],
        })
        working = normalize_dataframe(sample_df)
        self.assertEqual(list(working["label_encoded"]), [0, 1, 0, 1])
        self.assertEqual(list(working["label"]), ["ham", "spam", "ham", "spam"])

    def test_invalid_label_rejection(self):
        bad_df = pd.DataFrame({
            "label": ["ham", "unknown_class"],
            "message": ["hello", "mystery"],
        })
        with self.assertRaises(ValueError):
            normalize_dataframe(bad_df)


class TestDataSplitting(unittest.TestCase):
    """Test stratified splitting and check for zero data leakage."""

    def setUp(self):
        # Create a synthetic balanced representation to test proportions
        np.random.seed(42)
        n = 1000
        labels = np.random.choice([0, 1], size=n, p=[0.86, 0.14])
        self.df = pd.DataFrame({
            "message": [f"message number {i}" for i in range(n)],
            "label": ["spam" if l == 1 else "ham" for l in labels],
            "label_encoded": labels,
        })

    def test_split_proportions(self):
        train, val, test = split_dataset(self.df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)
        self.assertEqual(len(train) + len(val) + len(test), 1000)

    def test_zero_index_leakage(self):
        train, val, test = split_dataset(self.df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
        train_msgs = set(train["message"])
        val_msgs = set(val["message"])
        test_msgs = set(test["message"])
        self.assertTrue(train_msgs.isdisjoint(val_msgs))
        self.assertTrue(train_msgs.isdisjoint(test_msgs))
        self.assertTrue(val_msgs.isdisjoint(test_msgs))


class TestSMSTokenizer(unittest.TestCase):
    """Test custom tokenizer fitting, sequence generation, padding, and persistence."""

    def test_fit_and_oov_handling(self):
        tokenizer = SMSTokenizer(max_vocab_size=100)
        train_texts = ["hello world prize winner", "free mobile call now"]
        tokenizer.fit_on_texts(train_texts)

        self.assertTrue(tokenizer.is_fitted)
        self.assertEqual(tokenizer.word_index["<PAD>"], 0)
        self.assertEqual(tokenizer.word_index["<OOV>"], 1)

        # Unseen word should map to OOV (1)
        test_seq = tokenizer.texts_to_sequences(["hello unseen word"])
        self.assertEqual(test_seq[0][0], tokenizer.word_index["hello"])
        self.assertEqual(test_seq[0][1], 1)  # unseen -> OOV
        self.assertEqual(test_seq[0][2], 1)  # word -> OOV

    def test_padding_shapes_and_values(self):
        tokenizer = SMSTokenizer(max_vocab_size=100)
        tokenizer.fit_on_texts(["one two three", "four five"])
        seqs = tokenizer.texts_to_sequences(["one two", "three four five six"])

        padded = tokenizer.pad_sequences(seqs, maxlen=4, padding="post", truncating="post")
        self.assertEqual(padded.shape, (2, 4))
        # First seq: 2 tokens + 2 pads (0)
        self.assertEqual(padded[0, 2], 0)
        self.assertEqual(padded[0, 3], 0)
        # Second seq: truncated to 4 tokens
        self.assertEqual(len(padded[1]), 4)

    def test_serialization(self):
        tokenizer = SMSTokenizer(max_vocab_size=50)
        tokenizer.fit_on_texts(["test serialization of tokenizer"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkl_path = Path(tmp_dir) / "test_tok.pkl"
            tokenizer.save(pkl_path)
            self.assertTrue(pkl_path.exists())

            loaded = SMSTokenizer.load(pkl_path)
            self.assertEqual(tokenizer.word_index, loaded.word_index)
            self.assertEqual(
                tokenizer.texts_to_sequences(["test serialization"]),
                loaded.texts_to_sequences(["test serialization"]),
            )


if __name__ == "__main__":
    unittest.main()
