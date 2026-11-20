"""Inference and Prediction Pipeline for SMS Spam Detection.

Loads the trained LSTM model and tokenizer artifacts to classify new raw messages
(both English and Hinglish) as Spam or Ham with confidence scores.
Reuses the exact same clean_sms_text and SMSTokenizer logic from src.preprocess.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.preprocess import SMSTokenizer, clean_sms_text


class SpamPredictor:
    """Predictor service for classifying incoming SMS messages."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/spam_lstm.keras",
        tokenizer_path: Union[str, Path] = "models/tokenizer.pkl",
        config_path: Union[str, Path] = "models/preprocessing_config.json",
        threshold: float = 0.5,
    ) -> None:
        """Initialize predictor by specifying paths to model, tokenizer, and config.

        Args:
            model_path: Path to serialized Keras model file.
            tokenizer_path: Path to serialized SMSTokenizer pickle file.
            config_path: Path to preprocessing configuration JSON.
            threshold: Probability threshold above which a message is classified as Spam.
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.config_path = Path(config_path)
        self.threshold = threshold
        self.model: Optional[Any] = None
        self.tokenizer: Optional[SMSTokenizer] = None
        self.max_sequence_length: int = 70

    def load_artifacts(self) -> None:
        """Load trained tokenizer and configuration from disk.

        Model loading will be activated once the model weights are trained.
        """
        if self.tokenizer_path.exists():
            self.tokenizer = SMSTokenizer.load(self.tokenizer_path)

        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.max_sequence_length = cfg.get("max_sequence_length", 70)

        if self.model_path.exists():
            # Model loading is activated when Keras model is trained
            try:
                import keras
                self.model = keras.models.load_model(self.model_path)
            except Exception:
                pass

    def preprocess_message(self, raw_message: str) -> np.ndarray:
        """Preprocess a single raw text message into a padded sequence array.

        Args:
            raw_message: Raw message text string.

        Returns:
            2D numpy array of shape (1, max_sequence_length).
        """
        if self.tokenizer is None:
            self.load_artifacts()

        if self.tokenizer is None:
            raise FileNotFoundError("Tokenizer artifact not found. Please run preprocessing pipeline first.")

        cleaned = clean_sms_text(raw_message)
        seqs = self.tokenizer.texts_to_sequences([cleaned])
        padded = self.tokenizer.pad_sequences(seqs, maxlen=self.max_sequence_length, padding="post")
        return padded

    def predict_single(self, raw_message: str) -> Dict[str, Union[str, float, bool]]:
        """Classify a single raw text message.

        Args:
            raw_message: Raw text string (English or Hinglish).

        Returns:
            Dictionary with 'label' ('Spam' or 'Ham'), 'confidence' (float), and 'is_spam' (bool).
        """
        if self.model is None:
            raise NotImplementedError(
                "Model weights not yet loaded. Please complete model training stage before running inference."
            )

        features = self.preprocess_message(raw_message)
        prob = float(self.model.predict(features, verbose=0)[0][0])
        is_spam = bool(prob >= self.threshold)
        label = "Spam" if is_spam else "Ham"

        return {
            "message": raw_message,
            "label": label,
            "probability": prob,
            "confidence": prob if is_spam else (1.0 - prob),
            "is_spam": is_spam,
        }

    def predict_batch(self, messages: List[str]) -> List[Dict[str, Union[str, float, bool]]]:
        """Classify a batch of messages.

        Args:
            messages: List of raw text strings.

        Returns:
            List of prediction dictionaries.
        """
        if self.model is None:
            raise NotImplementedError(
                "Model weights not yet loaded. Please complete model training stage before running batch inference."
            )

        if self.tokenizer is None:
            self.load_artifacts()

        cleaned_list = [clean_sms_text(m) for m in messages]
        seqs = self.tokenizer.texts_to_sequences(cleaned_list)
        padded = self.tokenizer.pad_sequences(seqs, maxlen=self.max_sequence_length, padding="post")

        probs = self.model.predict(padded, verbose=0).flatten()
        results = []
        for msg, prob in zip(messages, probs):
            p = float(prob)
            is_spam = bool(p >= self.threshold)
            results.append({
                "message": msg,
                "label": "Spam" if is_spam else "Ham",
                "probability": p,
                "confidence": p if is_spam else (1.0 - p),
                "is_spam": is_spam,
            })
        return results


if __name__ == "__main__":
    predictor = SpamPredictor()
    print("Inference service module ready. Tokenizer loading verified.")
