"""Model Development and Training Module for SMS Spam Detection.

Builds, compiles, and trains an LSTM neural network using Keras/TensorFlow.
Reuses preprocessed sequence matrices, vocabulary configuration, and tokenizer artifacts.
Applies:
- Embedding layer (64 dimensions)
- LSTM layer (64 units)
- Dense output layer (1 neuron with sigmoid activation)
- Adam optimizer & Binary Cross-Entropy loss
- Balanced class weights computed exclusively from training labels
- EarlyStopping based on validation loss with best weights restored
- Checkpointing to models/spam_lstm.keras
"""

import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import callbacks, layers, metrics, optimizers
import tensorflow as tf


def set_reproducible_seeds(seed: int = 42) -> None:
    """Set global random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    input_length: int = 70,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build and compile the baseline LSTM network for SMS spam classification.

    Architecture:
    - Input: (input_length,) integer sequence
    - Embedding: input_dim=vocab_size, output_dim=embedding_dim (64)
    - LSTM: lstm_units (64)
    - Dense: 1 neuron, sigmoid activation

    Args:
        vocab_size: Size of vocabulary (including PAD and OOV).
        embedding_dim: Dimensionality of token embedding vectors (default: 64).
        lstm_units: Number of LSTM hidden memory cells (default: 64).
        input_length: Fixed sequence length of inputs (default: 70).
        learning_rate: Adam optimizer learning rate (default: 0.001).

    Returns:
        Compiled Keras Sequential Model.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(input_length,), dtype="int32", name="input_sequences"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=False,
                name="embedding_layer",
            ),
            layers.LSTM(lstm_units, name="lstm_layer"),
            layers.Dense(1, activation="sigmoid", name="output_dense"),
        ],
        name="sms_spam_lstm",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )
    return model


def calculate_training_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Calculate balanced class weights from training labels only.

    Args:
        y_train: 1D numpy array of binary training labels (0 and 1).

    Returns:
        Dictionary mapping class index (0, 1) to weight float.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    return class_weight_dict


def plot_and_save_training_curves(
    history_dict: Dict[str, list],
    output_path: Union[str, Path] = "outputs/training_curves.png",
) -> None:
    """Plot and save training vs. validation loss and accuracy curves.

    Args:
        history_dict: Dictionary containing loss, val_loss, accuracy, val_accuracy lists.
        output_path: Destination path for saved figure.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history_dict["loss"], "o-", color="#2b5c8f", label="Training Loss", linewidth=1.8)
    ax1.plot(epochs, history_dict["val_loss"], "s--", color="#d9534f", label="Validation Loss", linewidth=1.8)
    best_loss_epoch = int(np.argmin(history_dict["val_loss"]) + 1)
    best_loss_val = min(history_dict["val_loss"])
    ax1.axvline(best_loss_epoch, color="gray", linestyle=":", label=f"Best Val Loss (Ep {best_loss_epoch})")
    ax1.set_title("Training vs. Validation Loss (Binary Cross-Entropy)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right")

    # Accuracy plot
    ax2.plot(epochs, history_dict["accuracy"], "o-", color="#2b5c8f", label="Training Accuracy", linewidth=1.8)
    ax2.plot(epochs, history_dict["val_accuracy"], "s--", color="#28a745", label="Validation Accuracy", linewidth=1.8)
    best_acc_epoch = int(np.argmax(history_dict["val_accuracy"]) + 1)
    ax2.axvline(best_acc_epoch, color="gray", linestyle=":", label=f"Best Val Acc (Ep {best_acc_epoch})")
    ax2.set_title("Training vs. Validation Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def train(
    data_dir: Union[str, Path] = "data/processed",
    models_dir: Union[str, Path] = "models",
    outputs_dir: Union[str, Path] = "outputs",
    embedding_dim: int = 64,
    lstm_units: int = 64,
    batch_size: int = 32,
    epochs: int = 15,
    patience: int = 4,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute the complete LSTM model training, validation, and serialization workflow.

    Args:
        data_dir: Directory containing preprocessed .npy files.
        models_dir: Directory where model artifacts and configs are stored.
        outputs_dir: Directory for training metrics and plots.
        embedding_dim: Embedding dimension (64).
        lstm_units: LSTM units (64).
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience epochs.
        random_seed: Random seed for reproducibility.

    Returns:
        Dictionary summarizing final metrics and saved artifact locations.
    """
    set_reproducible_seeds(random_seed)

    data_path = Path(data_dir)
    mod_path = Path(models_dir)
    out_path = Path(outputs_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessing Configuration
    config_file = mod_path / "preprocessing_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file {config_file} not found. Run 'python -m src.preprocess' first."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    vocab_size = config["actual_vocab_size"]
    max_seq_len = config["max_sequence_length"]

    # 2. Load Processed Training and Validation Sequences (ONLY train and val, NEVER test)
    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    y_val = np.load(data_path / "y_val.npy")

    print("==================================================")
    print("           SMS SPAM LSTM MODEL TRAINING           ")
    print("==================================================")
    print(f"Vocabulary Size:          {vocab_size}")
    print(f"Max Sequence Length:      {max_seq_len}")
    print(f"Embedding Dimensions:     {embedding_dim}")
    print(f"LSTM Units:               {lstm_units}")
    print(f"Batch Size:               {batch_size}")
    print(f"Max Epochs:               {epochs}")
    print(f"Early Stopping Patience:  {patience}")
    print(f"Training Samples:         {X_train.shape[0]} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f"Validation Samples:       {X_val.shape[0]} (Ham: {(y_val == 0).sum()}, Spam: {(y_val == 1).sum()})")

    # 3. Compute Class Weights from Training Labels
    class_weights = calculate_training_class_weights(y_train)
    print(f"Computed Class Weights:   Ham (0): {class_weights[0]:.4f}, Spam (1): {class_weights[1]:.4f}")

    # 4. Build Model
    model = build_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        input_length=max_seq_len,
    )
    model.summary()

    # 5. Configure Callbacks
    model_save_path = mod_path / "spam_lstm.keras"
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # 6. Train Model (Using ONLY Training data; Validation data ONLY for monitoring)
    print("\nStarting model training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint],
        verbose=1,
    )

    # 7. Ensure best model is saved
    model.save(model_save_path)
    print(f"\nSaved trained model to: {model_save_path}")

    # 8. Verify Saved Model Can Be Loaded and Run
    print("Verifying saved model loading and inference...")
    loaded_model = keras.models.load_model(model_save_path)
    sample_pred = loaded_model.predict(X_val[:5], verbose=0)
    print(f"Verification successful! Sample predictions on 5 validation items: {sample_pred.flatten()[:5]}")

    # 9. Evaluate on Validation Set
    val_results = model.evaluate(X_val, y_val, verbose=0, return_dict=True)
    print("\n--- FINAL VALIDATION SET PERFORMANCE ---")
    for metric_name, val_val in val_results.items():
        print(f"  {metric_name:15}: {val_val:.4f}")

    # 10. Save Training History
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_path = out_path / "training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Saved training history to: {hist_path}")

    # 11. Generate and Save Loss/Accuracy Visualizations
    curve_path = out_path / "training_curves.png"
    plot_and_save_training_curves(hist_dict, output_path=curve_path)
    print(f"Saved training curves to: {curve_path}")

    return {
        "model_path": str(model_save_path),
        "history_path": str(hist_path),
        "curves_path": str(curve_path),
        "val_metrics": val_results,
        "epochs_trained": len(hist_dict["loss"]),
    }


if __name__ == "__main__":
    train()
