# Emotion Classification with Bidirectional GRU

This repository demonstrates emotion classification on short text using recurrent neural networks (RNN, LSTM, GRU) and an advanced Bidirectional GRU model implemented with TensorFlow / Keras.

## Overview
- **Goal:** Classify text into six emotions: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`.
- **Dataset:** `dair-ai/emotion` (Hugging Face `datasets`).
- **Notebook:** All steps (EDA, preprocessing, model building, training, evaluation, and inference) are implemented in `Emotion_Classification.ipynb`.
- **Saved artifacts:** A trained BiGRU model and tokenizer are included as `BiGru_model.keras` and `tokenizer.pkl` respectively.

## Repository Structure
- `Emotion_Classification.ipynb` — Jupyter notebook with full pipeline.
- `BiGru_model.keras` — Saved Keras model (BiGRU).
- `tokenizer.pkl` — Saved `Tokenizer` used for preprocessing.
- `README.md` — This file.

## Requirements
Create a Python 3.8+ virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# or Command Prompt
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

## Minimal Dependencies
- `tensorflow` (2.x)
- `datasets` (Hugging Face)
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `jupyter` (to run the notebook)

See `requirements.txt` for a pinned list.

## Quick Usage
1. Open and run `Emotion_Classification.ipynb` in Jupyter or VS Code. The notebook loads the dataset from Hugging Face, trains models, compares them, and saves the best BiGRU model.

2. To load the saved model and tokenizer for inference:

```python
from tensorflow import keras
import pickle

model = keras.models.load_model('BiGru_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Example preprocessing (matching the notebook):
from tensorflow.keras.preprocessing.sequence import pad_sequences
sample_text = ["I loved the movie!"]
seq = tokenizer.texts_to_sequences(sample_text)
seq = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
pred = model.predict(seq)
label = pred.argmax(axis=1)
print(label)
```

## Notes & Reproducibility
- The notebook uses balanced `class_weight` and `EarlyStopping` to mitigate imbalance and overfitting.
- The preprocessing uses a `Tokenizer` limited to `max_words=10000` and sequences padded/truncated to length `50` — match these if you retrain or reuse the tokenizer.

## Results
Open the notebook to view training/validation curves, a confusion matrix, and example predictions. The notebook computes and prints model comparisons for RNN, LSTM, GRU, Bidirectional GRU, and Bidirectional LSTM.

## Next Steps (suggested)
- Add a `requirements.txt` (provided) and optional `environment.yml` for Conda users.
- Add a short script `predict.py` for command-line inference.
- Package the model with a tiny Flask/FastAPI service for serving.

## License & Contact
Use as you like for experiments and learning. For questions, contact the repository owner.
