# SentimentScope 🎬  
**Sentiment Analysis using Transformers**

## 📌 Project Overview
SentimentScope is a transformer-based sentiment analysis project built using **PyTorch** and **Hugging Face tokenizers**.  
The goal of this project is to classify IMDB movie reviews as **positive** or **negative**, helping an entertainment company (CineScope) better understand user sentiment and improve its recommendation system.

This project demonstrates how a transformer model can be trained **from scratch** for a binary text classification task.

---

## 🎯 Objectives
By completing this project, we achieve the following:

- Load, explore, and preprocess the IMDB text dataset
- Implement a custom PyTorch `Dataset` and `DataLoader`
- Build a transformer architecture from scratch
- Adapt a transformer model for **binary classification**
- Train and evaluate the model on GPU
- Measure performance using **accuracy**

---

## 🧠 Dataset
**IMDB Movie Reviews Dataset**

- 25,000 training reviews
- 25,000 test reviews
- Binary labels:
  - `1` → Positive review
  - `0` → Negative review

## 🛠️ Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- BERT Tokenizer (`bert-base-uncased`)
- NumPy
- Pandas
- Matplotlib
- CUDA (GPU acceleration)

---

## 🔤 Tokenization
- Used **subword tokenization** via `AutoTokenizer`
- Model: `bert-base-uncased`
- Max sequence length: **128**
- Padding & truncation applied

---

## 🏗️ Model Architecture
Custom transformer architecture inspired by GPT-style blocks:

- Token Embeddings
- Positional Embeddings
- Multi-Head Self Attention
- Feed Forward Layers
- Layer Normalization
- Classification Head (2 output classes)

**Key Hyperparameters:**
- Embedding size: 128
- Transformer blocks: 4
- Attention heads: 4
- Dropout: 0.1

---

## 🚀 Training
- Loss Function: `CrossEntropyLoss`
- Optimizer: `AdamW`
- Learning Rate: `3e-4`
- Batch Size: `32`
- Epochs: `3`
- Device: GPU (CUDA)

---

## 📊 Results
- Validation Accuracy (initial): ~50%
- Test Accuracy: ~50%

> ⚠️ Since the model is trained from scratch (without pretrained weights), performance is limited.  
> Accuracy can be improved by:
> - Increasing epochs
> - Increasing embedding size
> - Adding more transformer blocks
> - Using pretrained transformer models

---

## ✅ Key Learnings
- How transformers process text for classification
- Importance of tokenization and padding
- Building attention mechanisms from scratch
- Training deep learning models on GPUs
- Evaluating classification models using accuracy

---

## 📌 Future Improvements
- Use pretrained BERT or DistilBERT
- Add learning rate scheduling
- Apply attention masking for padding tokens
- Improve pooling strategy (mean pooling / CLS token comparison)

---
