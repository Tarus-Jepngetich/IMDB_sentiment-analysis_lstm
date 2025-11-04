# 🎬 IMDB Sentiment Analysis with LSTM

This project uses a **Long Short-Term Memory (LSTM)** neural network to classify IMDB movie reviews as **positive** or **negative**. It was developed and trained in Google Colab using **TensorFlow** and **Keras**.

---

## 🧩 Overview
The goal of this project is to build a deep learning model that can understand the sentiment of a movie review by learning word patterns and context.  
The model uses an embedding layer, an LSTM layer, and a dense output layer with a sigmoid activation function for binary classification.

---

## 📂 Dataset
- **Dataset:** IMDB Movie Reviews (available in Keras datasets)
- **Training size:** 25,000 reviews  
- **Testing size:** 25,000 reviews  
- **Classes:** Positive (1), Negative (0)

---

## ⚙️ Model Architecture
- Embedding layer (vocab size = 10,000, embedding dim = 128)  
- LSTM layer (64 units, dropout = 0.2)  
- Dense output layer (1 neuron, sigmoid activation)  
- Optimizer: Adam (learning rate = 0.001)  
- Loss function: Binary Crossentropy  
- Metric: Accuracy  

---

## 🚀 Training
The model was trained for 10 epochs with early stopping and model checkpointing to prevent overfitting.

Example training code:
```python
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=2
)
