# Word2Vec from Scratch (CBOW + Negative Sampling)

This project implements **Word2Vec (CBOW with Negative Sampling)** entirely from scratch using NumPy — without deep learning frameworks.

The goal of this project is to deeply understand the mathematics behind word embeddings, including forward pass computation, loss formulation, and full gradient derivation.

---

## 📌 Features

- Text preprocessing and tokenization  
- Vocabulary construction and indexing  
- Negative sampling distribution (0.75 power as in the original paper)  
- CBOW forward pass implementation  
- Manual loss computation  
- Full gradient derivation and parameter updates  
- Training loop with embedding optimization  
- Similarity evaluation using cosine similarity  

---

## 🧠 Mathematical Foundation

The model optimizes:

\[
L =
- \log \sigma(u_o^\top h)
- \sum_{k=1}^{K} \log \sigma(-u_k^\top h)
\]

Where:
- \( h \) is the averaged context embedding  
- \( u_o \) is the target word output embedding  
- \( u_k \) are negative sample embeddings  

Gradients are derived manually and applied using gradient descent.

---

## 📊 Example Output

Below is an example output produced by the training script:

![Training Output](path_to_your_image.png)

> Replace `path_to_your_image.png` with your actual image file path.

---

## 🚀 How to Run

```bash
python train.py
