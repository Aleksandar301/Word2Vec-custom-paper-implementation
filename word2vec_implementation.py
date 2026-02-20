"""
Word2Vec Implementation in Pure NumPy
======================================

Implements CBOW (Continuous Bag of Words) with Negative Sampling using only NumPy.
No external ML frameworks (PyTorch, TensorFlow) are used.

Dataset: Alice's Adventures in Wonderland (Project Gutenberg)
"""
import re
from collections import Counter
import numpy as np

# DATA LOADING AND PREPROCESSING

def load_and_clean_text(filepath):
    """
    Load text from file and remove Gutenberg header/footer.
    
    Args:
        filepath (str): Path to the text file
        
    Returns:
        str: Cleaned and lowercase text
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Remove Gutenberg header and footer
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of word tokens
    """
    return text.split()


# VOCABULARY BUILDING

def build_vocab(tokens, min_count=2):
    """
    Build vocabulary from tokens, removing rare words.
    
    Args:
        tokens (list): List of word tokens
        min_count (int): Minimum frequency threshold for words
        
    Returns:
        tuple: (word_to_idx dict, idx_to_word dict)
    """
    word_counts = Counter(tokens)

    # Remove rare words
    vocab_words = [word for word, count in word_counts.items() if count >= min_count]

    word_to_idx = {word: idx for idx, word in enumerate(vocab_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word


def tokens_to_indices(tokens, word_to_idx):
    """
    Convert word tokens to their indices.
    
    Args:
        tokens (list): List of word tokens
        word_to_idx (dict): Word to index mapping
        
    Returns:
        list: List of word indices
    """
    return [word_to_idx[word] for word in tokens if word in word_to_idx]


def build_negative_sampling_dist(tokens, word_to_idx):
    """
    Builds negative sampling distribution.
    
    Applies 0.75 power to word frequencies as per original Word2Vec paper.
    
    Args:
        tokens (list): List of word tokens
        word_to_idx (dict): Word to index mapping
        
    Returns:
        np.ndarray: Probability distribution for negative sampling
    """
    word_counts = Counter(tokens)

    vocab_size = len(word_to_idx)
    freqs = np.zeros(vocab_size)

    for word, idx in word_to_idx.items():
        freqs[idx] = word_counts[word]

    # Apply 3/4 power (standard in Word2Vec)
    freqs = freqs ** 0.75
    freqs /= np.sum(freqs)

    return freqs


def preprocess(filepath, min_count=2):
    """
    Full preprocessing pipeline: load, tokenize, build vocab, convert to indices.
    
    Args:
        filepath (str): Path to text file
        min_count (int): Minimum word frequency
        
    Returns:
        tuple: (corpus_indices, word_to_idx, idx_to_word, neg_sampling_dist)
    """
    text = load_and_clean_text(filepath)
    tokens = tokenize(text)

    word_to_idx, idx_to_word = build_vocab(tokens, min_count)
    corpus_indices = tokens_to_indices(tokens, word_to_idx)

    neg_sampling_dist = build_negative_sampling_dist(tokens, word_to_idx)

    return corpus_indices, word_to_idx, idx_to_word, neg_sampling_dist

# TRAINING DATA GENERATION

def generate_cbow_samples(corpus_indices, window_size=2):
    """
    Generate CBOW training samples from corpus.
    
    CBOW (Continuous Bag of Words) predicts a target word from its context.
    
    Args:
        corpus_indices (list): List of word indices
        window_size (int): Context window size (words on each side)
        
    Returns:
        list: List of (context_indices, target_index) tuples
        
    Example:
        For corpus [1, 2, 3, 4, 5] with window_size=2:
        - context [1, 2, 4, 5] predicts target 3
    """
    samples = []
    T = len(corpus_indices)

    # Only process positions with enough context on both sides
    for t in range(window_size, T - window_size):
        
        # Target word w_O
        target = corpus_indices[t]

        # Context words (left and right)
        context = []

        # Left context
        for i in range(t - window_size, t):
            context.append(corpus_indices[i])

        # Right context
        for i in range(t + 1, t + window_size + 1):
            context.append(corpus_indices[i])

        samples.append((context, target))

    return samples

# WORD2VEC TRAINING (CBOW WITH NEGATIVE SAMPLING)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2 (np.ndarray): Input vectors
        
    Returns:
        float: Cosine similarity in range [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def train_cbow(training_samples, W, W_prime, neg_dist, 
               window_size=2, N=50, K=5, lr=0.025, num_epochs=10):
    """
    Train CBOW model with Negative Sampling.
    
    Algorithm:
    ----------
    For each training sample (context, target):
        1. Forward pass: average context embeddings, compute scores
        2. Loss: log sigmoid for positive pair + sum of log sigmoid for negative pairs
        3. Backward pass: compute gradients via chain rule
        4. Update: W and W_prime using gradient descent
    
    Args:
        training_samples (list): List of (context_indices, target_index) tuples
        W (np.ndarray): Input embedding matrix shape (V, N)
        W_prime (np.ndarray): Output embedding matrix shape (V, N)
        neg_dist (np.ndarray): Negative sampling distribution length V
        window_size (int): Context window size
        N (int): Embedding dimension
        K (int): Number of negative samples per training example
        lr (float): Learning rate
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (W, W_prime) Updated embedding matrices
    """
    
    V = W.shape[0]
    
    def sample_negative_words(K, neg_dist):
        """Sample K negative words according to neg_dist."""
        return np.random.choice(len(neg_dist), size=K, p=neg_dist)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for context_indices, target in training_samples:
            
            # FORWARD PASS 
            # Get context embeddings and average them
            v_context = W[context_indices]                 # (C, N) where C = window_size * 2
            h = np.mean(v_context, axis=0)                # (N,) - hidden representation
            
            # Get output embeddings
            u_target = W_prime[target]                     # (N,)
            neg_indices = sample_negative_words(K, neg_dist)
            u_neg = W_prime[neg_indices]                  # (K, N)
            
            # Compute scores
            s_pos = np.dot(u_target, h)                   # scalar
            s_neg = np.dot(u_neg, h)                      # (K,)
            
            # Apply sigmoid
            sig_pos = sigmoid(s_pos)
            sig_neg = sigmoid(-s_neg)
            
            # LOSS COMPUTATION 
            # L = -log(σ(u_target · h)) - Σ log(σ(-u_neg · h))
            loss = -np.log(sig_pos + 1e-10) - np.sum(np.log(sig_neg + 1e-10))
            total_loss += loss
            
            # BACKWARD PASS
            # Gradients of sigmoid-based loss
            delta_pos = sig_pos - 1                        # scalar
            delta_neg = 1 - sig_neg                        # (K,)
            
            # Gradients for output embeddings
            grad_u_target = delta_pos * h                  # (N,)
            grad_u_neg = delta_neg[:, None] * h           # (K, N)
            
            # Gradient wrt hidden layer (sum of gradients from positive and negative pairs)
            grad_h = delta_pos * u_target + np.dot(delta_neg, u_neg)  # (N,)
            
            # Distribute gradient to context embeddings (average because h is averaged)
            grad_v_context = grad_h / len(context_indices)
            
            # PARAMETER UPDATES
            # Update output embeddings
            W_prime[target] -= lr * grad_u_target
            W_prime[neg_indices] -= lr * grad_u_neg
            
            # Update input embeddings (context words)
            for idx in context_indices:
                W[idx] -= lr * grad_v_context
        
        avg_loss = total_loss / len(training_samples)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")
    
    return W, W_prime

# MAIN EXECUTION

if __name__ == "__main__":
    
    print("=" * 80)
    print("WORD2VEC IMPLEMENTATION IN PURE NUMPY")
    print("CBOW with Negative Sampling")
    print("=" * 80)
    
    # PREPROCESSING 
    print("\n[1/5] Loading and preprocessing dataset...")
    corpus_indices, word_to_idx, idx_to_word, neg_dist = preprocess("alice.txt", min_count=2)
    
    print(f"       Corpus length (tokens): {len(corpus_indices):,}")
    print(f"       Vocabulary size: {len(word_to_idx):,}")
    
    # TRAINING DATA GENERATION 
    print("\n[2/5] Generating CBOW training samples...")
    window_size = 2
    training_samples = generate_cbow_samples(corpus_indices, window_size)
    
    print(f"       Training samples: {len(training_samples):,}")
    print(f"       Example: {training_samples[0]}")
    print(f"       (context indices: {training_samples[0][0]}, target: {training_samples[0][1]})")
    
    # Convert example to words for visualization
    example_context = [idx_to_word[i] for i in training_samples[0][0]]
    example_target = idx_to_word[training_samples[0][1]]
    print(f"       Decoded: context={example_context}, target={example_target}")
    
    # PARAMETER INITIALIZATION 
    print("\n[3/5] Initializing embedding matrices...")
    N = 50  # Embedding dimension
    V = len(word_to_idx)
    
    # Initialize embeddings with small random values
    W = np.random.randn(V, N) * 0.01  # Input embeddings
    W_prime = np.random.randn(V, N) * 0.01  # Output embeddings
    
    print(f"       Input embedding matrix W shape: {W.shape}")
    print(f"       Output embedding matrix W' shape: {W_prime.shape}")
    
    # TRAINING 
    print("\n[4/5] Training CBOW model...")
    
    # Hyperparameters
    K = 5  # Number of negative samples
    lr = 0.025  # Learning rate
    num_epochs = 15
    
    W, W_prime = train_cbow(
        training_samples, 
        W, W_prime, 
        neg_dist,
        window_size=window_size,
        N=N,
        K=K,
        lr=lr,
        num_epochs=num_epochs
    )
    
    # EVALUATION
    print("\n[5/5] Computing word similarities...")
    
    # Find indices of interesting words
    target_words = ['alice', 'queen', 'rabbit', 'wonderland', 'time', 'think', 'said']
    valid_words = [w for w in target_words if w in word_to_idx]
    
    if valid_words:
        print(f"\n       Word similarities (using input embeddings W):\n")
        
        for i, word1 in enumerate(valid_words):
            for word2 in valid_words[i+1:]:
                idx1 = word_to_idx[word1]
                idx2 = word_to_idx[word2]
                
                sim = cosine_similarity(W[idx1], W[idx2])
                print(f"       {word1:15s} <-> {word2:15s} : {sim:7.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
