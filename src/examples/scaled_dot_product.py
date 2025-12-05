from numpy import np
import matplotlib.pyplot as plt
import seaborn as sns
from core_functions.scaled_dot_product import scaled_dot_product_attention


# Define word embeddings
embeddings = {
    'the': np.array([0.1, 0.2, 0.3]),
    'cat': np.array([0.4, 0.5, 0.6]),
    'sat': np.array([0.7, 0.8, 0.9]),
    'on': np.array([1.0, 1.1, 1.2]),
    'mat': np.array([1.3, 1.4, 1.5])
}

# Define input sentence
sentence = ['the', 'cat', 'sat', 'on', 'the', 'mat']

# Convert sentence to embeddings
embedded_tokens = np.array([embeddings[word] for word in sentence])


# Q = K = V for self-attention
Q = K = V = np.copy(embedded_tokens)

# Apply self-attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)

# Print attention weights
print("Attention Weights:")
print(attention_weights)

# Print output
print("Output:")
print(output)

# Visualize attention weights
tokens = sentence
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens, cmap='viridis', annot=True)
plt.xlabel('Input Tokens')
plt.ylabel('Attention given to Tokens')
plt.title('Attention Weights Heatmap')
plt.show()