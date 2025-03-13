"""Creating token embeddings"""

import torch
from typing import List

inputs_ids: List[int] = torch.tensor([2, 3, 5, 1])

# Suppose there is only 6 words in lexical resource
vocab_size: int = 6

# Break out each word into 3 dimensional weights
output_dim: int = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)  # 6 rows, each row represents a word, and each row include 3 elements, meaning 3 dimensions

# Access one of these token
print(embedding_layer(torch.tensor([3])))

# Access tokens by the order of 2, 3, 5, 1
print(embedding_layer(inputs_ids))

