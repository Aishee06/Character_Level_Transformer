import os
import pickle
import numpy as np

# Define the data directory
data_dir = 'data/enwik8'

# Read the text files
with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
    train_data = f.read()
with open(os.path.join(data_dir, 'valid.txt'), 'r', encoding='utf-8') as f:
    val_data = f.read()
with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as f:
    test_data = f.read()

# Get all unique characters from the training set
chars = sorted(list(set(train_data)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")

# Create mappings from characters to integers and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# Save the mappings for later use
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# Encode the data and convert to numpy arrays
def encode(s):
    # Only encode characters that exist in the stoi dictionary
    return [stoi[c] for c in s if c in stoi]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
test_ids = np.array(encode(test_data), dtype=np.uint16)

# Save the encoded data to binary files
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
test_ids.tofile(os.path.join(data_dir, 'test.bin'))

print("Data preparation complete.")