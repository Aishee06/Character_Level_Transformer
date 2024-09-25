from datasets import load_dataset

# Load the enwik8 dataset
dataset = load_dataset("LTCB/enwik8", split="train")

# Get the text data
text_data = ''.join(dataset['text'])

# Total number of characters
total_chars = len(text_data)
print(f"Total number of characters: {total_chars}")

# Set the number of characters for the training set
num_train_chars = 90_000_000

# Ensure we have enough characters for training
assert total_chars >= num_train_chars, "Not enough data for 90 million training characters."

# Calculate remaining characters for validation and testing
remaining_chars = total_chars - num_train_chars

# Allocate up to 5 million characters for validation
num_valid_chars = min(5_000_000, remaining_chars)

# Allocate remaining characters for testing
num_test_chars = remaining_chars - num_valid_chars

# Extract the splits
train_text = text_data[:num_train_chars]
valid_text = text_data[num_train_chars:num_train_chars + num_valid_chars]
test_text = text_data[num_train_chars + num_valid_chars:]

# Save to files
with open('train.txt', 'w') as f:
    f.write(train_text)
with open('valid.txt', 'w') as f:
    f.write(valid_text)
with open('test.txt', 'w') as f:
    f.write(test_text)

print("Data saved to train.txt, valid.txt, and test.txt")

# Verify sizes
print(f"Training characters: {len(train_text)}")
print(f"Validation characters: {len(valid_text)}")
print(f"Test characters: {len(test_text)}")