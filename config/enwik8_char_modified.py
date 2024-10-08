import math

# Configuration for the modified model
out_dir = 'out-enwik8-char-modified'  # Output directory for model checkpoints and logs
eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = True  # Changed to True to ensure we save checkpoints
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-modified'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64  # Adjust based on your GPU memory
block_size = 256  # Context length

# Model parameters
n_layer = 10
n_head = 8
n_embd = 512
dropout = 0.1  # Added some dropout for regularization
bias = False  # No bias in LayerNorm and Linear layers

# Optimization parameters
learning_rate = 1e-3
max_iters = 5000  # Increased number of iterations for better training
lr_decay_iters = 5000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
init_from = 'scratch'  # Initialize model from scratch

# Use the modified model
model_type = 'modified'

# System parameters
device = 'cuda'  # Use CUDA for training
dtype = 'float16'  # Use float16 for faster training
compile = False  # Disable compilation for now
