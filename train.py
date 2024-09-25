import argparse
import os
import time
import math
import sys
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from model import GPTConfig, GPT
from model_modified import ModifiedGPT  # Import ModifiedGPT

def get_serializable_config(config):
    return {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, type(None)))}

def main():
    # Parse arguments and load configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args = parser.parse_args()

    # Load the configuration file
    config_file = args.config
    config = {}
    with open(config_file, 'r') as f:
        exec(f.read(), config)

    # Check if 'out_dir' is specified in the config
    if 'out_dir' not in config:
        print("Error: 'out_dir' not specified in the configuration file.")
        sys.exit(1)

    # Create output directory
    if int(os.environ.get('RANK', -1)) == -1:
        os.makedirs(config['out_dir'], exist_ok=True)
        print(f"Output directory: {config['out_dir']}")

    # Initialize DistributedDataParallel (DDP) if needed
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1
        device = config['device']  # from config

    tokens_per_iter = (config['gradient_accumulation_steps'] * ddp_world_size *
                       config['batch_size'] * config['block_size'])
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    # Model initialization and random seed
    torch.manual_seed(1337 + int(time.time()))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype)

    # Data loading function
    data_dir = os.path.join('data', config['dataset'])

    def get_batch(split):
        data_path = os.path.join(data_dir, f'{split}.bin')
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
        x = torch.stack([torch.from_numpy(
            (data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(
            (data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # Model initialization
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None

    # Load vocab size from the metadata if available
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta.get('vocab_size', None)
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")
    else:
        print("meta.pkl not found, using default GPT-2 vocab size")

    # Ensure that the vocab size is set; default to GPT-2 vocab size if not available
    vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

    # Model arguments
    model_args = dict(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        bias=config['bias'],
        vocab_size=vocab_size,
        dropout=config['dropout'],
    )

    # Instantiate the model
    if config.get('model_type', 'gpt') == 'modified':
        gptconf = GPTConfig(**model_args)
        model = ModifiedGPT(gptconf)
        print("Using ModifiedGPT model.")
    else:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        print("Using GPT model.")

    model.to(device)

    # Initialize the optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    optimizer = model.configure_optimizers(
        config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)

    # Initialize iteration counters
    iter_num = 0
    best_val_loss = 1e9

    # Optionally resume from a checkpoint
    if config.get('init_from', 'scratch') == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from iteration {iter_num}, best val loss {best_val_loss}")

    # Print number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params/1e6:.2f}M")

    # Estimate loss function
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(config['eval_iters'])
            for k in range(config['eval_iters']):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # Learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        if it > config['lr_decay_iters']:
            return config['min_lr']
        decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    # Training loop with progress bar
    X, Y = get_batch('train')  # Fetch the very first batch
    running_mfu = -1.0
    t0 = time.time()

    # For logging purposes
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # Unwrap DDP container if needed

    with tqdm(total=config['max_iters'], desc="Training Progress") as pbar:
        while iter_num < config['max_iters']:
            # Set learning rate based on schedule
            lr = config['learning_rate'] if not config['decay_lr'] else get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward, backward, and update
            for micro_step in range(config['gradient_accumulation_steps']):
                if ddp:
                    model.require_backward_grad_sync = (
                        micro_step == config['gradient_accumulation_steps'] - 1)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / config['gradient_accumulation_steps']
                X, Y = get_batch('train')
                scaler.scale(loss).backward()

            if config['grad_clip'] != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Evaluation and logging
            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss()
                print(
                    f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': get_serializable_config(config),
                    }
                    checkpoint_path = os.path.join(config['out_dir'], 'ckpt.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to: {checkpoint_path}")

            if iter_num % config['log_interval'] == 0 and master_process:
                lossf = loss.item() * config['gradient_accumulation_steps']
                if local_iter_num > 0:
                    mfu = raw_model.estimate_mfu(
                        config['batch_size'] * config['gradient_accumulation_steps'], dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * \
                        running_mfu + 0.1 * mfu
                print(
                    f"Iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

            pbar.update(1)
            iter_num += 1
            local_iter_num += 1

            # Termination condition
            if iter_num >= config['max_iters']:
                break

    # Save final checkpoint
    if master_process:
        final_checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': get_serializable_config(config),
        }
        final_checkpoint_path = os.path.join(config['out_dir'], 'final_ckpt.pt')
        torch.save(final_checkpoint, final_checkpoint_path)
        print(f"Saved final checkpoint to: {final_checkpoint_path}")

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()