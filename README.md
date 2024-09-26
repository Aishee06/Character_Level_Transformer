# Character-Level Language Model using nanoGPT

This project trains and evaluates a character-level language model on the enwik8 dataset using [nanoGPT](https://github.com/karpathy/nanoGPT). We compare a baseline GPT model with a modified version (ModifiedGPT) that includes Multi-Scale Positional Encodings, combining both sinusoidal and learned positional embeddings.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Step-by-Step Instructions](#step-by-step-instructions)
  - [1. Clone the nanoGPT Repository](#1-clone-the-nanogpt-repository)
  - [2. Prepare the enwik8 Dataset](#2-prepare-the-enwik8-dataset)
  - [3. Preprocess the Data for nanoGPT](#3-preprocess-the-data-for-nanogpt)
  - [4. Set Up Configuration and Model Files](#4-set-up-configuration-and-model-files)
  - [5. Train the Modified Model](#5-train-the-modified-model)
  - [6. Train the Baseline Model](#6-train-the-baseline-model)
  - [7. Evaluate the Models](#7-evaluate-the-models)


## Installation

1. **Clone this repository** or navigate to your project directory.

2. **Ensure you have Python 3.x installed**.

3. **Install the required Python libraries**:

   ```sh
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   pip install transformers datasets tiktoken wandb tqdm numpy tabulate
   ```

   *Note: Adjust the CUDA version (`cu116`) in the `--extra-index-url` if necessary.*

## Project Overview

In this project, we:

- Train a baseline GPT model on the enwik8 dataset.
- Modify the GPT model to include additional sinusoidal and learned positional embeddings (ModifiedGPT).
- Train the modified model on the same dataset.
- Evaluate and compare the performance of both models.

## Step-by-Step Instructions

### 1. Clone the nanoGPT Repository

Clone the nanoGPT repository to obtain the necessary codebase:

```sh
git clone https://github.com/Aishee06/Character_Level_Transformer.git
```

Navigate to the `nanoGPT` directory:

```sh
cd Character_Level_Transformer
```

### 2. Prepare the enwik8 Dataset

Run the file:

```python
python enwik8_dataset_preparation.py
```

*This script loads the dataset, splits it into training, validation, and test sets, and saves them as text files.*

### 3. Preprocess the Data for nanoGPT

- **Create the Dataset Directory**:

  ```sh
  mkdir -p data/enwik8
  ```

- **Move the Data Files**:

  ```sh
  mv train.txt valid.txt test.txt data/enwik8/
  ```

- **Prepare the Data Using the Provided Script**:

  The `prepare_enwik8.py` script should already be present in the `data/` directory. If not, create it with the appropriate preprocessing steps.

  Run the script:

  ```sh
  python data/prepare_enwik8.py
  ```

*This script encodes the text files into token IDs and saves them as binary files for efficient loading during training.*

### 4. Set Up Configuration and Model Files

- **Created the Modified Configuration File**:

  Created `config/enwik8_char_modified.py` 


- **Created the Modified Model File**:

  Created `model_modified.py` with modifications to the GPT model.

*Note: The content of `model_modified.py` should define the `ModifiedGPT` class and any associated changes.*

### 5. Train the Modified Model

Run the training script with the modified configuration:

```sh
python train.py --config config/enwik8_char_modified.py
```

*This will train the modified GPT model on the enwik8 dataset. The model checkpoints and logs will be saved in the specified `out_dir`.*

### 6. Train the Baseline Model

- **Created the Baseline Configuration File**:

  Created a file named `config/enwik8_char_baseline.py`

Run the training script with the baseline configuration:

```sh
python train.py --config config/enwik8_char_baseline.py
```

*This will train the baseline GPT model on the enwik8 dataset. The model checkpoints and logs will be saved in the specified `out_dir`.*

### 7. Evaluate the Models

- **Evaluate the Baseline Model**:

  ```sh
  python evaluate.py --model_type gpt --checkpoint out-enwik8-char/ckpt.pt
  ```

- **Evaluate the Modified Model**:

  ```sh
  python evaluate.py --model_type modified --checkpoint out-enwik8-char-modified/ckpt.pt
  ```

*The evaluation script calculates the validation loss and bits per character (BPC) for each model.*
