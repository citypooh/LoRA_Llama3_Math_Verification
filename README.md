# Math Question Answer Verification - DL Fall 2025

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

Fine-tuning Llama-3.1-8B with LoRA for binary classification of mathematical answer correctness.

---

## 📖 Overview

**Competition Task**: Predict whether a student's answer to a math problem is correct or incorrect.

**Approach**: Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) on Llama-3.1-8B.

**Key Features**:
- Structured prompt engineering with explicit solution context
- Constrained decoding for reliable binary predictions
- Stratified sampling to maintain class balance
- 4-bit quantization for memory efficiency

**Result**: **85.8%** public leaderboard accuracy, **+13.2%** over the competition baseline.

**Full write-up**: [`docs/Midterm_Report.pdf`](docs/Midterm_Report.pdf) — ACL-format report with the complete methodology, ablations, and error analysis.

---

## 👥 Team

Midterm project for **CS-GY 6953 Deep Learning**, NYU Tandon, Fall 2025.

| Member | Notebooks |
|--------|-----------|
| David Hong (sh8348) | `David_Try_1_Runtime_Disconnected.ipynb`, `David_Try_2_Success.ipynb` |
| Byeong Heon Ahn (bha233) | `Ahn_Try_1_Success.ipynb`, `Ahn_Try_2_Not_Submitted.ipynb` |
| Yukta Kulkarni (yk3213) | `Yukta_Setup.ipynb` |

Because a single training run took 8–9 hours on an A100, we ran experiments in parallel, each sweeping a different part of the configuration space, and pooled the results. The submitted best model (85.8%) came from `David_Try_2_Success.ipynb`; this repository is the organized final version of our shared work.

The team's original working repository is at [yuktakul04/dl-midterm-f25](https://github.com/yuktakul04/dl-midterm-f25).

---

## 📊 Dataset

**Source**: [nyu-dl-teach-maths-comp](https://huggingface.co/datasets/ad6398/nyu-dl-teach-maths-comp) on Hugging Face

**Statistics**:
- Training Set: 1,000,000 samples
- Test Set: 10,000 samples
- Class Balance: 50% correct / 50% incorrect

**Features**:
| Column | Description |
|--------|-------------|
| `question` | Math problem text |
| `answer` | Student's provided answer |
| `solution` | Reference solution with reasoning |
| `is_correct` | Ground truth label (True/False) |

---

## 🎯 Experiments & Results

We conducted two major training experiments with different configurations:

### Try 1: Fast Training Configuration
- **Training Data**: 45,000 samples (90/10 train/val split)
- **LoRA Configuration**:
  - Rank: 16
  - Alpha: 32
  - Dropout: 0.05
  - Trainable Parameters: 41.9M (0.52%)
- **Training Settings**:
  - Epochs: 1
  - Learning Rate: 2e-4
  - Warmup Steps: 100
  - Max Sequence Length: 2048
- **Hardware**: Google Colab A100 GPU
- **Result**: **Public Score: 0.844 (84.4% accuracy)**

### Try 2: Enhanced Configuration ✨
- **Training Data**: 49,750 samples (99.5/0.5 train/val split)
- **LoRA Configuration**:
  - Rank: 32 ⬆️
  - Alpha: 64 ⬆️
  - Dropout: 0.05
  - Trainable Parameters: 83.9M (1.03%) ⬆️
- **Training Settings**:
  - Epochs: 2 ⬆️
  - Learning Rate: 2e-4
  - Warmup Steps: 300 ⬆️
  - Max Sequence Length: 2048
- **Hardware**: Google Colab A100 GPU
- **Result**: **Public Score: 0.858 (85.8% accuracy)** 🎉

### Performance Comparison

| Configuration | Public Score | Trainable Params | Epochs | LoRA Rank | Train Samples |
|--------------|--------------|------------------|--------|-----------|---------------|
| Try 1 | 0.844 | 41.9M (0.52%) | 1 | 16 | 45,000 |
| **Try 2** | **0.858** | **83.9M (1.03%)** | **2** | **32** | **49,750** |
| **Improvement** | **+1.4%** | **+2x** | **+2x** | **+2x** | **+10%** |

---

## 🧠 Model Architecture

### Base Model
- **Model**: Llama-3.1-8B via Unsloth
- **Quantization**: 4-bit for memory efficiency
- **Max Context Length**: 2048 tokens

### LoRA Configuration (Best)
```python
{
    "r": 32,                  # Low-rank dimension
    "lora_alpha": 64,         # Scaling factor (2×r)
    "lora_dropout": 0.05,     # Regularization
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "trainable_params": "83.9M (1.03%)"
}
```

### Prompt Engineering
```
<<SYS>>
You are a precise math answer verifier.
Given a problem, a student's answer, and a reference solution,
respond with a single digit:
- 1 if the student's answer is correct
- 0 if the student's answer is incorrect
<</SYS>>

<<USR>>
Problem: {question}
Student Answer: {answer}
Solution (reference): {solution}
Return only the digit (1 or 0).
<</USR>>

<<ASSISTANT>>
{label}
```

### Constrained Decoding
- Forces model to output only "0" or "1" tokens
- Prevents hallucinations and parsing errors
- Implemented via custom `AllowedTokensLogitsProcessor`

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Google Colab with A100 GPU (recommended)
Hugging Face account with Llama-3 access
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/citypooh/LoRA_Llama3_Math_Verification.git
cd LoRA_Llama3_Math_Verification
```

2. **Install dependencies** (in Google Colab):
```python
# Install Unsloth and dependencies
!pip install unsloth xformers
!pip install transformers==4.56.2 trl==0.22.2
```

3. **Authenticate with Hugging Face**:

Llama-3.1 is a gated model, so you need an access token. Store it in Colab Secrets
(🔑 icon in the left sidebar) as `HF_TOKEN` rather than pasting it into a cell:

```python
from google.colab import userdata
from huggingface_hub import login

login(userdata.get("HF_TOKEN"))
```

### Running Training

Open one of our notebooks in Google Colab:
- `Ahn_Try_1_Success.ipynb` - Fast training configuration
- `David_Try_2_Success.ipynb` - Best performing configuration

**Steps**:
1. Enable GPU (Runtime → Change runtime type → A100 GPU)
2. Enable Internet access
3. Add `HF_TOKEN` to Colab Secrets
4. Run all cells sequentially

**Expected Training Time**:
- Try 1: ~3-4 hours on A100
- Try 2: ~8-9 hours on A100

---

## 📁 Repository Structure

```
LoRA_Llama3_Math_Verification/
├── notebooks/
│   ├── Starter.ipynb                           # Competition starter code
│   ├── Yukta_Setup.ipynb                       # Environment / data loading setup
│   ├── Ahn_Try_1_Success.ipynb                 # Rank 16, 45K samples — 84.4%
│   ├── Ahn_Try_2_Not_Submitted.ipynb           # Later run, missed the Kaggle deadline
│   ├── David_Try_1_Runtime_Disconnected.ipynb  # Full 1M-sample attempt, Colab disconnected
│   └── David_Try_2_Success.ipynb               # Rank 32, 49.75K samples — 85.8% (best)
│
├── submissions/
│   ├── sample_sub.csv                          # Competition-provided format example
│   ├── Ahn_Submission.csv                      # Try 1 predictions
│   └── David_Submission.csv                    # Try 2 predictions (best)
│
├── docs/
│   └── Midterm_Report.pdf                      # ACL-format write-up
│
├── README.md
└── .gitignore
```

---

## 🔧 Training Configuration

### Hyperparameters (Try 2 - Best)
```python
{
    "model": "unsloth/Meta-Llama-3.1-8B",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    
    # Data
    "train_samples": 49750,
    "val_samples": 250,
    "stratified_split": True,
    
    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    
    # Training
    "num_epochs": 2,
    "learning_rate": 2e-4,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "fp16": False,
    "bf16": True,
}
```

---

## 📈 Lessons Learned

### Model Capacity Matters
Increasing LoRA rank from 16 to 32 significantly improved performance despite only adding 42M parameters.

### Data Efficiency
Using 99.5% of data for training (vs 90%) provided +1.4% accuracy improvement, demonstrating the value of maximizing training data when validation isn't used during training.

### Training Duration
2 epochs proved optimal - model continued learning without overfitting, suggesting mathematical reasoning benefits from extended training.

### Prompt Engineering
Explicit inclusion of reference solution in prompt context helped model better distinguish correct from incorrect answers.

---

## 🎯 Future Work

- **Scaling to full dataset**: Train on complete 1M samples (currently 5%)
- **Ensemble methods**: Combine multiple model checkpoints
- **Curriculum learning**: Start with simple problems, progress to complex
- **Larger LoRA ranks**: Experiment with rank 64, 128
- **Different base models**: Try Llama-3-70B or Mixtral variants
