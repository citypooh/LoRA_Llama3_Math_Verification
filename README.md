# DL-Fall-25 Kaggle Competition: Math Answer Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tuning Large Language Models (LLMs) with LoRA to verify the correctness of student answers to math problems.

## 📖 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Project Status](#project-status)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Known Issues](#known-issues)
- [Next Steps](#next-steps)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

**Competition Task**: Binary classification to predict whether a student's answer to a math problem is correct or incorrect.

**Method**: Supervised Fine-Tuning (SFT) of pre-trained language models using LoRA (Low-Rank Adaptation) for parameter-efficient training.

**Key Features**:
- ✅ End-to-end pipeline from data loading to submission generation
- ✅ LoRA fine-tuning for efficient training
- ✅ Structured prompt engineering for math verification
- ✅ Compatible with both local (macOS/MPS) and cloud GPU environments
- ✅ Constrained generation to output only binary predictions

---

## 📊 Dataset

**Source**: [nyu-dl-teach-maths-comp](https://huggingface.co/datasets/ad6398/nyu-dl-teach-maths-comp) on Hugging Face

**Statistics**:
- **Training Set**: 1,000,000 samples
- **Test Set**: 10,000 samples
- **Class Balance**: 50% correct, 50% incorrect

**Features**:
| Column | Description |
|--------|-------------|
| `question` | The math problem text |
| `answer` | Student's provided answer |
| `solution` | Reference solution with reasoning |
| `is_correct` | Ground truth label (True/False) |

**Sample**:
```python
{
  "question": "Solve for x: 2x + 5 = 13",
  "answer": "x = 4",
  "solution": "2x + 5 = 13\n2x = 13 - 5\n2x = 8\nx = 4",
  "is_correct": True
}
```

---

## 🧠 Approach

### Model Architecture
- **Base Model**: TinyLlama-1.1B-Chat-v1.0 (prototyping) / Llama-3-8B-Instruct (production)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank (r): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - **Trainable Parameters**: ~1.1% of total model parameters

### Prompt Engineering
Structured three-part prompt format:
```
<<SYS>>
You are a precise math answer verifier.
Given a problem, a student's answer, and an optional worked solution,
respond with a single digit label without explanation:
- 1 if the student's answer is correct
- 0 if the student's answer is incorrect
<</SYS>>

<<USR>>
Problem: {question}
Student Answer: {answer}
Solution (reference reasoning): {solution}
Return only the digit (1 or 0).
<</USR>>

<<ASSISTANT>>
{label}
```

### Training Strategy
1. **Data Preprocessing**: Stratified train/validation split (95/5)
2. **Training**: Supervised Fine-Tuning with language modeling objective
3. **Inference**: Constrained generation forcing binary output (0 or 1)
4. **Optimization**: Gradient accumulation for effective larger batch sizes

---

## 📈 Project Status

### ✅ Completed
- [x] Environment setup (macOS/MPS, Linux/CUDA compatible)
- [x] Dataset loading and preprocessing
- [x] Model architecture with LoRA configuration
- [x] Prompt engineering and data formatting
- [x] Training pipeline implementation
- [x] Inference pipeline with constrained generation
- [x] Submission file generation
- [x] Initial training run completed (120 steps)

### ⚠️ In Progress
- [ ] **Debugging all-False predictions** (CRITICAL)
- [ ] Scaling to full dataset training
- [ ] Hyperparameter optimization
- [ ] Validation metrics tracking

### 🎯 To Do
- [ ] Move to GPU environment (Kaggle/Colab)
- [ ] Switch to Llama-3-8B for better performance
- [ ] Train on full 1M dataset for 2-3 epochs
- [ ] Generate predictions for all 10k test samples
- [ ] Implement ensemble methods
- [ ] Add evaluation metrics (accuracy, F1, precision, recall)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10-3.12 (3.13 has compatibility issues)
CUDA-capable GPU (recommended) or Apple Silicon Mac
8GB+ RAM minimum, 16GB+ recommended
```

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd DL_Midterm
```

2. **Install dependencies**:

**For GPU (Kaggle/Colab)**:
```bash
pip install -U "accelerate>=0.33.0" "bitsandbytes>=0.43.0" \
  "datasets>=2.20.0" "transformers>=4.44.0" "trl>=0.10.1" \
  "peft>=0.12.0" "torch>=2.0.0" "pandas>=2.1.0" "numpy>=1.26.0"
```

**For macOS/CPU** (testing only):
```bash
pip install -U "numpy>=1.26,<2.0" "pandas>=2.1.0" "datasets>=2.20.0" \
  "accelerate>=0.33.0" "transformers==4.40.2" "trl==0.9.6" \
  "peft>=0.12.0" "torch>=2.3.0"
```

3. **Authenticate with Hugging Face** (for Llama-3):
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

### Running the Notebook

**Option A: Local Testing** (macOS/limited GPU)
```bash
jupyter notebook DL_Fall25_Clean_mac_TinyLlama_SFT_to_Submission-5.ipynb
# Run cells 1-16 sequentially
```

**Option B: Kaggle** (Recommended)
1. Upload notebook to Kaggle
2. Enable GPU (T4 or better)
3. Enable Internet access
4. Add `HF_TOKEN` secret in notebook settings
5. Run all cells

---

## 📁 Repository Structure

```
DL_Midterm/
│
├── DL_Fall25_Clean_mac_TinyLlama_SFT_to_Submission-5.ipynb  # Main notebook
├── README.md                                                  # This file
├── requirements.txt                                           # Python dependencies
├── LICENSE                                                    # License file
│
├── outputs_fast/                      # Training outputs (gitignored)
│   ├── adapter/                       # LoRA adapter weights
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── submission.csv                 # Generated predictions
│   └── [checkpoints]/                 # Training checkpoints
│
├── data/                              # (Optional) Local data cache
│   └── sample_sub.csv                 # Kaggle sample submission format
│
├── docs/                              # Documentation
│   ├── HANDOFF_REPORT.md              # Detailed progress report
│   └── TROUBLESHOOTING.md             # Common issues and solutions
│
└── scripts/                           # (Future) Utility scripts
    ├── train.py                       # Standalone training script
    ├── inference.py                   # Batch inference script
    └── evaluate.py                    # Evaluation metrics
```

---

## 🔧 Requirements

### Core Dependencies
```
torch>=2.3.0
transformers>=4.40.2
datasets>=2.20.0
accelerate>=0.33.0
peft>=0.12.0
trl>=0.9.6
pandas>=2.1.0
numpy>=1.26.0,<2.0
```

### Optional (for GPU training)
```
bitsandbytes>=0.43.0  # For 4-bit quantization
```

See `requirements.txt` for complete list.

---

## 💻 Usage

### Configuration
Edit the `Config` dataclass in the notebook (Cell 7):

```python
@dataclass
class Config:
    # Model
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Or "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Training
    epochs: int = 2
    lr: float = 2e-4
    per_device_bs: int = 1
    grad_accum: int = 16
    max_len: int = 1024
    
    # Data limits (None = use all)
    TRAIN_LIMIT: int | None = 5000   # Set to None for full training
    VAL_LIMIT: int | None = 500
    TEST_LIMIT: int | None = None
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
```

### Training
```python
# Fast prototyping (5k samples, ~3 minutes on MPS)
N_TRAIN = 5000
MAX_STEPS = 120

# Full training (recommended for competition)
cfg.TRAIN_LIMIT = None  # Use all 1M samples
cfg.epochs = 2
# Expected time: 2-4 hours on T4 GPU
```

### Inference
```python
# Generate predictions for all test samples
TEST_LIMIT = None  # Generate all 10k predictions
BATCH_SIZE = 4     # Adjust based on GPU memory

# Output: outputs_fast/submission.csv
```

### Submission
```bash
# Validate submission format
python -c "
import pandas as pd
df = pd.read_csv('outputs_fast/submission.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Value distribution:\n{df.is_correct.value_counts()}')
"

# Upload submission.csv to Kaggle competition page
```

---

## 📊 Results

### Current Status (Baseline Run)

| Metric | Value | Notes |
|--------|-------|-------|
| Training Samples | 600 | Limited subset |
| Training Steps | 120 | ~3 minutes |
| Training Loss | 0.8895 → 0.7886 | ✅ Decreasing |
| Test Predictions | 1,000 | Subset only |
| **Test Accuracy** | **❌ ~0%** | **Predicting all False** |

### Expected Performance (Full Training)

| Setup | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| TinyLlama-1.1B (50k samples) | 55-65% | ~30 min |
| TinyLlama-1.1B (full dataset) | 65-72% | ~3 hours |
| Llama-3-8B (full dataset) | **75-85%** | **2-4 hours** |

---

## ⚠️ Known Issues

### 🚨 Critical Issues

1. **All-False Predictions**
   - **Status**: Active bug
   - **Impact**: Model predicts `is_correct=False` for 100% of test samples
   - **Likely Causes**:
     - Insufficient training (only 120 steps on 600 samples)
     - Sequence length too short (384 tokens) causing truncation
     - Token ID selection logic issues
   - **Next Steps**: Debug inference pipeline (see Cell 16)

### ⚠️ Compatibility Issues

2. **Python 3.13 Incompatibility**
   - **Error**: `tokenizers` package fails to build (PyO3 version mismatch)
   - **Workaround**: Use fast tokenizer (currently implemented)
   - **Recommendation**: Use Python 3.10-3.12

3. **TRL API Version Mismatch** (Cell 14)
   - **Error**: `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`
   - **Status**: Non-blocking (Cell 11 works fine)
   - **Action**: Can be removed or fixed for newer TRL versions

### 🔍 Minor Issues

4. **Jupyter History Error**
   - **Error**: `OperationalError('attempt to write a readonly database')`
   - **Impact**: None (history not saved)
   - **Fix**: Disabled in Cell 19

5. **TOKENIZERS_PARALLELISM Warning**
   - **Warning**: Fork safety warning from tokenizers
   - **Fix**: Set `TOKENIZERS_PARALLELISM=false` (implemented in Cell 19)

---

## 🎯 Next Steps

### Immediate Actions (Priority Order)

#### 1. 🔴 Debug All-False Predictions
```python
# Add debugging cell to inspect model outputs
for i in range(10):
    prompt = make_text(test_ds[i], use_solution=True, label=None)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model_inf(**inputs)
    logits = outputs.logits[0, -1]
    
    id0 = tokenizer.convert_tokens_to_ids("0")
    id1 = tokenizer.convert_tokens_to_ids("1")
    print(f"Sample {i}: logit[0]={logits[id0]:.3f}, logit[1]={logits[id1]:.3f}")
```

#### 2. 🟡 Scale Up Training
- Move to Kaggle/Colab with T4 GPU
- Switch to Llama-3-8B-Instruct
- Train on full 1M dataset for 2-3 epochs
- Increase max_length to 1024-2048 tokens

#### 3. 🟢 Improve Evaluation
- Add validation accuracy tracking during training
- Implement F1 score, precision, recall metrics
- Add confusion matrix analysis
- Monitor class balance in predictions

#### 4. 🟢 Optimize Hyperparameters
Experiment with:
- Learning rates: [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
- LoRA ranks: [16, 32, 64]
- Batch sizes: [8, 16, 32] (with gradient accumulation)
- Max lengths: [512, 1024, 1536, 2048]

#### 5. 🟢 Advanced Techniques
- Ensemble multiple model checkpoints
- Try different prompt formats
- Experiment with temperature and sampling during inference
- Add data augmentation (paraphrasing)

---

## 🤝 Contributing

### Development Workflow
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make changes and test thoroughly
3. Update documentation if needed
4. Commit: `git commit -m "Description of changes"`
5. Push: `git push origin feature/your-feature-name`
6. Create Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Update README for new features

### Testing Checklist
- [ ] Code runs without errors
- [ ] Results are reproducible (seed set)
- [ ] Submission file format is valid
- [ ] Memory usage is reasonable
- [ ] Training loss decreases
- [ ] Validation accuracy improves

---

## 📚 Resources

### Documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Tutorials
- [Fine-tuning LLMs with LoRA](https://huggingface.co/blog/lora)
- [Efficient Training on a Single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [LLM Inference Optimization](https://huggingface.co/docs/transformers/llm_tutorial_optimization)

### Competition
- [Kaggle Competition Page](https://www.kaggle.com/competitions/dl-fall-25)
- [Dataset on Hugging Face](https://huggingface.co/datasets/ad6398/nyu-dl-teach-maths-comp)
- [Leaderboard](https://www.kaggle.com/competitions/dl-fall-25/leaderboard)

---

## 🙏 Acknowledgments

- **Course**: Deep Learning - Fall 2025
- **Competition Host**: NYU / Kaggle
- **Base Models**: 
  - [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) by TinyLlama team
  - [Llama-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) by Meta AI
- **Libraries**: Hugging Face Transformers, PEFT, TRL

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email/Team Contact]
- Course Forum: [Link if applicable]

---

## 🔖 Citation

If you use this code in your work, please cite:

```bibtex
@misc{dl-fall25-math-verification,
  title={Math Answer Verification using Fine-tuned Language Models},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/dl-fall25-kaggle}
}
```

---

**Last Updated**: October 27, 2025  
**Status**: 🟡 In Development  
**Competition Deadline**: [Add deadline if known]

---

## 📌 Quick Links

- [📓 Main Notebook](DL_Fall25_Clean_mac_TinyLlama_SFT_to_Submission-5.ipynb)
- [📊 Handoff Report](docs/HANDOFF_REPORT.md)
- [🐛 Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [🎯 Project Roadmap](https://github.com/yourusername/dl-fall25-kaggle/projects)
- [💬 Discussions](https://github.com/yourusername/dl-fall25-kaggle/discussions)

---

**Happy Training! 🚀**
