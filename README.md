# Conversational-Orientation-Reasoning
Official implementation for the paper "Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought" (arXiv 2025).

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yu-ti-huang/Conversational-Orientation-Reasoning.git
cd Conversational-Orientation-Reasoning
pip install -r requirements.txt
```

## Repository Structure
```bash
.
├── baseline/                 # Baseline models and scripts
├── evaluation/               # Evaluation scripts (A1, A2, A3, R2, R3, etc.)
├── training/                 # Pretrained and fine-tuned models
├── requirements.txt
├── README.md
└── LICENSE
```

## Data

All datasets are released at:  
[Conversational-Orientation-Reasoning Dataset](https://huggingface.co/datasets/yu-ti-huang/Conversational-Orientation-Reasoning)

## Training
To train the multimodal CoT model:
```bash
python training/train_step0.py
python training/train_step1.py
python training/train_step2.py
python training/train_step3.py
```
To train baselines:
```bash
python training/train_b4.py
```

## Evaluation
Run different evaluations:
```bash
# Model
python evaluation/eval_step0.py
python evaluation/eval_step1.py
python evaluation/eval_step2.py
python evaluation/eval_step3_clean.py
python evaluation/eval_step3_asr.py

# Baseline
python baseline/b1.py
python baseline/b2.py
python baseline/b3.py
python evaluation/eval_b4.py

# Ablation
python evaluation/eval_a1.py
python evaluation/eval_a2.py
python evaluation/eval_a3.py
python evaluation/eval_b4.py

# Cross-domain
python evaluation/eval_r2.py

# Referential ambiguity
python evaluation/eval_r3.py
```

## Requirements
Dependencies are listed in requirements.txt:
```bash
torch
transformers
bitsandbytes
pandas
numpy
openpyxl
accelerate
```

## License
This project is released under the MIT License.
