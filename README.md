# Antigen-Aware CDR Structure Prediction with Cross-Attention Graph Neural Networks

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Medium](https://img.shields.io/badge/Medium-Blog%20Post-black)](https://medium.com/@simonesurdi/antigen-aware-antibody-cdr-structure-prediction-with-cross-attention-graph-neural-networks-bcd870b6b467)

This repository contains the code for the **Stanford CS224W (Machine Learning with Graphs)** final project exploring antigen-aware antibody CDR structure prediction using cross-attention mechanisms integrated into RefineGNN.

📖 **Read the full blog post:** [Antigen-Aware Antibody CDR Structure Prediction with Cross-Attention Graph Neural Networks](https://medium.com/@simonesurdi/antigen-aware-antibody-cdr-structure-prediction-with-cross-attention-graph-neural-networks-bcd870b6b467)

## Overview

Antibodies recognize pathogens through their Complementarity-Determining Regions (CDRs). This project extends the [RefineGNN](https://arxiv.org/abs/2110.04624) model by incorporating antigen structural information via cross-attention, hypothesizing that explicit antigen features would improve CDR structure prediction.

**Key Finding:** Contrary to expectations, antigen conditioning did not improve CDR structure prediction—and in some cases slightly degraded performance. This suggests that when training on bound-state antibody-antigen complexes, the antibody structure alone contains sufficient information to predict CDR conformations.

## Project Structure

```
cs224w_final_project/
├── structgen/                     # Core model implementation
├── data/                          # Dataset files (train/val/test splits)
├── ckpts/
│   └── antigen_aware/             # Model checkpoints
├── plots/                         # Training curves and visualizations
├── predictions/                   # Inference outputs
├── test/                          # Test scripts
│
├── CS224W_Final_projecvt (1).ipynb  # 📓 Main Colab notebook with examples
│
├── ab_train.py                    # Original RefineGNN training script
├── ab_train_with_antigen.py       # Antigen-aware training script
├── ab_train_with_antigen_final.py # Final training script with logging
├── baseline_train.py              # Baseline model training
├── inference.py                   # Run inference on test samples
├── visualize.py                   # Structure visualization utilities
├── visualize_cdr.py               # Visualize predicted CDR structures
├── plot_from_checkpoints.py       # Generate training curves from checkpoints
├── print_cdr.py                   # Print CDR sequences
├── covid_optimize.py              # COVID antibody optimization
├── fold_train.py                  # Fold-based training
├── seq_train.py                   # Sequence-based training
├── rabd_test.py                   # RosettaAntibodyDesign benchmark test
├── neut_model.py                  # Neutralization predictor model
│
├── 5dsc_chothia.pdb               # Example PDB files
├── 7d5p_chothia.pdb
├── 7d5q_chothia.pdb
├── pdb_comparison.txt             # PDB comparison results
├── training_curves_with_antigen.png
├── LICENSE
└── README.md
```

## Quick Start with Colab

The easiest way to get started is with the **Google Colab notebook**:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simone-surdi/cs224w_final_project/blob/main/CS224W_Final_projecvt%20(1).ipynb)

The notebook includes:
- Dataset inspection and exploration
- Training examples for all CDR types
- Inference and visualization
- Result analysis

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/simone-surdi/cs224w_final_project.git
cd cs224w_final_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tqdm
pip install biopython  # For PDB file handling
```

## Dataset

The dataset is derived from the **Structural Antibody Database (SAbDab)**, containing antibody-antigen complexes with complete CDR annotations.

| CDR Type | Training | Validation | Test | Total |
|----------|----------|------------|------|-------|
| CDR-H1   | ~2,600   | ~380       | ~280 | ~3,260 |
| CDR-H2   | ~2,650   | ~390       | ~290 | ~3,330 |
| CDR-H3   | ~2,700   | ~270       | ~295 | ~3,265 |

## Usage

### Inspecting Dataset Structure

To explore the structure of the `.jsonl` data files:

```python
# =============================================================================
# JSONL Data Inspector
# Recursively scans a folder for .jsonl files and prints the structure of the
# first JSON object in each file (keys, types, sample values).
# Useful for understanding the data format without loading entire files.
# =============================================================================

import os
import json

# --- CONFIGURATION ---
DATA_FOLDER_PATH = "./data/"

def inspect_jsonl_structure(source_folder):
    if not os.path.exists(source_folder):
        print(f"Error: Folder '{source_folder}' does not exist.")
        return

    print(f"Scanning '{source_folder}' for .jsonl files...\n")

    files_found = 0

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".jsonl"):
                files_found += 1
                file_path = os.path.join(root, file)

                print("=" * 80)
                print(f"FILE: {file_path}")
                print("-" * 80)

                try:
                    with open(file_path, 'r') as f:
                        first_line = f.readline()

                        if not first_line:
                            print("  [EMPTY FILE]")
                            continue

                        try:
                            data = json.loads(first_line)

                            for key, value in data.items():
                                val_str = str(value)
                                type_str = type(value).__name__

                                if isinstance(value, list):
                                    summary = f"List with {len(value)} items"
                                    if len(value) > 0:
                                        summary += f" (First: {str(value[0])[:50]}...)"
                                    print(f"  • {key} ({type_str}): {summary}")

                                elif isinstance(value, dict):
                                    summary = f"Dict with keys {list(value.keys())}"
                                    print(f"  • {key} ({type_str}): {summary}")
                                    if key == 'coords':
                                        for k, v in value.items():
                                            v_len = len(v) if isinstance(v, list) else '?'
                                            print(f"      - {k}: List of {v_len} coords")

                                elif isinstance(value, str):
                                    if len(val_str) > 100:
                                        print(f"  • {key} ({type_str}): {val_str[:100]}... [Length: {len(val_str)}]")
                                    else:
                                        print(f"  • {key} ({type_str}): {val_str}")
                                else:
                                    print(f"  • {key} ({type_str}): {value}")

                        except json.JSONDecodeError:
                            print("  [INVALID JSON] Could not parse first line.")

                except Exception as e:
                    print(f"  [ERROR READING FILE] {e}")

                print("\n")

    if files_found == 0:
        print("No .jsonl files found.")
    else:
        print(f"Done. Inspected {files_found} files.")

# --- EXECUTION ---
inspect_jsonl_structure(DATA_FOLDER_PATH)
```

### Training

**Train antigen-aware model (CDR-H1):**
```bash
# =============================================================================
# Train the antigen-aware RefineGNN model for CDR-H1 prediction.
# Uses cross-attention to incorporate antigen structural features.
# Outputs: checkpoints saved to ckpts/antigen_aware/cdr_type_1/
# =============================================================================

python ab_train_with_antigen_final.py \
    --train_path data/sabdab/hcdr1_cluster/train_data_with_antigen_unique.jsonl \
    --val_path data/sabdab/hcdr1_cluster/val_data_with_antigen_unique.jsonl \
    --test_path data/sabdab/hcdr1_cluster/test_data_with_antigen_unique.jsonl \
    --save_dir ckpts/antigen_aware/cdr_type_1 \
    --cdr_type 1 \
    --epochs 10
```

**Train antigen-aware model (CDR-H3):**
```bash
# =============================================================================
# Train the antigen-aware RefineGNN model for CDR-H3 prediction.
# CDR-H3 is the most variable loop and hardest to predict.
# Outputs: checkpoints saved to ckpts/antigen_aware/cdr_type_3/
# =============================================================================

python ab_train_with_antigen_final.py \
    --train_path data/sabdab/hcdr3_cluster/train_data_with_antigen_unique.jsonl \
    --val_path data/sabdab/hcdr3_cluster/val_data_with_antigen_unique.jsonl \
    --test_path data/sabdab/hcdr3_cluster/test_data_with_antigen_unique.jsonl \
    --save_dir ckpts/antigen_aware/cdr_type_3 \
    --cdr_type 3 \
    --epochs 10
```

### Inference

```python
# =============================================================================
# Run inference on a test sample using a trained checkpoint.
# Loads the model, predicts CDR structure, visualizes the result,
# and optionally saves predicted/ground-truth PDB files.
# =============================================================================

from inference import CDRPredictor
from visualize_cdr import visualize_inference_result

# Step 1: Load model and predict
predictor = CDRPredictor('ckpts/antigen_aware/cdr_type_3/model.best.ckpt')
loader, data = predictor.load_test_data('data/sabdab/hcdr3_cluster/test_data_with_antigen_unique.jsonl')
result = predictor.predict_sample(loader, random_sample=True)

# Step 2: Visualize (result goes directly to visualization)
view = visualize_inference_result(result)
view.show()

# Step 3 (optional): Save files
predictor.save_structures(result, save_dir='predictions/')
```

### Visualization

```bash
# =============================================================================
# Generate training curves comparing models with and without antigen conditioning.
# Plots validation PPL and RMSD across epochs for all CDR types.
# Outputs: training curve plots saved to plots/
#
# IMPORTANT: Need to have antigen_no folder, which is obtained by training
# the model removing the antigen conditioning component, adding --no_antigen 
# in the training script, for all 3 CDRs, for this script below to work.
# =============================================================================

python plot_from_checkpoints.py \
    --ckpt_dirs ckpts/antigen_aware/cdr_type_1 ckpts/antigen_no/cdr_type_1 \
                ckpts/antigen_aware/cdr_type_2 ckpts/antigen_no/cdr_type_2 \
                ckpts/antigen_aware/cdr_type_3 ckpts/antigen_no/cdr_type_3 \
    --names "CDR1 + Antigen" "CDR1 No Antigen" \
            "CDR2 + Antigen" "CDR2 No Antigen" \
            "CDR3 + Antigen" "CDR3 No Antigen" \
    --save_dir plots/
```

To generate the `antigen_no` checkpoints, train without antigen conditioning:

```bash
# Train baseline (no antigen) for CDR-H1
python ab_train_with_antigen_final.py \
    --train_path data/sabdab/hcdr1_cluster/train_data_with_antigen_unique.jsonl \
    --val_path data/sabdab/hcdr1_cluster/val_data_with_antigen_unique.jsonl \
    --test_path data/sabdab/hcdr1_cluster/test_data_with_antigen_unique.jsonl \
    --save_dir ckpts/antigen_no/cdr_type_1 \
    --cdr_type 1 \
    --epochs 10 \
    --no_antigen
```

## Results

### Quantitative Results

| Model | CDR | Val PPL | Val RMSD (Å) | Test PPL | Test RMSD (Å) |
|-------|-----|---------|--------------|----------|---------------|
| No Antigen | H1 | 7.878 | 1.367 | 6.994 | 1.012 |
| With Antigen | H1 | 7.773 | 1.339 | 6.966 | 1.077 |
| No Antigen | H2 | 7.309 | 0.911 | 7.293 | 1.113 |
| With Antigen | H2 | 7.506 | 1.085 | 7.592 | 1.102 |
| No Antigen | H3 | 9.283 | 2.135 | 9.301 | 2.310 |
| With Antigen | H3 | 9.348 | 2.214 | 9.417 | 2.330 |

### Effect of Antigen Conditioning

| CDR | No Antigen (Val RMSD) | With Antigen (Val RMSD) | Change |
|-----|----------------------|------------------------|--------|
| H1 | 1.367 Å | 1.339 Å | -2.1% |
| H2 | 0.911 Å | 1.085 Å | +19.1% |
| H3 | 2.135 Å | 2.214 Å | +3.7% |

### Training Curves

![Training Curves](training_curves_with_antigen.png)

## Key Insights

1. **Bound-state prediction works well without antigen**: The baseline RefineGNN achieves 0.9-2.1Å RMSD without explicit antigen information
2. **Implicit encoding**: The antibody framework already encodes binding interface information
3. **Negative results are informative**: Simply adding cross-attention to antigen features does not help

## References

- [RefineGNN Paper (Jin et al., ICLR 2022)](https://arxiv.org/abs/2110.04624)
- [SAbDab: Structural Antibody Database](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/)
- [Previous CS224W Project](https://medium.com/@ssyoung_58002/bc7c2c55c016)

## Citation

```bibtex
@misc{surdi2024antigencdr,
  title={Antigen-Aware CDR Structure Prediction with Cross-Attention Graph Neural Networks},
  author={Surdi, Simone},
  year={2024},
  note={Stanford CS224W Final Project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Stanford CS224W course staff
- Sonny Young and Kif Lim for the previous project foundation
- RefineGNN authors for the base implementation
- SAbDab database maintainers
