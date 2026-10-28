# MCF-DILI: Multimodal Contrastive Few-Shot Learning Framework for DILI Risk Prediction

## Overview

MCF-DILI is a multimodal contrastive few-shot learning framework for multiclass prediction of Drug-Induced Liver Injury (DILI) risk. The framework integrates three complementary molecular representation modalities — graph neural networks (GNN), molecular descriptors (ECFP + MACCS), and SMILES sequence encoding (Transformer) — and employs a contrastive pretraining strategy followed by few-shot fine-tuning for robust DILI risk stratification.

## Requirements

The project requires the following dependencies:

- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Geometric (torch-geometric)
- RDKit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies using:

```
pip install torch torch-geometric rdkit scikit-learn pandas numpy matplotlib seaborn
```

## Project Structure

```
MCF-DILI/
├── main.py              # Main entry: fine-tuning and evaluation pipeline
├── parsers.py           # Argument parser: all hyperparameters and paths configuration
├── dataset.py           # Dataset loading: SMILES tokenization, molecular graph construction
├── model.py             # Model architecture: TotalEncoder, FineTuneModel, pretrain/finetune steps
├── pretrain.py          # Pretraining model loading and training logic
├── use_model.py         # Inference/prediction using the fine-tuned model
└── utils.py             # Utilities: augmentations, contrastive loss, attention fusion, metrics
```

### File Descriptions

- **main.py** — Orchestrates the fine-tuning and evaluation workflow. Loads the pre-trained encoder, processes test data (support/query sets), runs fine-tuning with contrastive learning, and computes evaluation metrics over multiple random runs.

- **parsers.py** — Defines all command-line arguments including dataset paths, model architecture hyperparameters, training settings, and output directories. This is the primary configuration file.

- **dataset.py** — Implements data loading and preprocessing:
  - `DynamicSmilesTokenizer`: Custom SMILES tokenizer with dynamic vocabulary building
  - `smiles_to_graph`: Converts RDKit molecules to PyTorch Geometric Data objects (atom features, bond types, ECFP/MACCS descriptors)
  - `MoleculeDataset` / `MoleculeDataset_`: PyTorch Geometric InMemoryDataset classes for loading and processing molecular data

- **model.py** — Defines the neural network architecture:
  - `TotalEncoder`: Multi-modal encoder combining GAT-based graph encoder, descriptor encoder, and SMILES Transformer encoder
  - `FineTuneModel`: Fine-tuning model with attention fusion (AttentionFusion) and task-specific classifiers
  - `pretrain_step`: Contrastive pretraining loop (cross-modal contrastive loss: Graph-Descriptor, Graph-SMILES, Descriptor-SMILES)
  - `finetune_step`: Few-shot fine-tuning loop with combined classification + contrastive loss

- **pretrain.py** — Handles pretraining model loading (from saved checkpoint) or training from scratch, and manages data preloading via pickle caching.

- **use_model.py** — Standalone inference script: loads a fine-tuned model and generates predictions on new data.

- **utils.py** — Utility functions and modules:
  - Data augmentation: `graph_augment` (edge dropout), `smiles_augment` (token masking), `desc_augment` (Gaussian noise)
  - Contrastive losses: `SupConLoss` (Supervised Contrastive Loss), `InfoNCE`
  - Attention modules: `AttentionFusion` (multi-head attention fusion), `AttentionPrototype`
  - Evaluation: `evaluate`, `evaluate_multi`, `evaluate_total` (accuracy, AUC, F1, precision, recall, confusion matrix)
  - Visualization: `visualize_tsne` (t-SNE embedding visualization)

## Configuration

All hyperparameters and paths are configured in **parsers.py** via command-line arguments. Key parameters include:

### Data Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./datasets` | Root directory containing dataset files |
| `--preload_data_path` | `./datasets/preload_data` | Directory for preloaded (cached) data |
| `--pretrain_model_save_path` | `output` | Directory to save pretraining model checkpoints |
| `--model_save_path` | `output` | Directory to save fine-tuned model checkpoints |
| `--model_load_path` | `output` | Directory to load model checkpoints from |
| `--result_path` | `output` | Directory to save evaluation results |

### Datasets

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_dataset` | `tox21` | Training dataset name |
| `--test_dataset` | `DILIconcern` | Test/evaluation dataset name |
| `--preload_train_data` | `True` | Whether to preload and cache training data |

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--in_channels` | `7` | Number of atom input features |
| `--hidden_channels` | `512` | Hidden layer dimension |
| `--embedding_size` | `128` | Embedding dimension |
| `--desc_channels` | `2215` | Molecular descriptor dimension (ECFP 2048 + MACCS 167) |
| `--sequence_channels` | `128` | SMILES sequence input dimension (vocab size) |
| `--concern_channels` | `128` | Concern channel dimension |
| `--dropout` | `0.1` | Dropout rate |
| `--head` | `4` | Number of attention heads |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr_pretrain` | `1e-3` | Learning rate for pretraining |
| `--lr_finetune` | `1e-5` | Learning rate for fine-tuning |
| `--temperature` | `0.2` | Temperature for contrastive loss |
| `--contrastive_weight` | `0.1` | Weight for contrastive loss in fine-tuning |
| `--epochs` | `500` | Maximum pretraining epochs |
| `--train_epochs` | `300` | Actual pretraining epochs |
| `--fine_tune_epochs` | `1000` | Maximum fine-tuning epochs |
| `--fine_tune_patience` | `50` | Patience for early stopping during fine-tuning |
| `--patience` | `20` | Patience for pretraining early stopping |
| `--batch_size` | `8` | Batch size |
| `--mean_times` | `5` | Number of independent runs for averaging |
| `--train_shot` | `4000` | Number of support samples for training |
| `--test_n_shot` | `20` | Number of support samples for testing (few-shot budget) |
| `--test_n_query` | `80` | Number of query samples for testing |
| `--test_tasks` | `3` | Number of test tasks |
| `--train_tasks` | `12` | Number of training tasks |
| `--num_classes` | `2` | Number of classes per task |
| `--max_length` | `256` | Maximum SMILES sequence length |
| `--device` | `1` | CUDA device index |
| `--num_threads` | `5` | Number of CPU threads |
| `--seed` | `42` | Random seed |

## Usage

### Step 1: Configure Parameters in parsers.py

Open **parsers.py** and modify the following settings according to your environment:

1. **Data paths**: Set `--data_dir` to point to your dataset directory. Ensure the directory contains the required dataset files (SMILES, labels, etc.).

2. **Device**: Set `--device` to the CUDA device index you want to use (e.g., `0` for GPU 0, `1` for GPU 1). Set to `-1` for CPU.

3. **Dataset names**: Adjust `--train_dataset` and `--test_dataset` if using different datasets.

4. **Hyperparameters**: Modify learning rates, batch size, epochs, contrastive weight, etc. as needed for your experiment.

5. **Output paths**: Change `--result_path`, `--model_save_path`, etc. to specify where results and checkpoints should be saved.

### Step 2: Run main.py for Pretraining + Fine-tuning + Evaluation

Execute the main script:

```
python main.py
```

The script will:
1. Load or pretrain the `TotalEncoder` model (contrastive pretraining across modalities)
2. Load test data (with support/query split for few-shot evaluation)
3. Fine-tune the `FineTuneModel` for each configuration in `in_con_ls`
4. Evaluate performance over `--mean_times` (default 5) independent runs
5. Save results (metrics CSV, prediction files, logs) to the result directory

**Example with custom parameters:**

```
python main.py --device 0 --batch_size 16 --lr_finetune 5e-5 --test_n_shot 10 --mean_times 5
```

### Step 3: (Optional) Run use_model.py for Inference

To make predictions on new data using a fine-tuned model:

```
python use_model.py
```

This script:
1. Loads the pre-trained encoder (or trains if not available)
2. Loads a fine-tuned model checkpoint (specified in use_model.py)
3. Runs inference on the specified dataset
4. Saves predictions to CSV

**Note:** You need to update the model path in use_model.py to point to your fine-tuned checkpoint:
```python
model.load_state_dict(torch.load('result/model_state_dict.joblib'))
```

## Output Files

After running main.py, the following files are generated in the result directory:

- **total_metrics_final_*.csv** — Evaluation metrics (accuracy, AUC, F1, precision) for each run
- **total_metrics_*.csv** — Per-epoch metrics during fine-tuning
- **record_*.txt** — Detailed training logs with hyperparameters and metrics
- **model_state_dict_*.joblib** — Saved model checkpoints

The output directory structure follows this pattern:

```
output/
└── DILIconcern/
    └── without_ap/True_True_True_True_noseverity/
        ├── total_metrics_final_1.csv
        ├── total_metrics_final_2.csv
        ├── ...
        ├── record_<timestamp>.txt
        └── model_state_dict_1.joblib
```

## Notes

- **CUDA availability**: The code sets `os.environ['CUDA_LAUNCH_BLOCKING'] = '1'` for debugging. Remove or set to `'0'` in production for better performance.
- **Data preprocessing**: The first run will process and cache data as pickle files. Subsequent runs load from cache for faster execution.
- **Reproducibility**: Set `--seed` to control random seeds. Results are averaged over `--mean_times` independent runs.
- **Memory**: The GAT encoder with `hidden_channels=512` and `embedding_size=128` requires significant GPU memory. Reduce these values or batch size if encountering OOM errors.
- **File naming**: Note that the argument parser file is named `parsers.py` (with 's'), not `parser.py`.
