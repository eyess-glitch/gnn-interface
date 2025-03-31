# Graph Machine Learning Framework

A modular framework for building and training graph neural networks with support for both homogeneous and heterogeneous graphs.


## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/graph-ml-framework.git
cd graph-ml-framework

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

## Usage
python run.py --config path/to/config.json

Here's an example of a config.json file

<details> <summary><strong>

{
  "data": {
    "graph_type": "homogenous",
    "folder_path": "./data/cora_dataset",
    "data_loader": "cora_loader",
    "batch_size": 32,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "shuffle": true
  },
  "model": {
    "model_name": "GAT",
    "input_dim": 1433,
    "hidden_dim": 64,
    "output_dim": 7,
    "num_heads": 8,
    "dropout": 0.6,
    "num_layers": 2
  },
  "training": {
    "task": "classification",
    "epochs": 200,
    "learning_rate": 0.005,
    "weight_decay": 0.0005,
    "early_stopping": {
      "patience": 20,
      "min_delta": 0.001
    },
  },
  "evaluation": {
    "metrics": ["accuracy"]
}
</strong></summary>
