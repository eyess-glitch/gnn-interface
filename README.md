# Overview

A modular framework for building and training Graph Neural Networks (GNNs). It provides a high-level interface to define datasets, models, and training parameters via JSON configuration files — supporting both homogeneous and heterogeneous graphs with minimal setup.
 
# Installation

```
git clone https://github.com/yourusername/gnn-interface.git
cd gnn-interface
pip install -r requirements.txt
```

# Configuration
The framework uses a JSON configuration file to fully define:
* Data: Graph structure and loading parameters. The input graph can be homogenous, providing CSV files for node features and edges. or heterogenousem, providing multiple CSVs — one for each node type and one for each edge/relation type. Preprocessing handles categorical features where needed.

* Model: Model type and its internal hyperparameters.
* Training: Task type, optimizer settings, and training options.
* Evaluation: Evaluation metrics for the task.
  
Below is an example config.json file:
```
{
  "data": {
    "graph_type": "homogeneous",
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
    }
  },
  "evaluation": {
    "metrics": ["accuracy"]
  }
}

```
# Usage
Once your configuration file is ready, run training with:
```
python run.py --config path/to/config.json
```
