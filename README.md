# Graph Machine Learning Framework

A modular framework for building and training graph neural networks with support for both homogeneous and heterogeneous graphs.

## Installation

# Clone repository
git clone https://github.com/yourusername/graph-ml-framework.git
cd graph-ml-framework

# Install dependencies
pip install -r requirements.txt

## Usage

python run.py --config path/to/your_config.json

An example of config.json file can be found below


{
  "data": {
    "graph_type": "homogenous",
    "folder_path": "./data",
    "data_loader": "default"
  },
  "model": {
    "model_name": "GCN",
    "input_dim": 64,
    "hidden_dim": 128,
    "output_dim": 10
  },
  "training": {
    "task": "classification",
    "epochs": 100,
    "learning_rate": 0.01
  }
} usage deve essere pero' una sezione a parte nella descrizione
