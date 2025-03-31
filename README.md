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

```bash
# Train with default configuration
python main.py --config configs/default.json

# Train with custom configuration
python main.py --config path/to/your_config.json


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
