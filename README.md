# Graph Machine Learning Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.3%2B-green)](https://pytorch-geometric.readthedocs.io/)

A modular framework for building and training graph neural networks with support for both homogeneous and heterogeneous graphs.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Extending the Framework](#extending-the-framework)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Functionality
- Flexible graph construction pipeline
- Support for multiple graph types (homogeneous/heterogeneous)
- Model registry system for easy model switching
- Configurable training strategies

### Technical Highlights
- Dynamic module loading system
- Automatic heterogeneous graph conversion
- Customizable data loaders
- Extensible evaluation metrics

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

### Basic Training
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
}
