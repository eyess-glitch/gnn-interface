import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from abc import ABC, abstractmethod
from typing import List
from .AbstractGraphBuilder import AbstractGraphBuilder


# In data path ci sono i path ai file csv che rappresentano le feature dei diversi nodi
# e poi i link tra i diversi 
class HeteroGraphBuilder(GraphBuilder):
    # ciascun data path contiene il path ad un file csv che rappresenta
    # un tipo di nodo nella rete
    def __init__(self, data_path: str]):
        for data_path in data_paths:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset '{data_path}' doesn't exist.")
        
        self.data_path = data_path


     @abstractmethod
    def create_data(self, features: dict):    
        pass
        

    
        






    

   


