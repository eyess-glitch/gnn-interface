from abc import ABC, abstractmethod
import torch
from torch_geometric.data import DataLoader

class AbstractEvaluator(ABC):
    """
    Classe astratta per la valutazione di un modello.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def evaluate(self, model, data_loader: DataLoader):
        """
        Metodo astratto per la valutazione del modello.

        :param model: Il modello PyTorch da valutare
        :param test_loader: DataLoader per i dati di test
        :return: Il valore della metrica di valutazione
        """
        pass
