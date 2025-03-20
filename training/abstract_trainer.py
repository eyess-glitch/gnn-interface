from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from loss.loss_registry import LossRegistry
from loss.task_loss import TaskLoss

# Classe base astratta
class AbstractTrainer(ABC):
    def __init__(self,
                iterations: int,
                epochs,
                task: str,
                loss: TaskLoss=None,
                data_loader: DataLoader = None,  # DataLoader per i dati
                lr: float = 0.001):  
        """
        Inizializza il trainer astratto.

        :param task: Tipo di task (es. 'classification', 'regression', etc.)
        :param criterion: Funzione di perdita
        :param data_loader: DataLoader per il dataset
        :param lr: Learning rate per l'ottimizzatore
        :param weight_decay: Fattore di decadimento dei pesi
        :param device: Dispositivo per l'elaborazione (es. torch.device("cuda"))
        """
        self.iterations = iterations
        self.epochs = epochs
        self.task = task
        self.loss = loss
        self.data_loader = data_loader
        self.lr = lr
    
    def set_param(self, name, value):
        """
        Imposta dinamicamente un attributo della classe.
        :param name: Nome dell'attributo (stringa).
        :param value: Valore da assegnare all'attributo.
        """
        if hasattr(self, name):  # Verifica che l'attributo esista
            setattr(self, name, value)
        else:
            raise AttributeError(f"L'attributo '{name}' non esiste nella classe.")

    @abstractmethod
    def train(self, model: torch.nn.Module):
        """
        Metodo astratto per addestrare il modello.

        :param model: Modello PyTorch da addestrare
        """
        pass
