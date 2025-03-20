from abc import ABC, abstractmethod

class TaskLoss(ABC):
    """
    Classe base astratta per le funzioni di perdita.
    """
    def __init__(self, criterion):
        self.criterion = criterion

    @abstractmethod
    def compute_loss(self, out, sub_data):
        """
        Metodo astratto per calcolare la perdita.
        """
        pass