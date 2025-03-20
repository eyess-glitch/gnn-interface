import torch
from loss.task_loss import TaskLoss
from loss.loss_registry import LossRegistry

@LossRegistry.register("classification", "default")
class CrossEntropyLossClassification(TaskLoss):
    def __init__(self):
        """
        Inizializza la funzione di perdita Cross-Entropy per la classificazione.
        """
        super().__init__(torch.nn.CrossEntropyLoss())

    def compute_loss(self, out, sub_data):
        """
        Calcola la perdita di classificazione usando Cross-Entropy Loss.
        """
        # Assumiamo che sub_data.y contenga le etichette
        target = sub_data.y  # Estrarre le etichette da sub_data

        # Verifica se il target è di tipo long (intero)
        if target.dtype != torch.long:
            target = target.long()  # Converti in long se necessario

        # Controlla che il target e l'output siano compatibili
        if target is None:
            raise ValueError("Le etichette devono essere presenti come 'sub_data.y'.")
        
        # Calcola la perdita
        loss = self.criterion(out, target)
        
        return loss
