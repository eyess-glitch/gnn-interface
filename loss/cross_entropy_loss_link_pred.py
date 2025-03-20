import torch
import torch.nn as nn
from abc import ABC
from training.loss_registry import LossRegistry  # Import del registratore delle loss
from loss.task_loss import TaskLoss

@LossRegistry.register("link_prediction", "default")
class CrossEntropyLossLinkPrediction(TaskLoss):
    """
    Calcola la perdita per la link prediction usando la Binary Cross-Entropy Loss.
    """
    def __init__(self, criterion=None):
        # Usa BCEWithLogitsLoss se il criterio non è fornito
        super().__init__(nn.BCEWithLogitsLoss())

    def compute_loss(self, out, sub_data):
        """
        Calcola la Binary Cross-Entropy Loss per la link prediction.

        :param out: Logits generati dal modello per la predizione del link.
        :param sub_data: Mini-batch di dati, che contiene l'indice degli edge e le etichette di connessione.
        :return: Valore della perdita.
        """
        # La BCEWithLogitsLoss si aspetta logit, non probabilità, quindi non è necessario applicare una funzione sigmoid
        return self.criterion(out, sub_data.edge_label)
