import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from .abstract_evaluator import AbstractEvaluator

@EvaluatorRegistry.register("link_prediction", "f1")
class LinkPredictionEvaluator(AbstractEvaluator):
    """
    Classe concreta per la valutazione di un modello nel task di link prediction.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
    
    def evaluate(self, model, data_loader: DataLoader):
        """
        Valuta il modello per il task di link prediction.

        :param model: Il modello PyTorch da valutare
        :param data_loader: DataLoader per i dati di test
        :return: La precisione, recall e F1 score del modello sui dati di test
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sub_data in data_loader:
                sub_data = sub_data.to(self.device)  # Sposta i dati sul dispositivo
                out = model(sub_data.x, sub_data.edge_index)  # Forward pass

                # Calcola le predizioni (probabilità) per ogni coppia di nodi (link)
                # Per link prediction, usiamo sigmoid per ottenere probabilità
                preds = torch.sigmoid(out[sub_data.edge_label_index])  # Applicazione della sigmoide
                labels = sub_data.edge_label  # Le etichette reali sono 0 o 1

                # Salviamo tutte le predizioni e le etichette per calcolare le metriche
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convertiamo le predizioni in binario (0 o 1) per calcolare le metriche
        all_preds = (torch.tensor(all_preds) > 0.5).float().numpy()

        # Calcoliamo precisione, recall e F1 score
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        return precision, recall, f1
