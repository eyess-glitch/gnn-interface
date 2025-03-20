import torch
from torch.utils.data import DataLoader
from evaluator.abstract_evaluator import AbstractEvaluator
from evaluator.evaluator_registry import EvaluatorRegistry
from loguru import logger  # Assuming you want to use Loguru for logging


@EvaluatorRegistry.register("accuracy")
class AccuracyEvaluator(AbstractEvaluator):
    """
    Classe concreta per la valutazione di un modello nel task di classificazione.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device
        logger.add("evaluation.log", format="{time} {level} {message}", level="INFO")  # Initialize logger

    def evaluate(self, model, data_loader: DataLoader):
        """
        Valuta il modello per il task di classificazione.

        :param model: Il modello PyTorch da valutare
        :param data_loader: DataLoader per i dati di test
        :return: L'accuratezza del modello sui dati di test
        """
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # No gradient calculation during evaluation
            for sub_data in data_loader:
                sub_data = sub_data.to(self.device)  # Sposta i dati sul dispositivo
                out = model(sub_data.x, sub_data.edge_index)  # Forward pass
                pred = out.argmax(dim=1)  # Predizione: massimo tra le classi
                correct += (pred == sub_data.y).sum().item()
                total += sub_data.y.size(0)

        accuracy = correct / total  # Calcolo dell'accuratezza

        # Log the accuracy
        logger.info(f"Accuracy: {accuracy:.4f}")

        return accuracy
