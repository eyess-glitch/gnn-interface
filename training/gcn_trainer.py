import torch
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from training.abstract_trainer import AbstractTrainer
from training.training_registry import TrainingRegistry
from loss.loss_registry import LossRegistry
from loguru import logger  
from loss.cross_entropy_loss_class import CrossEntropyLossClassification  # Importa qui la loss
from loss.task_loss import TaskLoss

@TrainingRegistry.register("gcn_trainer")
class GcnTrainer(AbstractTrainer):
    def __init__(self, 
                 iterations: int,
                 epochs: int,
                 task: str, 
                 loss: TaskLoss = None, 
                 data_loader: DataLoader = None, 
                 lr: float = 0.001, 
                 log_every: int = 10):
        """
        Inizializza la strategia di addestramento.
        """
        super().__init__(iterations, epochs, task, loss, data_loader, lr)  
        self.log_every = log_every  # Numero di iterazioni tra i log
        logger.add("training.log", format="{time} {level} {message}", level="INFO")  # Configurazione del logger

    def train(self, model: Module):
        """
        Esegue il ciclo di addestramento.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)  # Sposta il modello sul dispositivo
        model.train()  # Imposta il modello in modalità training

        # Se non è stato fornito un ottimizzatore specifico, usa Adam
        optimizer = Adam(model.parameters(), lr=self.lr)

        # Se la funzione di perdita non è stata fornita, caricala dal LossRegistry
        if self.loss is None:
            self.loss = LossRegistry.get_loss(self.task, "default")
            
        print(len(self.data_loader))

        # Loop over the epochs
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs} started.")
            
            # Loop over mini-batches in the data loader
            for iteration, sub_data in enumerate(self.data_loader):  # Itera su ogni mini-batch
                sub_data = sub_data.to(device)  # Sposta i dati sul dispositivo

                optimizer.zero_grad()  # Pulisce i gradienti

                out = model(sub_data.x, sub_data.edge_index)  # Forward pass

                # Calcola la perdita usando il criterio definito
                loss = self.loss.compute_loss(out, sub_data)  
                loss.backward()  # Calcola i gradienti
                optimizer.step()  # Aggiorna i parametri

                # Log della perdita ogni `log_every` iterazioni
                if iteration % self.log_every == 0:
                    logger.info(f"Iterazione {iteration}, Loss: {loss.item():.4f}")
            
            logger.info(f"Epoch {epoch + 1}/{self.epochs} completed.")

        logger.info("Training completed.")
