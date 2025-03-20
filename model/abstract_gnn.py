import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractGNN(nn.Module, ABC):
    """
    Classe astratta per definire una Graph Neural Network (GNN).
    Ogni classe derivata deve implementare i metodi definiti qui.
    
    Aggiunti parametri comuni per le GNN come input_dim, final_dim, numero di layer, funzione di attivazione e dropout.
    """
    
    def __init__(self, input_dim, final_dim, num_layers=2):
        """
        Inizializza i parametri classici della GNN.
        
        :param input_dim: Dimensione delle feature di input per i nodi (torch.Tensor).
        :param hidden_dim: Dimensione delle feature nei layer nascosti.
        :param final_dim: Dimensione delle feature in output (solitamente corrisponde al numero di classi per task di classificazione).
        :param num_layers: Numero di layer della rete (default è 2).
        :param activation: Funzione di attivazione da usare (default è ReLU).
        :param dropout: Tasso di dropout da applicare per prevenire overfitting (default è 0.5).
        :param learning_rate: Tasso di apprendimento (anche se solitamente gestito dall'ottimizzatore, è utile come parametro di configurazione).
        """
        super().__init__()
        
        # Parametri
        self.input_dim = input_dim
        self.final_dim = final_dim
        self.num_layers = num_layers
    
        
    @abstractmethod
    def forward(self, x, edge_index):
        """
        Metodo forward che ogni GNN deve implementare.
        
        :param x: Le feature dei nodi (torch.Tensor).
        :param edge_index: La matrice di adiacenza (torch.Tensor).
        :return: Output della rete (torch.Tensor).
        """
        pass
