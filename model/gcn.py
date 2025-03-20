import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from model.abstract_gnn import AbstractGNN
from model.model_registry import ModelRegistry

@ModelRegistry.register("gcn")
class Gcn(AbstractGNN):
    def __init__(self, input_dim, hidden_dim, final_dim, num_layers):
        # Chiamata al costruttore della classe base
        super().__init__(input_dim, final_dim, num_layers)
        
        self.hidden_dim = hidden_dim
        torch.manual_seed(12345)
        
        # Creazione della lista dei layer convoluzionali GCN (uno per ogni layer)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))  # Primo layer

        # Aggiungere layer nascosti se num_layers > 2
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Ultimo layer per la dimensione finale
        self.convs.append(GCNConv(hidden_dim, final_dim))
        
        # Funzione di attivazione ReLU
        self.activation = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # Passaggio attraverso i layer convoluzionali
        x = x.to(torch.float32)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation(x)  # Applicazione della funzione di attivazione ReLU
        
        # Applicazione della Softmax sull'output finale (lungo la dimensione delle classi)
        x = F.softmax(x, dim=1)  # Softmax lungo la dimensione delle classi (dim=1)
        
        return x
