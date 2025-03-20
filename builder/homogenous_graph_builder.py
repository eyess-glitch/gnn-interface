import os
import torch
import pandas as pd
from torch_geometric.data import Data
from .abstract_graph_builder import AbstractGraphBuilder  # Assicurati che GraphBuilder sia importato correttamente

class HomogenousGraphBuilder(AbstractGraphBuilder):
    """
    Concrete implementation of GraphBuilder for building homogeneous graphs.
    It reads node feature files and a single edge relationship file from a given data directory
    and constructs a PyTorch Geometric Data object.
    """

    def __init__(self, data_path: str):
        """
        Initialize the HomogenousGraphBuilder with the path to the graph data.

        Parameters:
        - data_path (str): Path to the directory containing the node features CSV
          and the edge list CSV.

        Raises:
        - FileNotFoundError: If the specified data path does not exist.
        """
        super().__init__(data_path)

    def create_data(self, features: dict):
      """
      Create a homogeneous Data object from the provided node features and edge relationships.

      Parameters:
      - features (dict): A dictionary where the key is a file name representing a node type,
        and the value is a tuple (x, mapping), where:
          - x (torch.Tensor): Encoded node features.
          - mapping (dict): Original-to-internal index mapping for the nodes.

      Returns:
      - Data: A PyTorch Geometric Data object representing the homogeneous graph.

      Raises:
      - FileNotFoundError: If no file containing 'links' is found in the directory.
      """
      files = os.listdir(self.data_path)

      # Step 1: Filtrare i file 'links' per gli edge
      link_files = [file for file in files if "links" in file]
      if not link_files:
          raise FileNotFoundError("No file containing 'links' found in the directory.")

      # Step 2: Caricare il file degli edge (source, target)
      link_file = link_files[0]
      link_file_path = os.path.join(self.data_path, link_file)

      # Leggi il CSV degli edge
      edge_df = pd.read_csv(link_file_path)

      # Step 3: Identificare la chiave corretta per i nodi (assumendo un solo tipo di nodo)
      node_key = next(iter(features.keys()))  # Usa la prima chiave dei nodi
      node_features, node_mapping = features[node_key]

      # Step 4: Mappare gli ID sorgenti e destinazione (da 'source' e 'target') in indici interni
      source_indices = []
      for node_id in edge_df['source'].values:
          if node_id in node_mapping:
              source_indices.append(node_mapping[node_id])
          else:
              print(f"Warning: Node ID {node_id} from 'source' not found in node_mapping.")
              source_indices.append(-1)  # Usa un valore predefinito per indicare errore

      target_indices = []
      for node_id in edge_df['target'].values:
          if node_id in node_mapping:
              target_indices.append(node_mapping[node_id])
          else:
              print(f"Warning: Node ID {node_id} from 'target' not found in node_mapping.")
              target_indices.append(-1)  # Usa un valore predefinito per indicare errore

      # Step 5: Creare il tensore edge_index (eliminando gli edge con valori non validi)
      valid_edges = [(source, target) for source, target in zip(source_indices, target_indices) if source != -1 and target != -1]
      if not valid_edges:
          raise ValueError("No valid edges after mapping.")

      edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

      # Step 6: Creare la matrice delle caratteristiche dei nodi (dai dati del nodo)
      x = node_features

      # Step 7: Crea l'oggetto Data per PyTorch Geometric
      data = Data(x=x, edge_index=edge_index)

      return data

  

