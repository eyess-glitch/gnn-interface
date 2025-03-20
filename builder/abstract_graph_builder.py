import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from abc import ABC, abstractmethod
from typing import List
from encoders.encoder_factory import EncoderFactory

# NOTA MOLTO BENE
# PER CONVENZIONE LA COLONNA LABEL (SE PRESENTE) DEVE ESSERE L'ULTIMA DEL DATAFRAME
class AbstractGraphBuilder(ABC):
    def __init__(self, data_path: str):
        """
        Initialize the GraphBuilder class.

        Parameters:
        - data_path (str): The path to the directory containing the CSV files representing the graph nodes and edges.

        Raises:
        - FileNotFoundError: If the specified `data_path` does not exist.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory '{data_path}' doesn't exist.")
        
        self.data_path = data_path

    def _load_node_csv(self, path: str, encoder_factory: EncoderFactory):
        """
        Load node data from a CSV file and encode its features.

        Parameters:
        - path (str): Path to the CSV file containing node information.
        - encoder_factory (EncoderFactory): Factory object for creating encoders for the node features.

        Returns:
        - x (torch.Tensor): The tensor containing the encoded node features. None if no features exist.
        - mapping (dict): A dictionary mapping original node indices to internal indices.
        """
        # Leggi il CSV in un DataFrame
        df = pd.read_csv(path)

        # Mappa gli indici originali nel CSV agli indici interni
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        xs = []  # Lista per raccogliere le colonne codificate
        for col in df.columns:
            tipo_df_colonna = df[col].dtype  # Ottieni il tipo della colonna

            # Ottieni l'encoder per il tipo della colonna usando l'encoder_factory
            encoder = encoder_factory.get_encoder(str(tipo_df_colonna))

            if encoder is not None:
                # Applica l'encoder alla colonna e aggiungila alla lista
                encoded_column = encoder(df[col])
                encoded_column = torch.tensor(encoded_column)
                xs.append(encoded_column)
            else:
                # Se non esiste un encoder, lascia la colonna invariata (potresti anche scegliere di gestirla diversamente)
                 xs.append(torch.tensor(df[col].values).reshape(-1, 1)) # Aggiungi la colonna originale

        x = torch.cat(xs, dim=-1)
        
        return x, mapping

    def build_graph(self):
        """
        Build the graph from the CSV files found in the specified directory. It reads node feature data,
        encodes them using the provided encoders, and delegates the graph construction to the `create_data` method.

        Returns:
        - data (Data)
        """
        # Get the list of files in the data path
        files = os.listdir(self.data_path)
        
        # Filter out files that contain 'links' in their name, as they represent edges
        node_data_paths = [file for file in files if "links" not in file]
        
        # Create an instance of the encoder factory
        encoder_factory = EncoderFactory()        
        features = {}

        # Process each node file to extract features and mappings
        for node_data_path in node_data_paths:
            file_name = os.path.basename(node_data_path)  # Name of the file (e.g., 'user.csv')
            file_path = os.path.join(self.data_path, node_data_path)  # Full path to the file
            x, mapping = self._load_node_csv(file_path, encoder_factory)  # Load and encode features
            features[file_name] = (x, mapping)
    
        # Call the subclass-implemented `create_data` method to construct the graph
        return self.create_data(features)

    @abstractmethod
    def create_data(self, features: dict):
        """
        Abstract method to be implemented by subclasses. Defines how the graph is constructed 
        using the processed node features and mappings.

        Parameters:
        - features (dict): A dictionary where keys are node types (file names) and values are tuples (x, mapping),
          where:
          - x (torch.Tensor): Encoded node features.
          - mapping (dict): Original-to-internal index mapping for the nodes.

        Returns:
        - HeteroData: The constructed graph as a PyG HeteroData object.
        """
        pass
