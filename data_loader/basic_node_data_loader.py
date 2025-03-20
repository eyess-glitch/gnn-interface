import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import random_node_split
from data_loader.data_loader_registry import DataLoaderRegistry

@DataLoaderRegistry.register("basic_node")
class BasicNodeDataLoader:
    """
    A class to create DataLoaders for node classification tasks.
    Splits the nodes into train, validation, and test sets and provides
    mini-batch loaders for efficient training.
    """

    def __init__(self, data: Data = None, batch_size: int = 32, num_neighbors: list = [10, 10], 
                 shuffle: bool = True, train_split: float = 0.7, val_split: float = 0.2, test_split: float = 0.1):
        """
        Initialize the NodeClassificationDataLoader.

        Parameters:
        - data (Data): The PyTorch Geometric Data object representing the graph.
        - batch_size (int): Number of nodes to include in each batch. Default is 32.
        - num_neighbors (list): Number of neighbors to sample for each hop. Default is [10, 10].
        - shuffle (bool): Whether to shuffle the nodes during training. Default is True.
        - train_split (float): The percentage of nodes to be used for training (default is 0.7).
        - val_split (float): The percentage of nodes to be used for validation (default is 0.2).
        - test_split (float): The percentage of nodes to be used for testing (default is 0.1).
        """
        # Initialize the class parameters
        self.data = None  # data is set during initialization
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.shuffle = shuffle
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

    def _split_data(self, train_split: float, val_split: float, test_split: float):
        """
        Splits the nodes into train, validation, and test sets based on the given split percentages.
        
        Parameters:
        - train_split (float): Percentage of nodes used for training.
        - val_split (float): Percentage of nodes used for validation.
        - test_split (float): Percentage of nodes used for testing.
        """
        # Calculate number of nodes
        num_nodes = self.data.num_nodes
        
        # Create indices for train, validation, and test splits
        train_size = int(train_split * num_nodes)
        val_size = int(val_split * num_nodes)
        test_size = num_nodes - train_size - val_size  # Remaining nodes go to test

        # Shuffle node indices if required
        all_indices = torch.randperm(num_nodes)
        
        self.train_mask = all_indices[:train_size]
        self.val_mask = all_indices[train_size:train_size + val_size]
        self.test_mask = all_indices[train_size + val_size:]

        # Add the masks to the data object
        self.data.train_mask = self.train_mask
        self.data.val_mask = self.val_mask
        self.data.test_mask = self.test_mask

    def set_data(self, data: Data):
        """
        Set the data for the DataLoader and perform the split.
        This method will also ensure that the split data masks are computed.
        """
        self.data = data
        # Perform the split and create the masks
        self._split_data(self.train_split, self.val_split, self.test_split)

    def get_loader(self, split: str):
        """
        Create a NeighborLoader for a specific split.

        Parameters:
        - split (str): One of 'train', 'val', or 'test'.

        Returns:
        - loader (NeighborLoader): A NeighborLoader for the given split.
        """
        if split == 'train':
            mask = self.data.train_mask
        elif split == 'val':
            mask = self.data.val_mask
        elif split == 'test':
            mask = self.data.test_mask
        else:
            raise ValueError(f"Unknown split: {split}")

        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,  # Number of neighbors for each hop
            batch_size=self.batch_size,
            input_nodes=mask,
            shuffle=(split == "train") and self.shuffle,  # Shuffle only for training
        )

