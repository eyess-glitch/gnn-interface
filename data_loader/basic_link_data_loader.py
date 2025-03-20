import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from data_loader.data_loader_registry import DataLoaderRegistry

@DataLoaderRegistry.register("basic_link")
class BasicLinkDataLoader:
    def __init__(self, data: Data = None, val_ratio=0.1, test_ratio=0.1, train_ratio=0.7, neg_sampling_ratio=2.0, batch_size=128, num_neighbors=[10, 20]):
        """
        Initialize the LinkPredictionDataLoader class.
        
        Parameters:
        - data: The original graph data (should be a PyG Data object).
        - val_ratio: The proportion of data to be used for validation.
        - test_ratio: The proportion of data to be used for testing.
        - train_ratio: The proportion of training edges for message passing (should add up with the val_ratio and test_ratio).
        - neg_sampling_ratio: The ratio of negative samples to positive samples for negative sampling.
        - batch_size: The batch size for the LinkNeighborLoader.
        - num_neighbors: List of the number of neighbors to sample at each hop for message passing.
        """
        self.data = data
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio = train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        
        # Automatically detect edge types (the ones in the data object)
        self.edge_types = list(data.edge_types)
        self.rev_edge_types = [(rev, f"rev_{rel}", head) for head, rel, rev in data.edge_types]

        # Perform the edge split
        self.train_data, self.val_data, self.test_data = self._split_data()

    def set_data(data: Data):
        self.data = data

    def _split_data(self):
        """Split the data into training, validation, and test sets."""
        transform = T.RandomLinkSplit(
            num_val=self.val_ratio,  # Validation ratio
            num_test=self.test_ratio,  # Test ratio
            disjoint_train_ratio=self.train_ratio,  # 70% training (message passing) and 30% for supervision
            neg_sampling_ratio=self.neg_sampling_ratio,  # Negative sampling ratio
            add_negative_train_samples=False,  # Negative edges are generated on the fly, not added to the graph
            edge_types=self.edge_types,  # Automatically use edge types from the data
            rev_edge_types=self.rev_edge_types,  # Reverse edge types
        )
        return transform(self.data)  # Split data into train, validation, and test

    def get_loader(self, split):
    """
    Return a LinkNeighborLoader for the specified split ('train', 'val', or 'test').
    
    Parameters:
    - split: A string indicating which data split to use ('train', 'val', or 'test').

    Returns:
    - A LinkNeighborLoader for the specified split.
    """
    # Select the appropriate dataset based on the split
    if split == "train":
        data_split = self.train_data
    elif split == "val":
        data_split = self.val_data
    elif split == "test":
        data_split = self.test_data
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'test'.")

    # Get edge information dynamically
    edge_label_index = data_split[self.edge_types[0]].edge_label_index
    edge_label = data_split[self.edge_types[0]].edge_label

    # Create and return the loader
    loader = LinkNeighborLoader(
        data=data_split,
        num_neighbors=self.num_neighbors,  # Number of neighbors to sample for each hop
        neg_sampling_ratio=self.neg_sampling_ratio,  # Negative sampling ratio
        edge_label_index=(self.edge_types[0], edge_label_index),
        edge_label=edge_label,
        batch_size=self.batch_size,  # Batch size
        shuffle=(split == "train"),  # Shuffle only for training
    )
    
    return loader


