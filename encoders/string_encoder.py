from sentence_transformers import SentenceTransformer
from .encoders.AbstractEncoder import AbstractEncoder
import torch

class StringEncoder(AbstractEncoder):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def encode(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()