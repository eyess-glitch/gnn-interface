from abc import ABC, abstractmethod

class AbstractBaseEncoder(ABC):
    @abstractmethod
    def encode(self, data):
        """Metodo astratto per codificare i dati."""
        pass
