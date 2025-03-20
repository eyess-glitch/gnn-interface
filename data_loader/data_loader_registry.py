import inspect

class DataLoaderRegistry:
    data_loaders = {}

    @classmethod
    def register(cls, name):
        def decorator(data_loader_class):
            cls.data_loaders[name] = data_loader_class
            return data_loader_class
        return decorator

    @classmethod
    def get_data_loader(cls, **kwargs):
        """
        Recupera un DataLoader dal registro usando il nome specificato in 'data_loader'.
        Ignora eventuali parametri extra non previsti dal costruttore.
        
        :param kwargs: Dizionario contenente i parametri del DataLoader, inclusa la chiave 'data_loader'.
        """
        # Estrai il nome del DataLoader
        if "data_loader" not in kwargs:
            raise ValueError("Il parametro 'data_loader' è obbligatorio per ottenere un DataLoader.")
        
        data_loader = kwargs.pop("data_loader")
        
        if data_loader not in cls.data_loaders:
            raise ValueError(f"DataLoader '{data_loader}' non è registrato.")
        
        # Ottieni la classe del DataLoader registrato
        data_loader_class = cls.data_loaders[data_loader]
        
        # Ottieni la firma del costruttore della classe DataLoader
        data_loader_signature = inspect.signature(data_loader_class.__init__)
        valid_params = data_loader_signature.parameters
        
        # Filtra i kwargs per includere solo i parametri validi
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Crea il DataLoader con i parametri validi
        return data_loader_class(**filtered_kwargs)
