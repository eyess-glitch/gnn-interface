import inspect

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get_model(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Modello '{name}' non trovato nel registry.")
        
        # Ottieni il costruttore del modello
        model_class = cls._registry[name]
        model_signature = inspect.signature(model_class.__init__)
        valid_params = model_signature.parameters
        
        # Filtra i kwargs per includere solo i parametri validi
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Crea il modello con i parametri filtrati
        return model_class(**filtered_kwargs)
