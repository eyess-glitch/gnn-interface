import inspect

class TrainingRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decoratore per registrare una strategia di addestramento.
        """
        def decorator(training_class):
            cls._registry[name] = training_class
            return training_class
        return decorator

    @classmethod
    def get_trainer(cls, name, **kwargs):
        """
        Recupera una strategia di addestramento dal registro.
        Se vengono passati parametri aggiuntivi non necessari, verranno ignorati.
        """
        if name not in cls._registry:
            raise ValueError(f"Strategia di addestramento '{name}' non trovata nel registro.")
        
        # Ottieni la classe della strategia di addestramento
        training_class = cls._registry[name]
        
        # Ottieni la firma del costruttore della classe della strategia
        training_signature = inspect.signature(training_class.__init__)
        valid_params = training_signature.parameters
        
        # Filtra i kwargs per includere solo i parametri validi
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Instanzia la classe passando solo i parametri validi
        return training_class(**filtered_kwargs)
