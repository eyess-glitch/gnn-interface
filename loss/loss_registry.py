import inspect

class LossRegistry:
    """
    Registry per gestire funzioni di perdita per task specifici.
    """
    _registry = {}  # Mappatura task -> {nome_loss: classe_loss}

    @classmethod
    def register(cls, task, loss_name):
        """
        Decoratore per registrare una nuova loss per un task specifico.
        :param task: Nome del task (es. "classification").
        :param loss_name: Nome della loss (es. "cross_entropy").
        """
        def decorator(loss_class):
            if task not in cls._registry:
                cls._registry[task] = {}
            cls._registry[task][loss_name] = loss_class
            return loss_class
        return decorator

    @classmethod
    def get_loss(cls, task, loss_name, *args, **kwargs):
        """
        Ottieni un'istanza della loss registrata.
        :param task: Nome del task.
        :param loss_name: Nome della loss.
        :param args: Parametri posizionali passati al costruttore della loss.
        :param kwargs: Parametri nominativi passati al costruttore della loss.
        """
        if task not in cls._registry or loss_name not in cls._registry[task]:
            raise ValueError(f"Loss '{loss_name}' per task '{task}' non registrata.")
        
        # Ottieni la classe della loss registrata
        loss_class = cls._registry[task][loss_name]
        
        # Ottieni la firma del costruttore della loss
        loss_signature = inspect.signature(loss_class.__init__)
        valid_params = loss_signature.parameters
        
        # Filtra i kwargs per includere solo i parametri validi per il costruttore della loss
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        # Instanzia la loss passando i parametri validi
        return loss_class(*args, **filtered_kwargs)
