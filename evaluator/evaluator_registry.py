import inspect

class EvaluatorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        """
        Decorator to register an evaluator.
        """
        def decorator(evaluator_class):
            cls._registry[name] = evaluator_class
            return evaluator_class
        return decorator

    @classmethod
    def get_evaluator(cls, name, **kwargs):
        """
        Retrieves an evaluator from the registry.
        Extra parameters are ignored if not required by the constructor.
        """
        # Debugging: Check the type and value of 'name'
        print(f"Attempting to retrieve evaluator '{name}' from registry.")
        
        if not isinstance(name, str):
            raise TypeError(f"Expected 'name' to be a string, but got {type(name)}")

        if name not in cls._registry:
            raise ValueError(f"Evaluator '{name}' not found in the registry.")

        # Get the evaluator class from the registry
        evaluator_class = cls._registry[name]
        
        # Get the constructor signature
        evaluator_signature = inspect.signature(evaluator_class.__init__)
        valid_params = evaluator_signature.parameters
        
        # Filter kwargs to include only valid parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        # Instantiate the evaluator class with the filtered kwargs
        return evaluator_class(**filtered_kwargs)
