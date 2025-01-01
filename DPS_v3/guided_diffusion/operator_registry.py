"""
Operator Registry for Measurement Processes.
"""

OPERATOR_REGISTRY = {}

def register_operator(name: str):
    """
    Decorator for registering a new operator.
    """
    def decorator(cls):
        if name in OPERATOR_REGISTRY:
            raise ValueError(f"Operator '{name}' is already registered!")
        OPERATOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_operator(name: str, **kwargs):
    """
    Retrieve a registered operator by name.
    """
    if name not in OPERATOR_REGISTRY:
        raise ValueError(f"Operator '{name}' is not defined.")
    return OPERATOR_REGISTRY[name](**kwargs)
