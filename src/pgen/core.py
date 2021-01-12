from enum import Enum, unique

class ModelNames(str, Enum):
    DEFAULT = "esm1_t34_670M_UR50S"
    T34_670M_UR50S = "esm1_t34_670M_UR50S"
    T34_670M_UR50D = "esm1_t34_670M_UR50D"
    T34_670M_UR100 = "esm1_t34_670M_UR100"
    T12_85M_UR50S = "esm1_t12_85M_UR50S"
    T6_43M_UR50S = "esm1_t6_43M_UR50S"
    T33_650M_UR50S = "esm1b_t33_650M_UR50S"

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]