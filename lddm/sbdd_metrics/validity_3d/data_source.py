from abc import ABC, abstractmethod

class DataSource(ABC):
    
    def __init__(self,
                 name: str) -> None:
        self.name = name
    
    @abstractmethod
    def __iter__(self):
        pass