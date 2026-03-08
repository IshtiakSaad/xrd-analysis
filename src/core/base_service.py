
from abc import ABC, abstractmethod

class BaseAnalysisService(ABC):
    """
    Abstract base class for all analytical modules (XRD, FTIR, etc.)
    Ensures a consistent interface for the UI router.
    """
    
    @abstractmethod
    def process(self, file_content):
        pass
    
    @abstractmethod
    def get_summary(self, result):
        pass
