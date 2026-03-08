
from src.modules.xrd.service import XRDService

class AnalysisFactory:
    """
    Factory to return the appropriate service based on technique type.
    """
    @staticmethod
    def get_service(technique_type: str):
        if "XRD" in technique_type:
            return XRDService()
        # elif "FTIR" in technique_type:
        #     return FTIRService()
        raise ValueError(f"Technique {technique_type} not supported.")
