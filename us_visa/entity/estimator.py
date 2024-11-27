from us_visa.exception import CustomException
from us_visa.logger import logging

class TargetValueMapping:
    def __init__(self):
        logging.info(f"Entered in {self.__class__.__name__} class")
        self.Certified: int = 1
        self.Denied: int = 0
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        logging.info(f"Exit from {self.__class__.__name__} class after mapping")
        return dict(zip(mapping_response.values(), mapping_response.keys()))