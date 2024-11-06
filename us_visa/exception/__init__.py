import sys
from us_visa.logger import logging

def error_message_details(error: str, error_detail_obj: sys) -> str:
    '''
    # exe_info() and it will give 3 info
    1. class of exception e.g. <class 'ZeroDivisionError'>
    2. exception/error message e.g. division by zero
    3. traceback memory location where object likes ['tb_frame', 'tb_lasti', 'tb_lineno', 'tb_next'] is located
    '''
    # Get the traceback information from error detail_obj
    _, _, exc_traceback = error_detail_obj.exc_info() 

    # Extract the filename from the traceback 
    file_name = exc_traceback.tb_frame.f_code.co_filename

    # Create a formatted error message 
    formated_error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_traceback.tb_lineno, str(error))

    return formated_error_message
  
class CustomException(Exception):
    def __init__(self, error_message, error_detail_obj: sys):
        super().__init__(error_message) # Call to Exception Class Constructor
        self.error_message = error_message_details(error_message, error_detail_obj)

    def __str__(self):
        return self.error_message


# if __name__ == "__main__":
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e, sys)
