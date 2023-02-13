import os
import sys

class Exception_Handling(Exception):
    """
    Exception Handling: Handling the exception by defining the function which catches the exception and root-cause
    of the error.
    """
    def __init__(self, error_message:Exception, error_details:sys):
        super().__init__(error_message) # passing the error message to parent class
        self.error_message = Exception_Handling.get_error_details(error_message=error_message,
                            error_details=error_details)
        
    @staticmethod
    def get_error_details(error_message:Exception, error_details:sys) -> str:
        """
        error_message -> Exception object
        error_detail -> Error Object from sys module
        """
        _,_, exec_traceback = error_details.exc_info()
        error_lineno = exec_traceback.tb_frame.f_lineno
        error_filename = exec_traceback.tb_frame.f_code.co_filename

        error_message = f"Error occurred in script: {error_filename} at line number: {error_lineno} error message: {error_message}"

        return error_message

    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return Exception_Handling.__name__.str()
