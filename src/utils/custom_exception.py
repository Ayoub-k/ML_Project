"""Custome Exception
"""
import sys

def error_message_details(error_msg: str, error_details: sys=sys) -> str:
    """This function retrieves information about errors.

    Args:
        error_msg (str): The error message.
        error_details (tuple, optional): The error details. Defaults to sys.exc_info().

    Returns:
        str: The formatted error message with information.
    """

    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename.split('/')[-1]
    line_number = exc_tb.tb_lineno

    err_msg = f"""
    -----------------------------------------------------------------------------------
        Error occured in file [{file_name}]
        line number [{line_number}]
        error message [{str(error_msg)}]
    -----------------------------------------------------------------------------------
    """

    return err_msg

class CustomException(Exception):
    """This class for custom exception in this project
    """

    def __init__(self, errors: object, error_details: sys=sys) -> None:
        super().__init__(errors)
        self.__error_details = error_message_details(errors, error_details)


    def __str__(self):
        return self.__error_details
