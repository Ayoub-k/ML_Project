"""Utils ml"""
import os
import pickle
from abc import ABC, abstractmethod


# --------------------------------------------------------------Seialize Object-----------------------------------------------------------------------------------

# class ObjectSerializer(ABC):
#     """
#     Abstract class for serializing and deserializing objects.
#     """

#     @abstractmethod
#     def serialize(obj, file_path):
#         """
#         Serialize the given object and save it to the specified file path.

#         Args:
#             obj: The object to be serialized.
#             file_path (str): The file path to save the serialized object.
#         """
#         pass

#     @abstractmethod
#     def deserialize(file_path):
#         """
#         Deserialize an object from the specified file path.

#         Args:
#             file_path (str): The file path from which to deserialize the object.

#         Returns:
#             The deserialized object.
#         """
#         pass

class PickleObject:
    """Class for save models"""

    @staticmethod
    def serialized(obj, file_path):
        """Save the model as a .pkl file"""
        # Create the directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def deserialized(file_path):
        """Load the model from a .pkl file"""
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
# --------------------------------------------------------------Seialize Object-----------------------------------------------------------------------------------
