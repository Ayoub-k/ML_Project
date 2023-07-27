# dependency_injection.py

from injector import Injector
from functools import wraps


class DependencyInjectionContainer:
    """Class for injection"""
    def __init__(self):
        self.injector = Injector()

    def configure(self, modules):
        """configure"""
        for module in modules:
            self.injector.binder.install(module)

    def get(self, dependency):
        """get dependecy"""
        return self.injector.get(dependency)


container = DependencyInjectionContainer()


def inject_dependencies(*dependencies):
    """function for injection"""
    def decorator(class_obj):
        @wraps(class_obj)
        def wrapper(*args, **kwargs):
            # Inject the dependencies into the class instance
            for dependency in dependencies:
                setattr(class_obj, dependency.__name__, dependency())

            # Create an instance of the class with the injected dependencies
            instance = class_obj(*args, **kwargs)

            return instance

        return wrapper

    return decorator
