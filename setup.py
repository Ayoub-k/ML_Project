"""File for setup the project into a package
"""

from typing import List
from setuptools import setup, find_packages


MD = '-e .'

def get_requires(file_path: str) -> List[str]:
    """Get requirements from a file and return them as a list

    Args:
        file_path (str): Path to the requirements file

    Returns:
        List[str]: List of requirements
    """
    with open(file_path, encoding='utf-8') as file:
        rqs = [ln.strip() for ln in file.readlines() if ln.strip() != MD and not ln.startswith('#')]
    return rqs

setup(
    name='ml_project',
    version='1.0.0',
    description='Generic project for machine leanring',
    long_description='Read the README.md file for more details.',
    author='Ayoub Elkhad',
    author_email='ayoubelkhaddouri@gmail.com',
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
)
