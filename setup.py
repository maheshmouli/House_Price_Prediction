from setuptools import setup
from typing import List

# Declaring the variables for setup functions
PROJECT_NAME = "house-price-prediction"
VERSION = "0.0.1"
AUTHOR = "MOULI SIRAMDASU"
DESCRIPTION = "Some information related to project"
PACKAGES = ['housing']
REQUIREMENTS_FILE = 'requirements.txt'

def get_requirements_list()->List[str]:
    """
    Description: This function is going to return list of requirements mentioned in 
    requirements.txt

    returns -> this contains a list which contain name of libraries
    metioned in requirements.txt file 
    """
    with open(REQUIREMENTS_FILE) as requirement_file:
        return requirement_file.readlines()
        
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=PACKAGES,
    install_requires = get_requirements_list()
)

