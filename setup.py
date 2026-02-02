from setuptools import setup, find_packages
from typing import List

HYPNEN='-e .'
def read_requirements(file_path: str) -> List[str]:
    '''this function will return the list of requirements'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPNEN in requirements:
            requirements.remove(HYPNEN)
    return requirements

setup(
    name='mlporject',  
    version='0.1.0',
    author='Mushtaq',
    author_email='mushtaqcdac@gmail.com',
    description='A machine learning project package',
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt')
    )
