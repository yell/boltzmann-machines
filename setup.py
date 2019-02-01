from setuptools import setup
from setuptools import find_packages

setup(name='boltzmann_machines',
      version='0.1',
      url='https://github.com/monsta-hd/boltzmann-machines',
      install_requires=[ 'tensorflow-gpu>=1.3.0', 
                        'scipy>=0.17', 
                        'keras>=2.0.8', 
                        'matplotlib>=1.5.3', 
                        'nose-exclude>=0.5.0', 
                        'nose>=1.3.4', 
                        'numpy>=1.13.0',
                        'scikit-learn>=0.19.0',
                        'seaborn>=0.8.1', 
                        'tqdm>=4.14.0' 
],
packages=find_packages())
