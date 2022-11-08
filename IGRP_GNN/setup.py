from setuptools import setup
from setuptools import find_packages

setup(name='IGRP_GNN',
      version='0.1',
      description='Iterative Gradient Rank Pruning for Finding Graph Lottery Ticket',
      author='Po-wei Harn, Sai Deepthi',
      author_email='harnpowei@gmail.com, saideepthi112@gmail.com',
      url='',
      download_url='',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy',
                        'matplotlib',
                        'sklearn',
                        'mlxtend',
                        'networkx',
                        'torchprofile',
#                        'dgl-cu101'
                        ],
      package_data={'IGRP_GNN': ['README.md']},
      packages=find_packages())
