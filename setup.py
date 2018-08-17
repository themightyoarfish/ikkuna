#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt', mode='r') as f:
    requirements = f.read.split()

setup(name='ikkuna',
      version='0.0.1',
      description='Ikkuna Neural Network Monitor',
      author='Rasmus Diederichsen',
      author_email='rasmus@peltarion.com',
      url='https://peltarion.github.io/ai_ikkuna/',
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering :: Visualization',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Programming Language :: Python :: 3.6',
                   'License :: OSI Approved :: MIT License',
                   'Intended Audience :: Developers',
                   ],
      keyword=['pytorch neural-networks machine-learning'],
      packages=['ikkuna'],
      install_requires=requirements)
