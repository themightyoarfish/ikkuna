#!/usr/bin/env python

from distutils.core import setup
import setuptools
import os

with open('requirements.txt', mode='r') as f:
    requirements = f.read().split()

with open('README.md') as f:
    readme = f.read()

setup(name='ikkuna',
      version='0.0.2',
      description='Ikkuna Neural Network Monitor',
      long_description=readme,
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
      keywords=['deep-learning pytorch neural-networks machine-learning'],
      packages=setuptools.find_packages('.', include=['ikkuna.*']),
      install_requires=requirements,
      entry_points={
          'ikkuna.export.subscriber': [
              'HistogramSubscriber = ikkuna.export.subscriber.histogram:HistogramSubscriber',
              'RatioSubscriber = ikkuna.export.subscriber.ratio:RatioSubscriber',
              'SpectralNormSubscriber = ikkuna.export.subscriber.spectral_norm:SpectralNormSubscriber',
              'TestAccuracySubscriber = ikkuna.export.subscriber.test_accuracy:TestAccuracySubscriber',
              'TrainAccuracySubscriber = ikkuna.export.subscriber.train_accuracy:TrainAccuracySubscriber',
              'VarianceSubscriber = ikkuna.export.subscriber.variance:VarianceSubscriber',
              'SumSubscriber = ikkuna.export.subscriber.sum:SumSubscriber',
              'NormSubscriber = ikkuna.export.subscriber.norm:NormSubscriber',
              'MeanSubscriber = ikkuna.export.subscriber.mean:MeanSubscriber',
              'ConditionNumberSubscriber = ikkuna.export.subscriber.condition:ConditionNumberSubscriber',
              'HessianEigenSubscriber = ikkuna.export.subscriber.hessian_eig:HessianEigenSubscriber',
              'MessageMeanSubscriber = ikkuna.export.subscriber.message_mean:MessageMeanSubscriber',
          ]
      },
      zip_safe=False,   # don't install egg, but source
      )
