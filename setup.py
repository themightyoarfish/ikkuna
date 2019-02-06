#!/usr/bin/env python

from distutils.core import setup
import setuptools
import os
import ikkuna

with open('requirements.txt', mode='r') as f:
    requirements = f.read()
    required_pkgs, required_repos = requirements.split('# git repos')
    required_pkgs = required_pkgs.split()
    required_repos = required_repos.split()

with open('README.md') as f:
    readme = f.read()

setup(name='ikkuna',
      version=ikkuna.__version__,
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
      keywords='deep-learning pytorch neural-networks machine-learning'.split(),
      packages=setuptools.find_packages('.', include=['ikkuna.*']),
      install_requires=required_pkgs,
      dependency_links=required_repos,
      entry_points={
          'ikkuna.export.subscriber': [
              'HistogramSubscriber = ikkuna.export.subscriber.histogram:HistogramSubscriber',
              'RatioSubscriber = ikkuna.export.subscriber.ratio:RatioSubscriber',
              'SpectralNormSubscriber = ikkuna.export.subscriber.spectral_norm:SpectralNormSubscriber',
              'TestAccuracySubscriber = ikkuna.export.subscriber.test_accuracy:TestAccuracySubscriber',
              'TrainAccuracySubscriber = ikkuna.export.subscriber.train_accuracy:TrainAccuracySubscriber',
              'VarianceSubscriber = ikkuna.export.subscriber.variance:VarianceSubscriber',
              'NormSubscriber = ikkuna.export.subscriber.norm:NormSubscriber',
              'MeanSubscriber = ikkuna.export.subscriber.mean:MeanSubscriber',
              'HessianEigenSubscriber = ikkuna.export.subscriber.hessian_eig:HessianEigenSubscriber',
              'MessageMeanSubscriber = ikkuna.export.subscriber.message_mean:MessageMeanSubscriber',
              'LossSubscriber = ikkuna.export.subscriber.loss:LossSubscriber',
              'SVCCASubscriber = ikkuna.export.subscriber.svcca:SVCCASubscriber',
              'BatchedSVCCASubscriber = ikkuna.export.subscriber.svcca:BatchedSVCCASubscriber',
          ]
      },
      zip_safe=False,   # don't install egg, but source
      )
