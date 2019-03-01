#!/usr/bin/env python

from distutils.core import setup
import setuptools
import os
import ikkuna
import warnings

# read the requirements from requirements.txt. Some specs need special handing as `install_requires`
# understands only a subset of what `pip install` understands
with open('requirements.txt', mode='r') as f:
    requirements = set(f.readlines())
    pip_pkgs     = set(r for r in requirements if not '@' in r)
    repositories = requirements.difference(pip_pkgs)

# we use the readme as the description to display on pypi
with open('README.md') as f:
    readme = f.read()

# these are the entry_points which always work
entry_points = {
          'ikkuna.export.subscriber': [
              'HistogramSubscriber = ikkuna.export.subscriber.histogram:HistogramSubscriber',
              'RatioSubscriber = ikkuna.export.subscriber.ratio:RatioSubscriber',
              'SpectralNormSubscriber = ikkuna.export.subscriber.spectral_norm:SpectralNormSubscriber',
              'TestAccuracySubscriber = ikkuna.export.subscriber.test_accuracy:TestAccuracySubscriber',
              'TrainAccuracySubscriber = ikkuna.export.subscriber.train_accuracy:TrainAccuracySubscriber',
              'VarianceSubscriber = ikkuna.export.subscriber.variance:VarianceSubscriber',
              'NormSubscriber = ikkuna.export.subscriber.norm:NormSubscriber',
              'MeanSubscriber = ikkuna.export.subscriber.mean:MeanSubscriber',
              'MessageMeanSubscriber = ikkuna.export.subscriber.message_mean:MessageMeanSubscriber',
              'LossSubscriber = ikkuna.export.subscriber.loss:LossSubscriber',
              'CallbackSubscriber = ikkuna.export.subscriber.subscriber:CallbackSubscriber',
          ]
      }

# if hessian_eigenthings is installed, we can supply the HessianEigenSubscriber, otherwise simply
# drop it
try:
    import hessian_eigenthings  # noqa
    entry_points['ikkuna.export.subscriber'].append(
        'HessianEigenSubscriber = ikkuna.export.subscriber.hessian_eig:HessianEigenSubscriber'
    )
except ImportError:
    warnings.warn('pytorch-hessian-eigenthings could not be imported. '
                  '`HessianEigenSubscriber` will not be installed. You can find the package at '
                  '`https://github.com/noahgolmant/pytorch-hessian-eigenthings/`')

# Problem: PyPi does not allow the syntax introduced in Pep 508 for listing requirements from a vcs
# repository, although setuptools understands it. This is supposedly because the developers want to
# avoid PyPi packages pulling from arbitrary URLs. So, as a workaround, we manually pull from this
# arbitrary url and install the SVCCA package. In the future, SVCCA should be on PyPi as well,
# making this unnecessary, but for now it's not tested enough.
try:
    import svcca  # noqa
    entry_points['ikkuna.export.subscriber'].extend(
        [
            'SVCCASubscriber = ikkuna.export.subscriber.svcca:SVCCASubscriber',
            'BatchedSVCCASubscriber = ikkuna.export.subscriber.svcca:BatchedSVCCASubscriber',
        ]
    )
except ImportError:
    cwd = os.getcwd()
    svcca_url = 'https://github.com/themightyoarfish/svcca-gpu/archive/master.zip'
    archive_dirname = 'svcca-gpu-master'
    warnings.warn(f'SVCCA is not installed. Installing from `{svcca_url}`')
    import tempfile
    import importlib
    from urllib.request import urlretrieve
    from zipfile import ZipFile
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        print(f'Downloading `{svcca_url}`')
        fname, headers = urlretrieve(svcca_url, f'{tempdir}/svcca.zip')
        f = ZipFile(fname)
        extracted_repo_path = os.path.join(tempdir, archive_dirname)
        print(f'Extracting `{archive_dirname}`')
        f.extractall()
        os.chdir(extracted_repo_path)
        print('Running setup script.')
        module_spec = importlib.util.spec_from_file_location('setup',
                                                             os.path.join(extracted_repo_path,
                                                                          'setup.py'))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        os.chdir(cwd)
        print('Done.')

setup(name='ikkuna',
      version=ikkuna.__version__,
      description='Ikkuna Neural Network Monitor',
      long_description=readme,
      long_description_content_type='text/markdown',
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
      install_requires=list(pip_pkgs),
      entry_points=entry_points,
      zip_safe=False,   # don't install egg, but source
      )
