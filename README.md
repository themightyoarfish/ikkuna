<p align="center">
<img src="https://github.com/Peltarion/ai_ikkuna/raw/master/logo/ikkuna_logo.svg" alt="logo" width="200"/>
</p>

# Ikkuna
A tool for monitoring neural network training.

---

Ikkuna provides a framework for adding live training metrics to your PyTorch
model with minimal configuration. It is a PubSub framework which allows
practitioners to quickly test metrics implemented against a simple API.

For supervising your model, you can define a metric as a subscriber which will
then be notified thorugh messages of incoming data required for computing the
metric. The subscriber is completely model-agnosting and can be installed as a
plugin for others to use.

The following data is provided

* Activations
* Gradients w.r.t weights and biases
* Gradients w.r.t layer outputs
* Weights
* Biases
* Weight updates
* Bias updates
* Metadata such as current step in the training, current labels and current
  perdictions

Subscribers consume this data and distill it into metrics. Different backends can be
used

* Matplotlib
* Tensorboard

# Working with this repository

You should create a `conda` envorinment for instance from the provided `conda_env.yaml` file and
`pip install -r` the provided `requirements.txt` file. You will also have to
install `numba` for building the documentation until I have the time to figure
out how to optionally turn off parts of a doc build.

You should also run `python setup.py develop` which will install the package
with symlinks to this repository. Since all subscribers are `setuptools` plugins, they are
not available unless `setup.py` is run.

# Documentation
The sphinx-generated html documentation is hosted [here](https://peltarion.github.io/ai_ikkuna/).
