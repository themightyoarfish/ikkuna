<p align="center">
<img src="./ikkuna.png" alt="logo" width="100"/>
</p>

# Ikkuna
A tool for monitoring neural network training.

---

Ikkuna provides a framework for adding live training metrics to your PyTorch
model with minimal configuration. It is a plugin framework which allows
practitioners to quickly test metrics implemented against a simple API. The
following data is provided

* Activations
* Gradients
* Weights
* Biases
* Weight updates
* Bias updates
* Metadata such as current step in the training

Plugins consume this data and distill it into metrics. Different backends can be
used

* Matplotlib
* Tensorboard (requires the plugin to be accompanied by a tensorboard plugin)

With a few lines, the full power of deep learning interpretations is at your
disposal.

## Documentation
The sphinx-generated html documentation is hosted [here](https://peltarion.github.io/ai_ikkuna/).
