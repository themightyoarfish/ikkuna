<p align="center">
<img src="./logo.png" alt="logo" width="100"/>
</p>

# Ikkuna
A tool for monitoring neural network training.

---

Ikkuna provides a framework for adding live training metrics to your PyTorch
model with minimal configuration. It is a PubSub framework which allows
practitioners to quickly test metrics implemented against a simple API. The
following data is provided

* Activations
* Gradients
* Weights
* Biases
* Weight updates
* Bias updates
* Metadata such as current step in the training

Subscribers consume this data and distill it into metrics. Different backends can be
used

* Matplotlib
* Tensorboard (for certain visualizations, this may require a tensorboard plugin)

With a few lines, the full power of deep learning interpretations is at your
disposal.

## Documentation
The sphinx-generated html documentation is hosted [here](https://peltarion.github.io/ai_ikkuna/).

## Working with the repository/notebooks
1. Clone the repository.
1. `cd` into the repository.
1. Tell git where to find the configuration information for the iPython Notebooks with this command: `git config --add include.path $(pwd)/.gitconfig` (The path needs to point to your root git repository where the `.gitconfig` is stored).

### Adding a new notebook
1. Create a new Jupyter Notebook.
1. Hit `Edit -> Edit Notebook Metadata`.
1. Add `"git": { "suppress_outputs": true },` as a top level element to the json metadata. This will be a notification to the git filter that we want to strip the metadata.

