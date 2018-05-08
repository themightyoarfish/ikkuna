Mini-Exploration-Framework
==========================

1. What should it do?
    - Allow plugging together a model, a dataset, a set of training parameters
      and a network metric/visualization to quickly evaluate a metric on many
      models and datasets
    - Logging network performance and metrics automatically
    - Allow insights into training data (c.f. influence functions)
    - Minimal user effort for using it in a pytorch model

2. Todos:
    - Serialising everything in order to plot statistics over many runs
        . Save data alongside model description (like pth or onnx) for validity
            check
    - Add some sample metrics
    - Figure out how to fit test metrics (like test loss or accuracy) into the
      framework
    - Add a second api without context manager for explicitly configuring the
      modules to supervise

Classes
=======

Supervisor
----------
Is created with a set of classes whose creation to supervise (typically
nn.Module or subclasses). Within a with-block, creation of these objects is
tracked and they are aggregated within the supervisor. It also attaches
activation and gradient hooks to the objects and is thus notified of stuff
happening there. The supervisor aggregates Handlers for the per-module
activations and gradients and passes all activations and gradients to them
without retaining them itself. It thus acts purelay as a gateway and entry
point.

Handler
-------
A Handler subclass receives activations and gradients for the modules it is
registered for with the supervisor and decides what to do with them. Most
probably, it will save, aggregate, and visualize them.
