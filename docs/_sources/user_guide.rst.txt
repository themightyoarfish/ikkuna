.. contents::

User Guide
==========

Introduction
------------

Ikkuna is a framework for supervising the training of your PyTorch models. It is
stupidly easy to use. It allows you to code your chosen metric once and then use
it from any kind of model. It also comes with a few metrics out of the box

Installation
------------

Prerequisites
.............

This package requires you to have PyTorch installed. Unfortunately, the PyPi
versions always lag behind, so you may have to compile PyTorch yourself. `Don't
worry, it is a straightforward albeit somewhat time-consuming process
<https://github.com/pytorch/pytorch#installation>`_.

You will need version at least 0.5.

.. warning::

    If you install the ``torchvision`` package after installing PyTorch from
    source, it will overwrite your PyTorch installation with an older version.
    So if you need it, install it `from source
    <https://github.com/pytorch/vision#installation>`_  as well or do it before
    installing PyTorch.

Installing the library
......................

``ikkuna`` can then be installed with ``pip``

.. code-block:: shell

    pip install ikkuna


Alternatively, run

.. code-block:: shell

    pip install git+https://github.com/Peltarion/ai_ikkuna.git#egg=ikkuna

to get the bleeding-edge version.


Reporting Issues
----------------

This project is under development and — by virtue of being a thesis project —
probably unstable and bug-ridden. Therefore, expect to encounter issues.
For reporting, please use the `issue tracker <https://github.com/Peltarion/ai_ikkuna/issues>`_.

Quickstart
---------------

Using the library is very simple. Assuming you have a PyTorch model given, like
this ConvNet

.. code-block:: python

    class AlexNetMini(torch.nn.Module):
        '''Reduced AlexNet (basically just a few conv layers with relu and
        max-pooling) which attempts to adapt to arbitrary input sizes, provided they are large enough to
        survive the strides and conv cutoffs.

        Attributes
        ---------
        features    :   torch.nn.Module
                        Convolutional module, extracting features from the input
        classifier  :   torch.nn.Module
                        Classifier with relu and dropout
        H_out   :   int
                    Output height of the classifier
        W_out   :   int
                    Output width of the classifier
        '''
        def __init__(self, input_shape, num_classes=1000):
            super(AlexNetMini, self).__init__()

            # if channel dim not present, add 1
            if len(input_shape) == 2:
                input_shape.append(1)
            H, W, C = input_shape

            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(C, 64, kernel_size=5, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Conv2d(64, 192, kernel_size=3, padding=2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            self.H_out =  H // (2 * 2 * 2)
            self.W_out =  W // (2 * 2 * 2)
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(),
                torch.nn.Linear(192 * self.H_out * self.W_out, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(2048, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), 192 * self.H_out * self.W_out)
            x = self.classifier(x)
            return x

For hooking this model up with the framework, you need only add three lines.

    #.  Add an :class:`~ikkuna.export.Exporter` object to the model, e.g. by
        passing it as a constructor parameter

        .. code-block:: python

            def __init__(self, input_shape, exporter, num_classes=1000):
                # ...

    #.  Inform the :class:`~ikkuna.export.Exporter` of the model:

        .. code-block:: python

            exporter.set_model(self)

    #.  Inform the :class:`~ikkuna.export.Exporter` of which layers to track.
        You can pass it the entire model in which case it will track everything
        recursively, or pass it individual modules.

        .. code-block:: python

            exporter.add_modules(self)
            # alternatively, only track some layers
            exporter.add_modules(self.features)

        For convenience, the following also works

        .. code-block:: python

            self.features = torch.nn.Sequential(
                exporter(torch.nn.Conv2d(C, 64, kernel_size=5, stride=2, padding=1)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                exporter(torch.nn.Conv2d(64, 192, kernel_size=3, padding=2)),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2),
                torch.nn.Conv2d(192, 192, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
    #.  Add :class:`~ikkuna.export.subscriber.Subscriber`\ s to the
        :class:`~ikkuna.export.Exporter`. They take certain parameters which
        you can look up in the documentation.

        .. code-block:: python

            # create a Subscriber which publishes the ratio between gradients
            # and weights (for each layer that has them) as a tensorboard scalar
            ratio_subscriber = RatioSubscriber(['gradients', 'weights_'],
                                               backend='tb')
            exporter.subscribe(ratio_subscriber)

There are two optional steps

    #.  You should call :meth:`~ikkuna.export.Exporter.epoch_finished()` whenever
        you've run through the training set once, at least if any of your
        :class:`~ikkuna.export.subscriber.Subscriber`\ s rely on the
        ``'epoch_finished'`` message or the epoch-local step counter.
    #.  You should inform the Exporter of the loss function in use by calling
        :meth:`~ikkuna.export.Exporter.set_loss()`, if any of your
        :class:`~ikkuna.export.subscriber.Subscriber`\ s need access to the input
        labels or the final output of the network


Details
-------

Ikkuna is a Publisher-Subscriber framework, which means that in this case, a
central authority publishes data from the training process and relays it to all
registered subscribers. This central authority is the
:class:`~ikkuna.export.Exporter` class. Internally, it replaces some of the
Model's methods with wrappers so it can be transparently informed of anything
interesting happening. It uses PyTorch hooks (see
:meth:`torch.nn.Module.register_forward_hook()` and related methods) on the
:class:`~torch.nn.Module`\ s it is tracking and the :class:`~torch.Tensor`\ s
inside.

Messages published from the :class:`~ikkuna.export.Exporter` come in two types,
the :class:`~ikkuna.export.messages.MetaMessage` for events which are not tied
to any specific module and :class:`~ikkuna.export.messages.TrainingMessage` for
those that are. All messages have a :attr:`~ikkuna.export.messages.Message.kind`
attribute, which is the topic the message is about. For
:class:`~ikkuna.export.messages.MetaMessage`\ s, the following kinds are
available:

.. literalinclude:: ../../ikkuna/export/messages.py
    :lines: 16-19

Most of these topics do not come with any data attached, but for some, the
messages :attr:`~ikkuna.export.messages.Message.data` attribute will not be
``None``, but contain :class:`~torch.Tensor`\ s.

For :class:`~ikkuna.export.messages.TrainingMessage`\ s, the following
kinds are available:

.. literalinclude:: ../../ikkuna/export/messages.py
    :lines: 21-24

These topics always come with data attached and it is an error to attempt
creating a :class:`~ikkuna.export.messages.TrainingMessage` without passing data.

Creating a new Subscriber
.........................

For adding your own metric, you must subclass
:class:`~ikkuna.export.subscriber.Subscriber` or the more specialised
:class:`~ikkuna.export.subscriber.PlotSubscriber` if the metric can be displayed
in a line plot. All you need to do is write an ``__init__`` method and override
:meth:`~ikkuna.export.subscriber.Subscriber._metric()`. Your initializer should
contain at least the following arguments:

    .. code-block:: python

        def __init__(self, kinds, tag=None, subsample=1, ylims=None, backend='tb'):

Since you'll have to create a :class:`~ikkuna.export.subscriber.Subscription`
object which represents the kind of connection to the Publisher.

A :class:`~ikkuna.export.subscriber.Subscription` object contains the
information about the topic, subsampling (maybe you want to process only every `n`-th
message) and tagging. Tags can be used to filter messages, but are currently
unused. A more specialised form is :class:`~ikkuna.export.subscriber.SynchronizedSubscription`.
This subscription takes care of synchronising topics, meaning if your Subscriber
needs several kinds of messages for each module at each time step, this class
takes care of only releasing the messages in bundles after all kinds have been
received for a module.

The :class:`~ikkuna.export.subscriber.Subscription`\ s will invoke the
:class:`~ikkuna.export.subscriber.Subscriber`\ s
:meth:`~ikkuna.export.subscriber.Subscriber._metric()` method with either single
messages, if no synchronisation is used, or
:class:`~ikkuna.export.messages.MessageBundle` objects which contain the data
for one module and all desired kinds. As an example, consider the
:class:`~ikkuna.export.subscriber.RatioSubscriber`:

.. literalinclude:: ../../ikkuna/export/subscriber/ratio.py
    :lines: 9-62

As you can see, the :class:`~ikkuna.export.subscriber.Subscriber` initialiser
takes a ``plot_config`` dictionary to pass along some information to the
visualisation backend.
