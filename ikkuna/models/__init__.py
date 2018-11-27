from .alexnetmini import AlexNetMini
from .densenet import DenseNet
from .resnet import ResNet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import VGG

__all__ = ['AlexNetMini', 'DenseNet', 'ResNet', 'VGG']


def get_model(name, *args, **kwargs):
    '''Obtain a model instance by name.

    Parameters
    ----------
    name    :   str
                One of the class names of the models defined in this package, or ``resnet<dd>`` for
                the different ResNets

    Other arguments
    ---------------
    Further positional args and kwargs are passed to the model initialiser, except for ResNets,
    which only receive the keyword args.
    '''
    try:
        if name.startswith('ResNet'):
            model_fn = globals()[name.lower()]
            model    = model_fn(**kwargs)
        else:
            Model    = globals()[name]
            model    = Model(*args, **kwargs)
        return model
    except AttributeError:
        raise ValueError(f'Unknown model {model}')
