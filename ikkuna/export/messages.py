from collections import namedtuple


class NetworkData(namedtuple('NetworkData', ['tag', 'seq', 'step', 'epoch', 'kind', 'module',
                                             'payload'])):
    '''
    Attributes
    ----------
    tag :   str
            The tag associated with this message
    seq :   int
            Global sequence number. This counter should not reset after each epoch.
    step    :   int
                Epoch-local sequence number (the current batch index)
    epoch   :   int
                Current epoch number
    kind    :   str
                Message kind
    module  :   ikkuna.utils.NamedModule
    payload :   torch.Tensor
    '''
