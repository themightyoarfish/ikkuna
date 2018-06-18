from collections import namedtuple


NetworkData = namedtuple('NetworkData', ['tag', 'seq', 'step', 'epoch', 'kind', 'module',
                                         'payload'])
