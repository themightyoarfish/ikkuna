'''
.. moduleauthor:: Rasmus Diederichsen

.. module:: subscriber

This module contains the base definition for subscriber functionality. The :class:`Subscriber` class
should be subclassed for adding new metrics.

'''
import abc


class Subscription(object):

    '''Specification for a subscription that can span multiple kinds and a tag.'''

    def __init__(self, subscriber, *kinds, tag=None):
        '''
        Parameters
        ----------
        subscriber  :   Subscriber
                        Object that wants to receive the messages
        kinds   :   list(str)
                    List of string identifiers for different message kinds. These are all the
                    message kinds the subscriber wishes to receive
        tag :   str or None
                Optional tag for filtering messages. If ``None`` is passed, all messages will be
                relayed
        '''
        self._tag         = tag
        self._kinds       = {kind: False
                             for kind in kinds}
        self._subscriber  = subscriber

    def _new_message(self, network_data):
        '''Process a newly arrived message. Subclasses should override this method for any special
        treatment.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        self._subscriber(set([network_data]))

    def __call__(self, network_data):
        '''Callback for receiving an incoming message.

        Parameters
        ----------
        network_data    :   ikkuna.messages.NetworkData
        '''
        if network_data.kind not in self._kinds:
            return
        if self._tag is None or self._tag == network_data.tag:
            self._new_message(network_data)


class SynchronizedSubscription(Subscription):

    def __init__(self, subscriber, *kinds, tag=None):
        super().__init__(subscriber, *kinds, tag)
        self._current_seq = None
        self._buffer      = set()

    def _new_round(self, seq):
        '''Start a new round of buffering, clearing the previous cache and resetting the record for
        which kinds were received in this round.

        Parameters
        ----------
        seq :   int
                Sequence number for the new round

        Raises
        ------
        RuntimeError
            If not all desired kinds have been received yet in the current round
        '''
        self._current_seq = seq
        self._buffer.clear()
        # clear bits for received kinds in this round
        for kind in self._kinds.keys():
            if not self._kinds[kind]:
                raise RuntimeError(f'Did not receice message for "{kind}" before new round.')
            self._kinds[kind] = False

    def _new_message(self, network_data):
        '''Start a new round if a new sequence number is seen. If the buffer is full, all is
        published.'''
        if self._current_seq is None or self._current_seq != network_data.seq:
            self._new_round(network_data.seq)
        kind = network_data.kind
        if self._kinds[kind]:
            raise RuntimeError(f'Received mutliple messages of kind "{kind}" in one round')
        self._kinds[kind] = True
        self._buffer.add(network_data)

        # all full? publish
        if all(self._kinds.keys()):
            self._subscriber(self._buffer)


class Subscriber(abc.ABC):

    '''Base class for receiving and processing activations, gradients and other stuff into
    insightful metrics.

    Attributes
    ----------
    _counter    :   int
                    Number of times the subscriber was called
    '''

    def __init__(self):
        self._counter = 0

    @abc.abstractmethod
    def __call__(self, network_data):
        self._counter += 1
