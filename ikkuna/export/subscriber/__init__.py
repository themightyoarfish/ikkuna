import importlib
import pkg_resources

from .subscriber import (Subscriber, Subscription, SynchronizedSubscription, PlotSubscriber,
                         CallbackSubscriber)

__all__ = ['Subscriber', 'Subscription', 'SynchronizedSubscription', 'PlotSubscriber',
           'CallbackSubscriber']

####################################################################################################
#                                             PLUGINS                                              #
####################################################################################################

# Discover all installed plugins at the entry point
subscribers = {
    entry_point.name: entry_point.load()
    for entry_point in pkg_resources.iter_entry_points('ikkuna.export.subscriber')
}
__all__.extend(list(subscribers.keys()))


# use importlib to add them to the module's properties, so that we can write
# `ikkuna.export.subscriber.CustomSubscriber`, even if the class was defined in
# `some_module.foobar.custom.py`. How else would plugins make sense?
# This seems like a hack.
for name, clazz in subscribers.items():
    locals()[clazz.__name__] = getattr(importlib.import_module(clazz.__module__), clazz.__name__)
