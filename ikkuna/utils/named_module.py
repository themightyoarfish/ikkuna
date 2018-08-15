from collections import namedtuple


class NamedModule(namedtuple('NamedModule', ['module', 'name'])):
    def __str__(self):
        return self.name
