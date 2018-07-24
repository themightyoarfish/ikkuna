'''
.. module:: module_tree

.. moduleauthor:: Rasmus Diederichsen

This module defines the :class:`ModuleTree` class for easily traversing a module hierarchy in order
to generate unique hierarchical names for all involved modules to be used as dictinary keys.
'''
import re
from collections import defaultdict

NUMBER_REGEX = re.compile(r'\d+')


class ModuleTree(object):
    '''
    Attributes
    ----------
    _module :   torch.nn.Module
    _name   :   str
                Hierarchical name for this module
    _children   :   list(ModuleTree)
                    Children of this module. Can be empty.
    _type_counter   :   dict(int)
                        Dict for keeping track of number of child modules of each class. Used for
                        disambiguating e.g. different successive conv layers which are children of
                        the same sequential module.
    '''

    def __init__(self, module, name=None, drop_name=True, recursive=True):
        '''
        Parameters
        ----------
        module  :   torch.nn.Module
        name    :   str or None
                    If no name is given, one will be generated. If ``drop_name == True``, this
                    parameter is ignored
        drop_name   :   bool
                        Ignore the given name and set it to ``''``. Useful for dropping the root
                        name (e.g. ``alexnet``) lest it appear in every child name.
        recursive   :   bool
                        Add all :meth:`torch.nn.Module.named_children` as children to this tree
        '''
        # for the root node, it often makes sense not to include the name, since it will not add any
        # information
        if drop_name:
            name = ''
        else:
            # if the name is just the index, assume it's autogenerated by e.g. nn.Sequential and
            # make a better one
            if name is None or re.match(NUMBER_REGEX, name):
                name = module.__class__.__name__.lower()

        self._module = module
        self._name = name
        self._children = []
        self._type_counter = defaultdict(int)

        if recursive:
            named_children = list(module.named_children())

            # children could be empty
            if named_children:
                for child_name, child in named_children:
                    child_class = child.__class__
                    # again, if it's just an index, make a new name. TODO: Figure out how to
                    # deduplicate this
                    if re.match(NUMBER_REGEX, child_name):
                        class_name = child_class.__name__.lower()
                        class_index = self._type_counter[child_class]
                        child_name = f'{class_name}{class_index}'

                    self._children.append(ModuleTree(child,
                                                     name=f'{self._name}/{child_name}',
                                                     drop_name=False,
                                                     recursive=True)
                                          )
                    self._type_counter[child.__class__] += 1

    def preorder(self, depth=-1):
        '''Traverse the tree in preorder.

        Yields
        ------
        tuple(str, torch.nn.Module)
            Pairs of generated hierarchical names with their associated modules
        '''
        if not self._children:
            yield (self._name, self._module)
        else:
            if depth == 0:
                yield (self._name, self._module)
            elif depth > 0:
                depth -= 1
                for child in self._children:
                    yield from child.preorder(depth)
            else:
                for child in self._children:
                    yield from child.preorder()
