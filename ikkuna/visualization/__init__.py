from .backend import TBBackend, MPLBackend, Backend

__all__ = ['TBBackend', 'MPLBackend', 'Backend']

backend_choices = ('tb', 'mpl')


def get_backend(name, plot_config, **kwargs):
    if name not in backend_choices:
        raise ValueError(f'Backend must be one of {backend_choices}')
    if name == 'tb':
        return TBBackend(**plot_config, **kwargs)
    if name == 'mpl':
        return MPLBackend(**plot_config, **kwargs)
