from typing import NamedTuple
import torch


class NamedModule(NamedTuple):
    module: torch.nn.Module
    name: str

    def __str__(self):
        return f'<NamedModule: {self.name}>'
