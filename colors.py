import enum
import numpy as np


class Color(enum.Enum):

    YELLOW   = '#d9d874'
    SKYBLUE  = '#f3f3fa'
    DARKBLUE = '#254167'
    RED      = '#ee5679'
    SLATE    = '#9db0bf'

    def hex(self):
        return self.value

    def rgba(self):
        from matplotlib import colors
        return colors.to_rgba(self.value, 1)

    def __array__(self):
        return np.array(self.rgba())[:, None]

    def __add__(self, other):
        return np.array(self) + np.array(other)

    def __mul__(self, other):
        return np.array(self) * np.array(other)

    def __sub__(self, other):
        return np.array(self) - np.array(other)
