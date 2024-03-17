import datetime
from typing import Union, cast
import numpy as np


def make_timestamp(prefix: str="", suffix: str="") -> str:
    tmstamp = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
    return prefix + tmstamp + suffix


class RandomCycleIter(object):

    def __init__ (self, data_list: Union[list, np.ndarray]):
        if type(data_list) == list:
            self.data_list = data_list
        else:
            data_list = cast(np.ndarray, data_list)
            self.data_list = data_list.tolist()
            
        self.length = len(self.data_list)
        self.i = self.length - 1

    def __iter__ (self):
        return self

    def __next__ (self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            np.random.shuffle(self.data_list)

        return self.data_list[self.i]