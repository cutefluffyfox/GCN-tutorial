from abc import ABC, abstractmethod

import numpy as np


def non_batchable_method(const_idx: int = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            obj, *args = args
            for items in zip(*(args[:const_idx] + args[const_idx + 1:] if const_idx else args)):
                if const_idx:
                    items = items[:const_idx] + (args[const_idx],) + items[const_idx + 1:]
                func(obj, *items, **kwargs)

        return wrapper

    return decorator


class BaseLayer:
    """
    Base class for any Layer. Each layer should contain:
        > forward(*args, **kwargs)
        > backward(*args, **kwargs)
    """

    cache = None  # additional variable for any sort of cache

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.array:
        """
        Forward pass through the layer.
        DO NOT CALL this method in sequential/models forward
        as it do not store additional info that is present in
        __call__ method. Please see documentation for more info.

        :param args:    batch
        :param kwargs:  additional info
        :return:        transformed data
        """
        raise NotImplementedError('Please define your forward function')

    @abstractmethod
    def backward(self, *args, **kwargs) -> tuple[np.array]:
        """
        Backward pass through the layer.
        This method is taking previous layer gradient
        and returning update grad for the next layer.
        Some additional info could be passed to boost calculations.
        Please check CachedXXXLayers for more info.

        :param args:    previous gradient
        :param kwargs:  additional info
        :return:        next gradient
        """
        raise NotImplementedError('Please define your backward function')

    def __call__(self, *args, **kwargs) -> np.array:
        """
        Wrapper for forward() function that adds additional
        functional to base function.
        Please check CachedXXXLayers for more info.

        :param args:    batch
        :param kwargs:  additional info
        :return:        transformed data
        """
        return self.forward(*args, **kwargs)


class CachedInputLayer(BaseLayer, ABC):
    """
    Base class that caches input data for faster
    backward calculation.
    """

    def __call__(self, *args, **kwargs) -> np.array:
        self.cache = args
        return self.forward(*args, **kwargs)


class CachedOutputLayer(BaseLayer, ABC):
    """
    Base class that caches output data for faster
    backward calculation.
    """

    def __call__(self, *args, **kwargs) -> np.array:
        self.cache = self.forward(*args, **kwargs)
        return args


class CachedCustomLayer(BaseLayer, ABC):
    """
    Base class that caches custom data for faster
    backward calculation. Need to define custom __call__ function
    that will save all required data in `self.cache` variable
    """

    def __call__(self, *args, **kwargs) -> np.array:
        raise NotImplementedError('Please define your custom __call__ function')
