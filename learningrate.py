import numpy as np
import inspect
import sys

class LearningrateFactory(object):

    @staticmethod
    def build(learningrate_func):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and learningrate_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % learningrate_func)


class Linear(object):

    name = 'linear'

    @staticmethod
    def calculate(initial_lr, iteration, max_iterations):
        return initial_lr * (1/iteration)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)


class InverseTime(object):

    name = 'inverse_time'

    @staticmethod
    def calculate(initial_lr, iteration, max_iterations):
        return initial_lr * (1 - (iteration/max_iterations))

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)

    

class PowerSeries(object):

    name = 'power_series'

    @staticmethod
    def calculate(initial_lr, iteration, max_iterations):
        return initial_lr * np.exp(iteration/max_iterations)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)

          
