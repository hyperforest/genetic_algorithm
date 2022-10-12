from copy import deepcopy
from hashlib import new

import numpy as np


BINARY_CHTYPE = 'binary'
INTEGER_CHTYPE = 'integer'
REAL_CHTYPE = 'real'
PERMUTATION_CHTYPE = 'permutation'


class Mutation:
    def __init__(self) -> None:
        pass

    def __call__(self, chromosome):
        pass


class BitFlipMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        index = np.random.randint(length)
        ch.genotype[index] = 1 - ch.genotype[index]

        return ch

    def __repr__(self):
        return 'BitFlipMutation()'


class SwapMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        
        p1, p2 = np.random.choice(range(length), 2, replace=False)
        temp = ch.genotype[p1]
        ch.genotype[p1] = ch.genotype[p2]
        ch.genotype[p2] = temp

        return ch

    def __repr__(self):
        return 'SwapMutation()'

class IntegerMutation(Mutation):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        index = np.random.randint(length)

        cur_val = ch.genotype[index]
        min_val, max_val = ch.min_value, ch.max_value

        space = np.array(list(range(min_val, cur_val)) + \
            list(range(cur_val + 1, max_val)))
        new_val = np.random.choice(space)
        ch.genotype[index] = new_val
        
        return ch

    def __repr__(self):
        return 'IntegerMutation()'


class RadiusMutation(Mutation):
    def __init__(self, radius=0.1, min_value=0., max_value=1.) -> None:
        super().__init__()
        self.radius = radius
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, chromosome):
        ch = deepcopy(chromosome)
        length = len(ch.genotype)
        index = np.random.randint(length)
        value = ch.genotype[index]

        new_value_1 = max([(1. - self.radius) * value, self.min_value])
        new_value_2 = min([(1. + self.radius) * value, self.max_value])
        if new_value_1 > new_value_2:
            new_value_1, new_value_2 = new_value_2, new_value_1
        
        diff = new_value_2 - new_value_1
        if diff == 0:
            prob = [0.5, 0.5]
        else:
            prob = [(value - new_value_1) / diff, (new_value_2 - value) / diff]
            prob = np.exp(prob) / np.sum(np.exp(prob))
        
        new_value = np.random.choice([new_value_1, new_value_2], p=prob)
        ch.genotype[index] = new_value

        return ch

    def __repr__(self):
        repr = ', '.join([
            f'radius={self.radius}',
            f'min_value={self.min_value}',
            f'max_value={self.max_value}'
        ])
        repr = f'RadiusMutation({repr})'

        return repr


__all__ = {
    'bitflip': BitFlipMutation,
    'swap': SwapMutation
}


_default_mutation_by_chromosome_type = {
    BINARY_CHTYPE: BitFlipMutation,
    INTEGER_CHTYPE: SwapMutation,
    REAL_CHTYPE: SwapMutation,
    PERMUTATION_CHTYPE: SwapMutation
}


def get_mutation_method_by_name(mutation_name):
    return __all__[mutation_name]


def get_default_mutation_by_chromosome_type(chromosome_type):
    return _default_mutation_by_chromosome_type[chromosome_type]
